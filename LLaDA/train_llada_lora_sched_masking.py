import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import argparse
import wandb
import random
from scipy.stats import beta
import numpy as np
from datasets import load_dataset
import pandas as pd
from wordfreq import word_frequency


MASK_TOKEN_ID = 126336  # [MASK] token for LLaDA

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
vocab_dict = tokenizer.get_vocab()

def create_idf_dict():
    idf_df = pd.read_csv("LLaDA/wiki_tfidf_terms.csv")
    idf_terms = set(str(t).lower() for t in idf_df['token'].dropna())

    real_words = [
        t for t in vocab_dict.keys()
        if word_frequency(t, "en") > 1e-6 or word_frequency(t[1:], "en") > 1e-6
    ]

    # Min-max normalization
    idf_df = idf_df.dropna(subset=['idf'])
    idf_min = idf_df['idf'].min()
    idf_max = idf_df['idf'].max()
    idf_df['idf_norm'] = (idf_df['idf'] - idf_min) / (idf_max - idf_min)

    # Now filter after idf_norm exists
    real_words_lower = set(t.lower() for t in real_words)
    idf_real = idf_df[idf_df['token'].str.lower().isin(real_words_lower)]
    idf_real_values_norm = idf_real['idf_norm'].dropna()
    mean_idf_norm = idf_real_values_norm.mean()

    # Build final token -> priority map
    idf_lookup = dict(zip(idf_df['token'].str.lower(), idf_df['idf_norm']))
    idf_dict = {}
    for token, token_id in vocab_dict.items():
        token_clean = token.lower().lstrip("Ġ")  # remove BPE space if any
        idf_dict[(token, token_id)] = idf_lookup.get(token_clean, mean_idf_norm)

    return idf_dict


def build_tokenid_to_priority(vocab_dict, rand_dict, default_priority=0.5):
    max_id = max(vocab_dict.values()) + 1
    table = np.full(max_id, default_priority)
    for (_, token_id), priority in rand_dict.items():
        table[token_id] = priority
    return table


def is_valid(example):
    messages = example["messages"]
    return (
        len(messages) >= 2 and
        messages[0]["role"] == "user" and
        messages[1]["role"] == "assistant" and
        all(item["content"].strip() != "" for item in messages[:2])
    )

class SFTDataset(Dataset):
    def __init__(self, tokenizer, sect, max_len=512):
        self.samples = []
        self.tokenizer = tokenizer

        # Load and filter raw TULU data
        ds = load_dataset("allenai/llama-3-tulu-v2-sft-subset")["raw"]
        print("Length before filtering: ", len(ds))
        ds = ds.filter(is_valid)
        ds = ds.filter(lambda ex: len(ex["messages"]) >= 2 
                                  and ex["messages"][0]["role"] == "user" 
                                  and ex["messages"][1]["role"] == "assistant" 
                                  and ex["messages"][1]["content"].strip() != "")

        print("Length before filtering: ", len(ds))
        if sect == "train":
            ds = ds.select(range(10000))
        else:
            ds = ds.select(range(10000, len(ds)))

        # Convert each example into tokenized input
        for ex in ds:
            prompt = ex["messages"][0]["content"]
            answer = ex["messages"][1]["content"]

            # Format with chat-style template
            full_text = (
                "<s><|startofuser|>\n" + prompt +
                "<|endofuser|><|startofassistant|>\n" + answer +
                "<|endoftext|>"
            )
            prompt_text = (
                "<s><|startofuser|>\n" + prompt +
                "<|endofuser|><|startofassistant|>\n"
            )

            input_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
            prompt_len = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.shape[1]

            pad_len = max(0, max_len - input_ids.shape[0])
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.tensor([tokenizer.eos_token_id] * pad_len)])

            self.samples.append({
                "input_ids": input_ids,
                "prompt_length": torch.tensor(prompt_len)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def timestep_schedule(step, total_steps, gamma=4.0):
    """ Returns a timestep t ~ Beta(a, b) that shifts right over time. """

    t = random.uniform(0,1)
    return t

    # rand_num = random.uniform(0,1)
    # t = 0
    # if rand_num >= 0.75:
    #     t = random.uniform(0.8, 1)
    # else:
    #     t = random.uniform(0, 1)

    # progress = step / total_steps
    # # Higher gamma = more right-shifted → more late-timestep training
    # a = 1 + gamma * progress
    # b = 1 + gamma * (1 - progress)
    # return beta.rvs(a, b)

    return t


#FIX THIS
#for 60k, 5 and 12k
#for 10k 7 and 1500
def sampling_schedule(step, total_steps, schedule_type='inv_sigmoid', k=7.0, t=1500):
    """
    Returns the probability of using the scheduled sampling logic (i.e. get_timestamp with uniform mix)
    """
    if schedule_type == 'linear':
        return max(1 - step / total_steps, 0.0)
    elif schedule_type == 'inv_sigmoid':
        sched_prob = k / (k + np.exp(step / t))
        #print(f"Scheduled probability: {sched_prob:.2f}")
        return k / (k + np.exp(step / t))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def orig_forward_process(input_ids, mask_prob, mask_token_id=MASK_TOKEN_ID):
    t = mask_prob
    p_mask = (torch.rand_like(input_ids.float()) < t)
    noisy_input = input_ids.clone()
    noisy_input[p_mask] = mask_token_id
    return noisy_input, p_mask, None

def new_forward_process(input_ids, mask_prob, tokenid_to_priority,
                        mask_token_id=MASK_TOKEN_ID, temperature=0.4, gamma=4.0):
    device = input_ids.device
    token_ids = input_ids.cpu().numpy()  # (B, L)
    priorities = tokenid_to_priority[token_ids]  # (B, L)

    use_uniform = np.random.rand() < temperature
    k = int(mask_prob * token_ids.shape[1])
    if use_uniform:
        timestamps = np.random.rand(*token_ids.shape)  # shape (B, L)
    else:
        a = 1 + gamma * priorities
        b = 1 + gamma * (1 - priorities)
        timestamps = beta.rvs(a, b, size=token_ids.shape)

    sorted_indices = np.argsort(timestamps, axis=1)  # (B, L)
    B, L = sorted_indices.shape
    row_idx = np.arange(B)[:, None]  # shape (B, 1)
    col_idx = sorted_indices[:, :k]  # shape (B, k)
    
    mask_flags = np.zeros_like(sorted_indices, dtype=bool)
    mask_flags[row_idx, col_idx] = True
        
    p_mask = torch.from_numpy(mask_flags).to(device=input_ids.device)
    noisy_input = input_ids.clone()
    noisy_input[p_mask] = mask_token_id
    return noisy_input, p_mask, sorted_indices

def forward_process(input_ids, mask_prob, step, total_steps, tokenid_to_priority, mask_token_id=MASK_TOKEN_ID, temperature=0.3):
    p = sampling_schedule(step, total_steps)
    if np.random.rand() < p:
        #print("Original Forward Process")
        noisy_input, p_mask, sorted_indices = orig_forward_process(input_ids, mask_prob, mask_token_id=mask_token_id)
    else:
        #print("New Forward Process")
        noisy_input, p_mask, sorted_indices = new_forward_process(input_ids, mask_prob, tokenid_to_priority, mask_token_id=mask_token_id, temperature=temperature)

    return noisy_input, p_mask, sorted_indices


def forward_process_answer_only(input_ids, prompt_lengths, mask_prob, step, total_steps, tokenid_to_priority, mask_token_id=MASK_TOKEN_ID, temperature=0.3):
    B, L = input_ids.shape
    device = input_ids.device

    noisy_input = input_ids.clone()
    p_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(B):
        prompt_len = prompt_lengths[b].item()
        answer_ids = input_ids[b, prompt_len:]

        # Apply forward_process only on answer part
        noisy_answer, mask_flags, _ = forward_process(
            input_ids=answer_ids.unsqueeze(0),
            mask_prob=mask_prob,
            step=step,
            total_steps=total_steps,
            tokenid_to_priority=tokenid_to_priority,
            mask_token_id=mask_token_id,
            temperature=temperature
        )

        # Replace answer portion with noisy version
        noisy_input[b, prompt_len:] = noisy_answer.squeeze(0)
        p_mask[b, prompt_len:] = mask_flags.squeeze(0)

    return noisy_input, p_mask, None


def compute_llada_sft_loss(logits, input_ids, p_mask, prompt_lengths):
    """
    Computes token-level loss only on the masked (answer) portion of the input.

    Args:
        logits (Tensor): Output logits from the model (batch_size, seq_len, vocab_size)
        input_ids (Tensor): Ground-truth token IDs (batch_size, seq_len)
        p_mask (Tensor): Mask positions (same shape as input_ids), 1 where tokens were masked
        prompt_lengths (Tensor): Lengths of prompt portion per sample (batch_size)

    Returns:
        torch.Tensor: Scalar loss averaged over masked answer tokens
    """
    device = input_ids.device

    # Compute mask for where prediction should happen
    masked_indices = p_mask.bool()  # shape: (batch, seq_len)

    # Debug: total masked tokens
    total_masked = masked_indices.sum().item()
    print(f"[DEBUG] Total masked tokens: {total_masked}")

    if total_masked == 0:
        print("[WARNING] No masked tokens found — returning zero loss.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Flatten for loss computation
    logits = logits[masked_indices]           # (num_masked, vocab)
    targets = input_ids[masked_indices]       # (num_masked,)

    # Check shapes
    print(f"[DEBUG] Logits shape: {logits.shape}, Targets shape: {targets.shape}")

    # Compute per-token cross-entropy loss
    token_loss = F.cross_entropy(logits, targets, reduction="mean")

    print(f"[DEBUG] Avg masked token loss: {token_loss.item():.4f}")
    return token_loss



def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    wandb.init(
        project="llada-lora-finetuning",
        name="llada-finetuning-scheduled-masking-unif-timesteps",
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "gradient_accumulation_steps": args.gradient_accumulation_steps
        }
    )


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"]
    ))

    idf_dict = create_idf_dict()
    tokenid_to_priority = build_tokenid_to_priority(vocab_dict, idf_dict)

    model.print_trainable_parameters()

    dataset = SFTDataset(tokenizer, sect="train", max_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    step_count = 0

    for epoch in range(args.epochs):
        total_loss = 0.0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_length"].to(device)


            # t = random.uniform(0,1)
            t = timestep_schedule(step_count, total_steps, gamma=4.0)

            noisy_batch, p_mask, _ = forward_process_answer_only(
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
                mask_prob=t,
                step=step_count,
                total_steps=total_steps,
                tokenid_to_priority=tokenid_to_priority,
                mask_token_id=MASK_TOKEN_ID
            )
            # token_positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            # prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
            # noisy_batch[prompt_mask] = input_ids[prompt_mask]

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                outputs = model(input_ids=noisy_batch)
                loss = compute_llada_sft_loss(outputs.logits, input_ids, p_mask, prompt_lengths)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            total_loss += loss.item()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                # Log useful debug info to wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/mask_prob_t": t,
                    "train/scheduled_sampling_prob": sampling_schedule(step_count, total_steps),
                    "train/step": step_count,
                    "train/epoch": epoch + (i / len(dataloader)),
                    "train/total_masked_tokens": p_mask.sum().item(),
                    "train/avg_mask_ratio": p_mask.float().mean().item()
                })


            # cleanup
            del loss, outputs
            torch.cuda.empty_cache()

        epoch_avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {epoch_avg_loss:.4f}")
        wandb.log({f"epoch_{epoch+1}/avg_loss": epoch_avg_loss})

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    print("Starting evaluation")
    model.eval()
    eval_dataset = SFTDataset(tokenizer, sect="test", max_len=args.max_seq_len)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    total_eval_loss = 0.0
    total_eval_tokens = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_length"].to(device)

            # Mask the entire answer portion (no randomness)
            p_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for b in range(input_ids.size(0)):
                p_mask[b, prompt_lengths[b]:] = 1

            noisy_input = input_ids.clone()
            noisy_input[p_mask] = MASK_TOKEN_ID

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                outputs = model(input_ids=noisy_input)
                loss = compute_llada_sft_loss(outputs.logits, input_ids, p_mask, prompt_lengths)

            total_eval_loss += loss.item() * p_mask.sum().item()
            total_eval_tokens += p_mask.sum().item()

    avg_eval_loss = total_eval_loss / total_eval_tokens if total_eval_tokens > 0 else 0.0
    print(f"[Eval] Avg loss: {avg_eval_loss:.4f}")
    wandb.log({"eval/avg_loss": avg_eval_loss})
    
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="./llada-lora-sft-masking-sched")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    args = parser.parse_args()
    train(args)

