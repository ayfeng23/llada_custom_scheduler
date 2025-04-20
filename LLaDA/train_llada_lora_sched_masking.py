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
import argparse


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
        token_clean = (token.lstrip("Ġ")).lower()  # remove BPE space if any
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

        # ds = load_dataset("allenai/llama-3-tulu-v2-sft-subset")["raw"]
        ds = load_dataset("HuggingFaceTB/smoltalk", "all")["train"]
        print("Length before filtering: ", len(ds))
        ds = ds.filter(is_valid)
        ds = ds.filter(lambda ex: len(ex["messages"]) >= 2 
                                  and ex["messages"][0]["role"] == "user" 
                                  and ex["messages"][1]["role"] == "assistant" 
                                  and ex["messages"][1]["content"].strip() != "")

        print("Length after filtering: ", len(ds))
        if sect == "train":
            ds = ds.select(range(60000))
        elif sect == "test":
            ds = ds.select(range(60000, 68000))
        elif sect == "validate":
            ds = ds.select(range(68000, 70000))

        for ex in ds:
            prompt = ex["messages"][0]["content"]
            answer = ex["messages"][1]["content"]

            prompt_text = "<s><|startofuser|>\n" + prompt + "<|endofuser|><|startofassistant|>\n"
            answer_text = answer + "<|endoftext|>"

            prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
            answer_ids = tokenizer(answer_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]

            # Merge and pad
            input_ids = torch.cat([prompt_ids, answer_ids])
            input_ids = input_ids[:max_len]
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id] * pad_len)])

            self.samples.append({
                "input_ids": input_ids,
                "prompt_length": torch.tensor(len(prompt_ids)),
                "answer_length": torch.tensor(len(answer_ids)),
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def timestep_schedule(args):
    t = 0
    if args.weighted:
        rand_num = random.uniform(0,1)
        if rand_num >= 0.75:
            t = random.uniform(0.8, 1)
        else:
            t = random.uniform(0, 1)
    else:
        t = random.uniform(0,1)

    return t


#FIX THIS
#for 60k, 5 and 12k
#for 10k 7 and 1500
def sampling_schedule(step, total_steps, schedule_type='inv_sigmoid', k=5.0, t=12000):
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
    print("Running Original Forward Process")
    t = mask_prob
    p_mask = (torch.rand_like(input_ids.float()) < t)
    noisy_input = input_ids.clone()
    noisy_input[p_mask] = mask_token_id
    return noisy_input, p_mask, None

def new_forward_process(input_ids, mask_prob, tokenid_to_priority,
                        mask_token_id=MASK_TOKEN_ID, temperature=0.3, gamma=4.0):
    # print("Running New Forward Process")
    device = input_ids.device
    token_ids = input_ids.cpu().numpy()  # (B, L)
    priorities = tokenid_to_priority[token_ids]  # (B, L)

    use_uniform = np.random.rand() < temperature

    if use_uniform:
        return orig_forward_process(input_ids, mask_prob, mask_token_id=MASK_TOKEN_ID)
        # timestamps = np.random.rand(*token_ids.shape)  # shape (B, L)
    else:
        b = 1 + gamma * priorities
        a = 1 + gamma * (1 - priorities)
        timestamps = beta.rvs(a, b, size=token_ids.shape)

    print("Running New Forward Process")

    sorted_indices = np.argsort(timestamps, axis=1)  # (B, L)

    mat_to_mask = np.random.rand(*token_ids.shape)
    num_mask = (mat_to_mask < mask_prob).sum(axis=1)

    B, L = sorted_indices.shape
    row_idx = np.repeat(np.arange(B), num_mask)

    # Gather all the col indices into one flat array
    col_idx = np.concatenate([
        sorted_indices[i, :k] for i, k in enumerate(num_mask)
    ])
   
    # Now assign
    mask_flags = np.zeros_like(sorted_indices, dtype=bool)
    mask_flags[row_idx, col_idx] = True

    p_mask = torch.from_numpy(mask_flags).to(device=input_ids.device)
    noisy_input = input_ids.clone()
    noisy_input[p_mask] = mask_token_id

    return noisy_input, p_mask, sorted_indices

def forward_process(input_ids, mask_prob, step, total_steps, tokenid_to_priority, mask_token_id=MASK_TOKEN_ID, temperature=0.3, idf_storage=None):
    
    p = sampling_schedule(step, total_steps)
    print("Sampling Schedule Degbugging", step, p)
    if np.random.rand() < p:
        #print("Original Forward Process")
        noisy_input, p_mask, sorted_indices = orig_forward_process(input_ids, mask_prob, mask_token_id=mask_token_id)
        strategy = "orig"
    else:
        #print("New Forward Process")
        noisy_input, p_mask, sorted_indices = new_forward_process(input_ids, mask_prob, tokenid_to_priority, mask_token_id=mask_token_id, temperature=temperature)
        strategy = "new"

    token_ids = input_ids.detach().cpu().numpy()
    mask_flags = p_mask.detach().cpu().numpy()

    masked_token_ids = token_ids[mask_flags]
    unmasked_token_ids = token_ids[~mask_flags]

    # print(f"[DEBUG] Strategy: {strategy}")
    # print(f"[DEBUG] Available idf_storage keys: {list(idf_storage.keys())}")
    # print(f"[DEBUG] idf_storage[{strategy}_masked]: {idf_storage.get(f'{strategy}_masked')}")

    idf_storage[f"{strategy}_masked"].extend(tokenid_to_priority[masked_token_ids].tolist())
    idf_storage[f"{strategy}_unmasked"].extend(tokenid_to_priority[unmasked_token_ids].tolist())
    idf_storage["mask_prob"].append(mask_prob)


    mean_masked = tokenid_to_priority[masked_token_ids].mean()
    mean_unmasked = tokenid_to_priority[unmasked_token_ids].mean()
    print(f"[{strategy.upper()}] mask_prob: {mask_prob:.2f} | masked IDF avg: {mean_masked:.4f} | unmasked IDF avg: {mean_unmasked:.4f}")



    return noisy_input, p_mask, sorted_indices



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

    # Exclude masked PAD tokens
    pad_mask = input_ids != tokenizer.pad_token_id
    masked_indices = p_mask.bool() & pad_mask  # (batch, seq_len)

    total_masked = masked_indices.sum().item()
    # print(f"[DEBUG] Total masked (non-PAD) tokens: {total_masked}")

    if total_masked == 0:
        # print("[WARNING] No masked tokens found — returning zero loss.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits = logits[masked_indices]      # (num_masked, vocab)
    targets = input_ids[masked_indices]  # (num_masked,)


    # print(f"[DEBUG] Logits shape: {logits.shape}, Targets shape: {targets.shape}")

    token_loss = F.cross_entropy(logits, targets, reduction="mean")

    # print(f"[DEBUG] Avg masked token loss: {token_loss.item():.4f}")
    return token_loss



def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.baseline and args.weighted:
        name = "llada-finetuning-baseline-weighted"
    elif not args.baseline and args.weighted:
        name = "llada-finetuning-scheduled-masking-weighted"
    elif args.baseline and not args.weighted:
        name = "llada-finetuning-baseline-uniform"
    else:
        name = "llada-finetuning-scheduled-masking-uniform"

    wandb.init(
        project="llada-lora-finetuning",
        name=name,
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
        r=16, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"]
    ))

    idf_dict = create_idf_dict()
    tokenid_to_priority = build_tokenid_to_priority(vocab_dict, idf_dict)

    model.print_trainable_parameters()

    dataset = SFTDataset(tokenizer, sect="train", max_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = SFTDataset(tokenizer, sect="validate", max_len=args.max_seq_len)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    step_count = 0

    idf_storage = {
        "mask_prob": [],
        "orig_masked": [],
        "orig_unmasked": [],
        "new_masked": [],
        "new_unmasked": [],
    }


    for epoch in range(args.epochs):
        total_loss = 0.0
        running_loss = 0.0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            t = timestep_schedule(args)
            prompt_ids = batch["prompt_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)


            if args.baseline:
                temperature = 1
            else:
                temperature = 0

            noisy_answer, p_mask, _ = forward_process(
                input_ids=answer_ids,
                mask_prob=t,
                step=i,
                total_steps=total_steps,
                tokenid_to_priority=tokenid_to_priority,
                mask_token_id=MASK_TOKEN_ID,
                temperature=temperature,
                idf_storage=idf_storage
            )

            # Reconstruct full input
            noisy_batch = torch.cat([prompt_ids, noisy_answer], dim=1)
            input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            prompt_lengths = batch["prompt_length"].to(device)
            # Build p_mask aligned with full input
            full_mask = torch.zeros_like(noisy_batch, dtype=torch.bool)
            for b in range(noisy_batch.size(0)):
                full_mask[b, prompt_lengths[b]:] = p_mask[b]

            # print("DEBUGGING OCCURING HERE:")
            # print("MASK PROB: ", t)
            decoded_gt = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
            # for j in range(len(decoded_gt)):
            #     print(f"[Batch {j}] ORIGINAL ANSWER {decoded_gt[j]}\n{'-'*60}")
            # print()

            decoded_noisy = tokenizer.batch_decode(noisy_batch, skip_special_tokens=False)
            # for j, decoded in enumerate(decoded_noisy):
            #     print(f"[Batch {j}] MASKED ANSWER {decoded}\n{'-'*60}")
            # print(p_mask.int())  # 1 where masked, 0 elsewhere
            # print("DEBUGGING FINISHED")

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                outputs = model(input_ids=noisy_batch)
                loss = compute_llada_sft_loss(outputs.logits, input_ids, full_mask, prompt_lengths)   

            # # Get logits and predictions
            # logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
            # pred_ids = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)

            # # Only get answer portion for each sample
            # for b in range(pred_ids.size(0)):
            #     answer_start = prompt_lengths[b].item()
            #     predicted_tokens = pred_ids[b, answer_start:]
            #     decoded_pred = tokenizer.decode(predicted_tokens.cpu(), skip_special_tokens=False)

            #     print(f"\n🔮 [DEBUG] Batch {b} Predicted token IDs (answer part):")
            #     print(predicted_tokens.tolist())
            #     print(f"📝 [DEBUG] Batch {b} Decoded predicted answer:\n{decoded_pred}")         

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 100 == 0:
                    avg_loss_100 = running_loss / 100
                    print(f"📉 [Train] Step: {step_count} | Avg Loss (last 100 steps): {avg_loss_100:.4f}")
                    wandb.log({"train/loss_100step_avg": avg_loss_100})
                    running_loss = 0

                if step_count % 500 == 0:
                    print(f"\n🔍 Running validation at step {step_count}...")
                    model.eval()
                    total_val_loss = 0.0
                    total_val_tokens = 0

                    with torch.no_grad():
                        for val_batch in val_dataloader:

                            input_ids = val_batch["input_ids"].to(device)
                            prompt_lengths = val_batch["prompt_length"].to(device)

                            p_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                            for b in range(input_ids.size(0)):
                                p_mask[b, prompt_lengths[b]:] = 1

                            noisy_input = input_ids.clone()
                            noisy_input[p_mask] = MASK_TOKEN_ID

                            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                                outputs = model(input_ids=noisy_input)
                                loss = compute_llada_sft_loss(outputs.logits, input_ids, p_mask, prompt_lengths)

                            total_val_loss += loss.item() * p_mask.sum().item()
                            total_val_tokens += p_mask.sum().item()

                        avg_val_loss = total_val_loss / total_val_tokens if total_val_tokens > 0 else 0.0
                        val_perplexity = np.exp(avg_val_loss) if avg_val_loss < 100 else float('inf')

                        print(f"✅ [Validation] Step: {step_count} | Avg Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f} | Tokens Evaluated: {total_val_tokens}")
                        wandb.log({
                            "val/loss": avg_val_loss,
                            "val/perplexity": np.exp(avg_val_loss),
                            "train/step": step_count
                        })
                    model.train()

            total_loss += loss.item()
            running_loss += loss.item()

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

    df = pd.DataFrame.from_dict(idf_storage, orient="index").transpose()
    df.to_csv("idf_masking_stats.csv", index=False)

    # print("Starting evaluation")
    # model.eval()
    # eval_dataset = SFTDataset(tokenizer, sect="test", max_len=args.max_seq_len)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # total_eval_loss = 0.0
    # total_eval_tokens = 0

    # with torch.no_grad():
    #     for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #         input_ids = batch["input_ids"].to(device)
    #         prompt_lengths = batch["prompt_length"].to(device)

    #         # Mask the entire answer portion (no randomness)
    #         p_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    #         for b in range(input_ids.size(0)):
    #             p_mask[b, prompt_lengths[b]:] = 1

    #         noisy_input = input_ids.clone()
    #         noisy_input[p_mask] = MASK_TOKEN_ID

    #         with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
    #             outputs = model(input_ids=noisy_input)
    #             loss = compute_llada_sft_loss(outputs.logits, input_ids, p_mask, prompt_lengths)

    #         total_eval_loss += loss.item() * p_mask.sum().item()
    #         total_eval_tokens += p_mask.sum().item()

    # avg_eval_loss = total_eval_loss / total_eval_tokens if total_eval_tokens > 0 else 0.0
    # print(f"[Eval] Avg loss: {avg_eval_loss:.4f}")
    # wandb.log({"eval/avg_loss": avg_eval_loss})
    
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument('--baseline', action='store_true', help='Enable baseline mode.')
    parser.add_argument('--weighted', action='store_true', help='Enable weighted timesteps mode.')
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        if args.baseline and not args.weighted:
            print("Running Baseline Uniform Masking with Uniform Timesteps")
            args.output_dir = "./llada-lora-sft-baseline"
        elif args.baseline and args.weighted:
            print("Running Baseline Uniform Masking with Weighted Timesteps")
            args.output_dir = "./llada-lora-sft-baseline-weighted"
        elif not args.baseline and args.weighted:
            print("Running Custom Masking Schedule with Weighted Timesteps")
            args.output_dir = "./llada-lora-sft-masking-sched-weighted"
        else:
            print("Running Custom Masking Schedule with Uniform Timesteps")
            args.output_dir = "./llada-lora-sft-masking-sched"

    train(args)