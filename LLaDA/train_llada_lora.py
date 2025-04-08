import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import argparse
import wandb
import random

MASK_TOKEN_ID = 126336  # [MASK] token for LLaDA


from datasets import load_dataset
from torch.utils.data import Dataset
import torch

def is_valid(example):
    messages = example["messages"]
    return (
        len(messages) >= 2 and
        messages[0]["role"] == "user" and
        messages[1]["role"] == "assistant" and
        all(item["content"].strip() != "" for item in messages[:2])
    )

class SFTDataset(Dataset):
    def __init__(self, tokenizer, max_len=512):
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
        ds = ds.select(range(10000))

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


def forward_process(input_ids, mask_token_id=MASK_TOKEN_ID):
    t = random.uniform(0, 1)
    p_mask = (torch.rand_like(input_ids.float()) < t)
    noisy_input = input_ids.clone()
    noisy_input[p_mask] = mask_token_id
    return noisy_input, p_mask


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

    model.print_trainable_parameters()

    dataset = SFTDataset(tokenizer, max_len=args.max_seq_len)
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

            noisy_batch, p_mask = forward_process(input_ids)
            token_positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

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
                wandb.log({"train/loss": loss.item(), "step": step_count})

            # cleanup
            del loss, outputs
            torch.cuda.empty_cache()

        epoch_avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {epoch_avg_loss:.4f}")
        wandb.log({f"epoch_{epoch+1}/avg_loss": epoch_avg_loss})

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="./llada-lora-sft")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    args = parser.parse_args()
    train(args)

