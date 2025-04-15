import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from peft import PeftModel
from peft import LoraConfig, get_peft_model

from generate import generate  # <- this is LLaDA's generate() function


MASK_TOKEN_ID = 126336  # LLaDA's [MASK] token

def is_valid(example):
    messages = example["messages"]
    return (
        len(messages) >= 2 and
        messages[0]["role"] == "user" and
        messages[1]["role"] == "assistant" and
        all(item["content"].strip() != "" for item in messages[:2])
    )

class SFTDataset(Dataset):
    def __init__(self, tokenizer, sect, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer

        ds = load_dataset("allenai/llama-3-tulu-v2-sft-subset")["raw"]
        print("Length before filtering: ", len(ds))
        ds = ds.filter(is_valid)
        ds = ds.filter(lambda ex: len(ex["messages"]) >= 2 
                                  and ex["messages"][0]["role"] == "user" 
                                  and ex["messages"][1]["role"] == "assistant" 
                                  and ex["messages"][1]["content"].strip() != "")

        print("Length after filtering: ", len(ds))
        if sect == "train":
            ds = ds.select(range(10000))
        elif sect == "test":
            ds = ds.select(range(10000, 11800))
        elif sect == "validate":
            ds = ds.select(range(11800, len(ds)))

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
                "answer_ids": answer_ids,
                "prompt": prompt,
                "answer": answer
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def avg(metric_list, key):
    return np.mean([x[key] for x in metric_list])

def main():
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load separate base models for each adapter
    base_model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    base_model.eval()

    base_model_for_lora_baseline = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    lora_model_baseline = PeftModel.from_pretrained(base_model_for_lora_baseline, "./llada-lora-sft-baseline").to(device)
    lora_model_baseline.eval()

    base_model_for_lora_sched = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    lora_model_mask_sched = PeftModel.from_pretrained(base_model_for_lora_sched, "./llada-lora-sft-masking-sched").to(device)
    lora_model_mask_sched.eval()


    base_model_for_lora_sched_weighted = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    lora_model_mask_sched_weighted = PeftModel.from_pretrained(base_model_for_lora_sched_weighted, "./llada-lora-sft-masking-sched-weighted-timesteps").to(device)
    lora_model_mask_sched_weighted.eval()

    print("📦 Loading test dataset...")
    dataset = SFTDataset(tokenizer, sect="test", max_len=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for batch in tqdm(dataloader, desc="Evaluating generate()"):
        prompt_ids = batch["prompt_ids"][0].unsqueeze(0).to(device)
        answer_ids = batch["answer_ids"][0]
        reference = tokenizer.decode(answer_ids, skip_special_tokens=True)
        prompt_text = batch["prompt"][0]
        gen_len = answer_ids.shape[0]

        # === Base model ===
        base_generated_ids = generate(
            base_model,
            prompt=prompt_ids,
            steps=8,
            gen_length=gen_len,
            block_length=gen_len
        )
        base_gen_text = tokenizer.decode(base_generated_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        base_bleu = sentence_bleu([reference.split()], base_gen_text.split())
        base_rouge = scorer.score(reference, base_gen_text)

        # === LoRA baseline ===
        lora_baseline_generated_ids = generate(
            lora_model_baseline,
            prompt=prompt_ids,
            steps=8,
            gen_length=gen_len,
            block_length=gen_len
        )
        lora_baseline_gen_text = tokenizer.decode(lora_baseline_generated_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        lora_baseline_bleu = sentence_bleu([reference.split()], lora_baseline_gen_text.split())
        lora_baseline_rouge = scorer.score(reference, lora_baseline_gen_text)

        # === LoRA masking-sched ===
        lora_mask_generated_ids = generate(
            lora_model_mask_sched,
            prompt=prompt_ids,
            steps=8,
            gen_length=gen_len,
            block_length=gen_len
        )
        lora_mask_gen_text = tokenizer.decode(lora_mask_generated_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        lora_mask_bleu = sentence_bleu([reference.split()], lora_mask_gen_text.split())
        lora_mask_rouge = scorer.score(reference, lora_mask_gen_text)

        # === LoRA masking-sched-weighted-timesteps===
        lora_mask_generated_ids_weighted = generate(
            lora_model_mask_sched_weighted,
            prompt=prompt_ids,
            steps=8,
            gen_length=gen_len,
            block_length=gen_len
        )
        lora_mask_gen_text_weighted = tokenizer.decode(lora_mask_generated_ids_weighted[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        lora_mask_bleu_weighted = sentence_bleu([reference.split()], lora_mask_gen_text_weighted.split())
        lora_mask_rouge_weighted = scorer.score(reference, lora_mask_gen_text_weighted)


        # === Log metrics ===
        print("\n===========================")
        print(f"📥 Prompt: {prompt_text}")
        print(f"🎯 Reference: {reference}\n")

        print("🧠 Base Model:")
        print(f"→ {base_gen_text}")
        print(f"📊 BLEU: {base_bleu:.4f}, ROUGE-1: {base_rouge['rouge1'].fmeasure:.4f}, ROUGE-L: {base_rouge['rougeL'].fmeasure:.4f}")

        print("\n🔥 LoRA Baseline:")
        print(f"→ {lora_baseline_gen_text}")
        print(f"📊 BLEU: {lora_baseline_bleu:.4f}, ROUGE-1: {lora_baseline_rouge['rouge1'].fmeasure:.4f}, ROUGE-L: {lora_baseline_rouge['rougeL'].fmeasure:.4f}")

        print("\n⚡ LoRA Masking Schedule:")
        print(f"→ {lora_mask_gen_text}")
        print(f"📊 BLEU: {lora_mask_bleu:.4f}, ROUGE-1: {lora_mask_rouge['rouge1'].fmeasure:.4f}, ROUGE-L: {lora_mask_rouge['rougeL'].fmeasure:.4f}")
        print("===========================\n")

        print("\n⚡ LoRA Masking Schedule Weighted Timesteps:")
        print(f"→ {lora_mask_gen_text_weighted}")
        print(f"📊 BLEU: {lora_mask_bleu_weighted:.4f}, ROUGE-1: {lora_mask_rouge_weighted['rouge1'].fmeasure:.4f}, ROUGE-L: {lora_mask_rouge_weighted['rougeL'].fmeasure:.4f}")
        print("===========================\n")

        # === Optional: Collect for averages ===
        bleu_scores.append({
            "base": base_bleu,
            "lora_baseline": lora_baseline_bleu,
            "lora_mask_sched": lora_mask_bleu,
            "lora_mask_sched_weighted": lora_mask_bleu_weighted
        })
        rouge1_scores.append({
            "base": base_rouge['rouge1'].fmeasure,
            "lora_baseline": lora_baseline_rouge['rouge1'].fmeasure,
            "lora_mask_sched": lora_mask_rouge['rouge1'].fmeasure,
            "lora_mask_sched_weighted": lora_mask_rouge_weighted['rouge1'].fmeasure
        })
        rougeL_scores.append({
            "base": base_rouge['rougeL'].fmeasure,
            "lora_baseline": lora_baseline_rouge['rougeL'].fmeasure,
            "lora_mask_sched": lora_mask_rouge['rougeL'].fmeasure,
            "lora_mask_sched_weighted": lora_mask_rouge_weighted['rougeL'].fmeasure
        })



        print("\n📊 === Final Evaluation Scores ===")
        for label in ["base", "lora_baseline", "lora_mask_sched", "lora_mask_sched_weighted"]:
            print(f"🔍 {label.upper()}")
            print(f"BLEU:     {avg(bleu_scores, label):.4f}")
            print(f"ROUGE-1:  {avg(rouge1_scores, label):.4f}")
            print(f"ROUGE-L:  {avg(rougeL_scores, label):.4f}")
            print("-" * 30)



if __name__ == "__main__":
    main()
