import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from app import generate_response_with_visualization, format_chat_history  # <-- from your Gradio demo
import argparse

def run_llada_inference(model, tokenizer, device, prompt, label):
    print(f"🔮 Generating with {label}...")
    messages = format_chat_history([[prompt, None]])
    _, response = generate_response_with_visualization(
        model=model,
        tokenizer=tokenizer,
        device=device,
        messages=messages,
        gen_length=64,
        steps=32,
        temperature=0.7,
        cfg_scale=0.0,
        block_length=32,
        remasking='low_confidence'
    )
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What's the capital of Italy?")
    parser.add_argument("--adapter_path", type=str, default="./llada-lora-sft")
    args = parser.parse_args()

    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load base model
    print("🧠 Loading base model...")
    base_model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    base_model.eval()

    # Load LoRA-adapted model
    print("🌱 Loading fine-tuned LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device)
    lora_model.eval()

    # Run inference
    base_response = run_llada_inference(base_model, tokenizer, device, args.prompt, "Base Model")
    lora_response = run_llada_inference(lora_model, tokenizer, device, args.prompt, "LoRA Fine-Tuned Model")

    # Show results
    print("\n📊 ========== RESPONSE COMPARISON ==========")
    print(f"📥 Prompt:\n{args.prompt}\n")
    print("🧠 Base Model Response:\n" + "-"*40)
    print(base_response)
    print("\n🔥 LoRA Fine-Tuned Response:\n" + "-"*40)
    print(lora_response)
    print("="*44)

if __name__ == "__main__":
    main()
