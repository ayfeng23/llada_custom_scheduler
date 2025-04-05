# IMPORTANT: NOT WORKIGN LMAO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==== Config ====
base_model_name = "GSAI-ML/LLaDA-8B-Instruct"
adapter_path = "./llada-lora-sft"  # Path to your fine-tuned LoRA adapter
prompt = "What's the capital of Italy?"
max_new_tokens = 64

# ==== Load Tokenizer ====
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# ==== Format Input ====
input_text = f"<s><|startofuser|>\n{prompt}<|endofuser|><|startofassistant|>\n"
inputs = tokenizer(input_text, return_tensors="pt")

# ==== Load Base Model ====
print("🧠 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
inputs_base = {k: v.to(base_model.device) for k, v in inputs.items()}

# ==== Generate with Base ====
print("✨ Generating output from base model...")
with torch.no_grad():
    base_output = base_model.generate(
        **inputs_base,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id
    )
base_decoded = tokenizer.decode(base_output[0], skip_special_tokens=True)

# ==== Load Fine-Tuned (LoRA) Model ====
print("🧪 Loading LoRA fine-tuned model...")
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
inputs_lora = {k: v.to(lora_model.device) for k, v in inputs.items()}

# ==== Generate with Fine-Tuned ====
print("🔥 Generating output from fine-tuned LoRA model...")
with torch.no_grad():
    lora_output = lora_model.generate(
        **inputs_lora,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id
    )
lora_decoded = tokenizer.decode(lora_output[0], skip_special_tokens=True)

# ==== Print Comparison ====
print("\n📊 ===================== COMPARISON =====================")
print(f"📥 Prompt:\n{prompt}")
print("\n🧠 Base Model Response:\n" + "-" * 40)
print(base_decoded)

print("\n🔥 Fine-Tuned LoRA Model Response:\n" + "-" * 40)
print(lora_decoded)
print("========================================================\n")
