import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
DOMAIN_MODEL = "./output/domain-model"
LORA_MODEL = "./output/final-model"

PROMPTS = [
    "What is a Value Object in Domain-Driven Design?",
    "What is an Aggregate in Domain-Driven Design?",
    "What is event-driven architecture?",
]

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


print("Loading original model...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

print("Loading domain-pretrained model...")
domain_tokenizer = AutoTokenizer.from_pretrained(DOMAIN_MODEL)
domain_model = AutoModelForCausalLM.from_pretrained(DOMAIN_MODEL)

print("Loading LoRA model...")
lora_tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)
lora_base = AutoModelForCausalLM.from_pretrained(DOMAIN_MODEL)
lora_model = PeftModel.from_pretrained(lora_base, LORA_MODEL)

for prompt in PROMPTS:

    print("\n" + "=" * 70)
    print("QUESTION:", prompt)
    print("=" * 70)

    print("\n--- ORIGINAL ---")
    print(generate(base_model, base_tokenizer, prompt))

    print("\n--- DOMAIN PRETRAINED ---")
    print(generate(domain_model, domain_tokenizer, prompt))

    print("\n--- DOMAIN + LoRA ---")
    print(generate(lora_model, lora_tokenizer, prompt))