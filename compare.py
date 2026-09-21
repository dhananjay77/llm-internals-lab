from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "HuggingFaceTB/SmolLM2-135M"
TRAINED = "./output/domain-model"

prompt = "Domain Driven Design is"

def generate(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


print("\n===== BASE MODEL =====")
print(generate(BASE))

print("\n===== DOMAIN-PRETRAINED MODEL =====")
print(generate(TRAINED))