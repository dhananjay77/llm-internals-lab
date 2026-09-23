import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

model.eval()

prompt = "A Value Object is defined by"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print("Prompt:")
print(prompt)

print("\nInput tokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# --------------------------------------------------
# Generate
# --------------------------------------------------

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

# --------------------------------------------------
# Inspect generated tokens
# --------------------------------------------------

new_tokens = output[0][inputs["input_ids"].shape[1]:]

print("\nGenerated token IDs:")
print(new_tokens.tolist())

print("\nGenerated tokens:")
print(tokenizer.convert_ids_to_tokens(new_tokens))

print("\nFinal text:")
print(tokenizer.decode(output[0]))