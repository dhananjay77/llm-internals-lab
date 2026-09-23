import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./output/domain-model"
LORA_MODEL = "./output/final-model"

text = "A Value Object is defined by its"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Load LoRA model
lora_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
lora_model = PeftModel.from_pretrained(lora_base, LORA_MODEL)

# Get token IDs
inputs = tokenizer(text, return_tensors="pt")

print("Tokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# --------------------------------------------------
# Extract layer 0 Q projection
# --------------------------------------------------

base_q = base.model.layers[0].self_attn.q_proj

lora_q = lora_model.base_model.model.model.layers[0].self_attn.q_proj

# Original W
W = base_q.weight.detach().float()

# LoRA A/B
A = lora_q.lora_A["default"].weight.detach()
B = lora_q.lora_B["default"].weight.detach()

scaling = lora_q.scaling["default"]

# Actual LoRA update
delta_W = (B @ A) * scaling

# --------------------------------------------------
# Get hidden states entering Q projection
# --------------------------------------------------

with torch.no_grad():

    # Embeddings
    hidden = base.model.embed_tokens(inputs["input_ids"]).float()

    # SmolLM2 applies RMSNorm before attention
    hidden = base.model.layers[0].input_layernorm(hidden)

    # Q before LoRA
    Q_before = hidden @ W.T

    # Q after LoRA
    W_after = W + delta_W

    Q_after = hidden @ W_after.T

# --------------------------------------------------
# Compare
# --------------------------------------------------

difference = Q_after - Q_before

print("\nQ shape:")
print(Q_before.shape)

print("\nQ before LoRA:")
print("Mean:", Q_before.mean().item())
print("Std :", Q_before.std().item())

print("\nQ after LoRA:")
print("Mean:", Q_after.mean().item())
print("Std :", Q_after.std().item())

print("\nDifference:")
print("Mean abs:", difference.abs().mean().item())
print("Max abs :", difference.abs().max().item())
print("Norm    :", torch.linalg.norm(difference).item())

print("\nRelative Q change:")

ratio = (
    torch.linalg.norm(difference) /
    torch.linalg.norm(Q_before)
)

print(ratio.item() * 100, "%")