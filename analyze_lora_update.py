import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./output/domain-model"
LORA_MODEL = "./output/final-model"

# Load frozen base model
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Load trained LoRA
model = PeftModel.from_pretrained(base, LORA_MODEL)

# Layer 0 Q projection
layer = model.base_model.model.model.layers[0].self_attn.q_proj

# Original weight W
W = layer.base_layer.weight.detach()

# LoRA matrices
A = layer.lora_A["default"].weight.detach()
B = layer.lora_B["default"].weight.detach()

# LoRA scaling
scaling = layer.scaling["default"]

# Actual LoRA update
delta_W = (B @ A) * scaling

print("Original W")
print("Shape:", tuple(W.shape))
print("Parameters:", W.numel())

print("\nLoRA A")
print("Shape:", tuple(A.shape))

print("\nLoRA B")
print("Shape:", tuple(B.shape))

print("\nDelta W = B @ A * scaling")
print("Shape:", tuple(delta_W.shape))

print("\n--- Magnitudes ---")

print("W mean abs:       ", W.abs().mean().item())
print("Delta W mean abs: ", delta_W.abs().mean().item())

print("\nW Frobenius norm:       ", torch.linalg.norm(W).item())
print("Delta W Frobenius norm: ", torch.linalg.norm(delta_W).item())

ratio = (
    torch.linalg.norm(delta_W) /
    torch.linalg.norm(W)
)

print("\nDelta / W ratio:", ratio.item())
print("Delta / W %:    ", ratio.item() * 100)