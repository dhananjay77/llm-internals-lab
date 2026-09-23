import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

# --------------------------------------------------
# 1. Pick one real model weight
# --------------------------------------------------

weight = model.model.layers[0].self_attn.q_proj.weight.detach()

print("Original weight")
print("Shape:", tuple(weight.shape))
print("Dtype:", weight.dtype)


# --------------------------------------------------
# 2. Quantize base weight to INT4
# --------------------------------------------------

max_value = weight.abs().max()

scale = max_value / 7

base_int4 = torch.round(
    weight / scale
).clamp(-8, 7)

base_dequantized = base_int4 * scale

print("\nINT4 base model")
print("Quantized range:",
      base_int4.min().item(),
      "to",
      base_int4.max().item())

print("Scale:", scale.item())


# --------------------------------------------------
# 3. Create LoRA matrices
# --------------------------------------------------

rank = 8

A = torch.randn(
    rank,
    weight.shape[1]
) * 0.01

B = torch.zeros(
    weight.shape[0],
    rank
)

scaling = 16 / rank


# --------------------------------------------------
# 4. LoRA update
# --------------------------------------------------

delta_W = (B @ A) * scaling

effective_weight = (
    base_dequantized + delta_W
)


# --------------------------------------------------
# 5. Compare
# --------------------------------------------------

print("\nLoRA")
print("A shape:", tuple(A.shape))
print("B shape:", tuple(B.shape))
print("Delta W shape:", tuple(delta_W.shape))

print("\nParameter counts:")

print(
    "Base weight:",
    weight.numel()
)

print(
    "LoRA A:",
    A.numel()
)

print(
    "LoRA B:",
    B.numel()
)

print(
    "LoRA total:",
    A.numel() + B.numel()
)

print("\nEffective weight shape:",
      tuple(effective_weight.shape))

print("\nBase INT4 reconstruction error:")
print(
    (base_dequantized - weight)
    .abs()
    .mean()
    .item()
)

print("\nLoRA update norm:")
print(
    delta_W.norm().item()
)