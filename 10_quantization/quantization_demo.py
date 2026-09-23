import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

print("Model parameters:")
print(f"{sum(p.numel() for p in model.parameters()):,}")

print("\nEstimated weight memory:")

num_params = sum(p.numel() for p in model.parameters())

for name, bits in [
    ("FP32", 32),
    ("BF16", 16),
    ("INT8", 8),
    ("INT4", 4),
]:
    memory_gb = num_params * bits / 8 / (1024 ** 3)

    print(
        f"{name:5s}: "
        f"{memory_gb:.3f} GB"
    )


# --------------------------------------------------
# Inspect one real weight tensor
# --------------------------------------------------

weight = model.model.layers[0].self_attn.q_proj.weight

print("\nQ projection:")
print("Shape:", tuple(weight.shape))
print("Dtype:", weight.dtype)

print("\nFirst 10 FP32 weights:")
print(weight.flatten()[:10])


# --------------------------------------------------
# Simulate INT8 quantization
# --------------------------------------------------

max_value = weight.abs().max()

scale = max_value / 127

quantized = torch.round(weight / scale).to(torch.int8)

dequantized = quantized.float() * scale

error = dequantized - weight

print("\nINT8 simulation:")

print("Scale:", scale.item())

print("Quantized dtype:", quantized.dtype)

print("\nFirst 10 INT8 values:")
print(quantized.flatten()[:10])

print("\nFirst 10 dequantized values:")
print(dequantized.flatten()[:10])

print("\nQuantization error:")
print("Mean absolute error:", error.abs().mean().item())
print("Maximum error:", error.abs().max().item())