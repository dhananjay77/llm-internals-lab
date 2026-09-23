import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

weight = model.model.layers[0].self_attn.q_proj.weight.detach()

max_value = weight.abs().max()


def quantize_and_measure(bits):
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)

    scale = max_value / qmax

    quantized = torch.round(
        weight / scale
    ).clamp(qmin, qmax)

    dequantized = quantized * scale

    error = dequantized - weight

    return (
        scale,
        quantized,
        dequantized,
        error
    )


# --------------------------------------------------
# INT8
# --------------------------------------------------

scale8, q8, dq8, error8 = quantize_and_measure(8)

print("========== INT8 ==========")
print("Range: -128 to 127")
print("Scale:", scale8.item())
print("Mean absolute error:", error8.abs().mean().item())
print("Maximum error:", error8.abs().max().item())


# --------------------------------------------------
# INT4
# --------------------------------------------------

scale4, q4, dq4, error4 = quantize_and_measure(4)

print("\n========== INT4 ==========")
print("Range: -8 to 7")
print("Scale:", scale4.item())
print("Mean absolute error:", error4.abs().mean().item())
print("Maximum error:", error4.abs().max().item())


# --------------------------------------------------
# Compare
# --------------------------------------------------

print("\n========== COMPARISON ==========")

print(
    "INT8 mean error:",
    error8.abs().mean().item()
)

print(
    "INT4 mean error:",
    error4.abs().mean().item()
)

print(
    "INT4 / INT8 error ratio:",
    (
        error4.abs().mean()
        / error8.abs().mean()
    ).item()
)


# --------------------------------------------------
# Example values
# --------------------------------------------------

print("\nFirst 10 original:")
print(weight.flatten()[:10])

print("\nFirst 10 INT8:")
print(q8.flatten()[:10])

print("\nFirst 10 INT4:")
print(q4.flatten()[:10])

print("\nFirst 10 INT8 dequantized:")
print(dq8.flatten()[:10])

print("\nFirst 10 INT4 dequantized:")
print(dq4.flatten()[:10])