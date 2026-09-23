import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.eval()

text = "A Value Object is defined by its"
inputs = tokenizer(text, return_tensors="pt")

# ---------------------------------------------------------
# 1. Run the COMPLETE model
# ---------------------------------------------------------

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)

print("\nNumber of hidden-state tensors:")
print(len(outputs.hidden_states))

# hidden_states[0] = embedding output
# hidden_states[1] = output after Transformer block 0
# hidden_states[2] = output after Transformer block 1
# ...

x_after_block0 = outputs.hidden_states[1]

print("\nOutput of Transformer block 0:")
print(x_after_block0.shape)

# ---------------------------------------------------------
# 2. We need the input to the MLP.
#
# The first block has:
#
# RMSNorm
# Attention
# Residual
# RMSNorm
# MLP
# Residual
#
# We already understand the attention part.
#
# For this experiment, use the actual block input and
# manually execute the MLP portion.
# ---------------------------------------------------------

x_block_input = outputs.hidden_states[0]

layer = model.model.layers[0]
mlp = layer.mlp

# ---------------------------------------------------------
# 3. Post-attention representation
#
# We can obtain it by using the actual model block
# internals through the complete block output:
#
# Instead of reproducing the attention internals again,
# we'll demonstrate the MLP mathematically using the
# block's input representation.
# ---------------------------------------------------------

# For educational purposes:
# apply RMSNorm to the block input.

x_norm = layer.post_attention_layernorm(
    x_block_input
)

print("\nMLP input:")
print(x_norm.shape)

# ---------------------------------------------------------
# 4. Inspect the three MLP matrices
# ---------------------------------------------------------

print("\nMLP projection weights:")

print(
    "gate_proj:",
    mlp.gate_proj.weight.shape
)

print(
    "up_proj:",
    mlp.up_proj.weight.shape
)

print(
    "down_proj:",
    mlp.down_proj.weight.shape
)

# ---------------------------------------------------------
# 5. Gate projection
# ---------------------------------------------------------

gate = mlp.gate_proj(x_norm)

print("\nGate projection:")
print(gate.shape)

# ---------------------------------------------------------
# 6. Up projection
# ---------------------------------------------------------

up = mlp.up_proj(x_norm)

print("\nUp projection:")
print(up.shape)

# ---------------------------------------------------------
# 7. SwiGLU
# ---------------------------------------------------------

activated_gate = mlp.act_fn(gate)

print("\nAfter activation:")
print(activated_gate.shape)

# ---------------------------------------------------------
# 8. Element-wise multiplication
# ---------------------------------------------------------

combined = activated_gate * up

print("\nAfter Gate × Up:")
print(combined.shape)

# ---------------------------------------------------------
# 9. Down projection
# ---------------------------------------------------------

mlp_output = mlp.down_proj(combined)

print("\nMLP output:")
print(mlp_output.shape)

# ---------------------------------------------------------
# 10. Inspect final token
# ---------------------------------------------------------

idx = -1

print("\nFinal token:")
print(tokens[idx])

print("\nFirst 10 values:")

print("\nGate:")
print(gate[0, idx, :10])

print("\nActivated Gate:")
print(activated_gate[0, idx, :10])

print("\nUp:")
print(up[0, idx, :10])

print("\nGate × Up:")
print(combined[0, idx, :10])

print("\nMLP output:")
print(mlp_output[0, idx, :10])

# ---------------------------------------------------------
# 11. Norms
# ---------------------------------------------------------

print("\nNorms:")

print(
    "MLP input  =",
    torch.norm(x_norm[0, idx].float()).item()
)

print(
    "MLP output =",
    torch.norm(mlp_output[0, idx].float()).item()
)