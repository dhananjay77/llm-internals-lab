import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.eval()

text = "A Value Object is defined by its"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

# ---------------------------------------------------------
# 1. Basic information
# ---------------------------------------------------------

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("Tokens:")
print(tokens)

hidden = outputs.hidden_states[0]

print("\nInput embedding shape:")
print(hidden.shape)

# ---------------------------------------------------------
# 2. Get first attention layer
# ---------------------------------------------------------

layer = model.model.layers[0]
self_attn = layer.self_attn

# LayerNorm before attention
x = layer.input_layernorm(hidden)

# Q/K/V projections
q = self_attn.q_proj(x)
k = self_attn.k_proj(x)
v = self_attn.v_proj(x)

print("\nRaw projection shapes:")
print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

# ---------------------------------------------------------
# 3. Split into attention heads
# ---------------------------------------------------------

batch_size, seq_len, _ = q.shape

num_q_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim

q = q.view(batch_size, seq_len, num_q_heads, head_dim)
k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

# [batch, heads, seq, head_dim]
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

print("\nAfter splitting heads:")
print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

# ---------------------------------------------------------
# 4. GQA: repeat K/V heads
# ---------------------------------------------------------

repeat_factor = num_q_heads // num_kv_heads

k = k.repeat_interleave(repeat_factor, dim=1)
v = v.repeat_interleave(repeat_factor, dim=1)

print("\nAfter GQA:")
print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

# ---------------------------------------------------------
# 5. Implement RoPE
# ---------------------------------------------------------

theta = model.config.rope_parameters["rope_theta"]

print("\nRoPE:")
print("theta =", theta)
print("head_dim =", head_dim)
print("dimension pairs =", head_dim // 2)


def apply_rope(x, theta):
    """
    x shape:
        [batch, heads, seq_len, head_dim]

    Applies standard Llama-style RoPE.
    """

    device = x.device
    dtype = x.dtype

    dim = x.shape[-1]

    # Frequencies for each pair of dimensions
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(
                0,
                dim,
                2,
                device=device,
                dtype=torch.float32
            ) / dim
        )
    )

    # Position indices
    positions = torch.arange(
        x.shape[-2],
        device=device,
        dtype=torch.float32
    )

    # [seq_len, dim/2]
    freqs = torch.outer(positions, inv_freq)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Convert back to model dtype
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    # Split even/odd dimensions
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Rotation
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave even/odd dimensions
    result = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1
    ).flatten(-2)

    return result


q_rope = apply_rope(q, theta)
k_rope = apply_rope(k, theta)

print("\nAfter RoPE:")
print("Q:", q_rope.shape)
print("K:", k_rope.shape)

# ---------------------------------------------------------
# 6. Compare Q before / after RoPE
# ---------------------------------------------------------

q_before = q.float()
q_after = q_rope.float()

q_difference = q_after - q_before

print("\nQ change caused by RoPE:")

print(
    "Mean absolute difference:",
    q_difference.abs().mean().item()
)

print(
    "Maximum absolute difference:",
    q_difference.abs().max().item()
)

# ---------------------------------------------------------
# 7. Show one head
# ---------------------------------------------------------

head = 0

print("\nHead 0, position 1")

print("Q before RoPE:")
print(q_before[0, head, 1, :10])

print("\nQ after RoPE:")
print(q_after[0, head, 1, :10])

# ---------------------------------------------------------
# 8. Attention scores BEFORE RoPE
# ---------------------------------------------------------

q0 = q.float()
k0 = k.float()

scores_before = torch.matmul(
    q0,
    k0.transpose(-2, -1)
) / (head_dim ** 0.5)

# ---------------------------------------------------------
# 9. Attention scores AFTER RoPE
# ---------------------------------------------------------

q1 = q_rope.float()
k1 = k_rope.float()

scores_after = torch.matmul(
    q1,
    k1.transpose(-2, -1)
) / (head_dim ** 0.5)

# ---------------------------------------------------------
# 10. Compare attention scores for final token
# ---------------------------------------------------------

print("\nAttention scores for final token: 'its'")

print("\nBefore RoPE:")
print(scores_before[0, head, -1])

print("\nAfter RoPE:")
print(scores_after[0, head, -1])

print("\nDifference:")
print(
    scores_after[0, head, -1]
    - scores_before[0, head, -1]
)

# ---------------------------------------------------------
# 11. Softmax comparison
# ---------------------------------------------------------

weights_before = torch.softmax(
    scores_before,
    dim=-1
)

weights_after = torch.softmax(
    scores_after,
    dim=-1
)

print("\nAttention weights for final token:")
print("\nBefore RoPE:")

for token, weight in zip(
    tokens,
    weights_before[0, head, -1]
):
    print(f"{token:12s} {weight.item():.4f}")

print("\nAfter RoPE:")

for token, weight in zip(
    tokens,
    weights_after[0, head, -1]
):
    print(f"{token:12s} {weight.item():.4f}")