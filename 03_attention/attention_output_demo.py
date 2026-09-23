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

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# ---------------------------------------------------------
# 1. Input representation
# ---------------------------------------------------------

x = outputs.hidden_states[0]

print("Tokens:")
print(tokens)

print("\nX shape:")
print(x.shape)

# ---------------------------------------------------------
# 2. First Transformer layer
# ---------------------------------------------------------

layer = model.model.layers[0]
attn = layer.self_attn

# Pre-attention LayerNorm
x_norm = layer.input_layernorm(x)

# ---------------------------------------------------------
# 3. Q / K / V
# ---------------------------------------------------------

q = attn.q_proj(x_norm)
k = attn.k_proj(x_norm)
v = attn.v_proj(x_norm)

batch, seq_len, _ = q.shape

num_q_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim

# Split heads
q = q.view(
    batch,
    seq_len,
    num_q_heads,
    head_dim
).transpose(1, 2)

k = k.view(
    batch,
    seq_len,
    num_kv_heads,
    head_dim
).transpose(1, 2)

v = v.view(
    batch,
    seq_len,
    num_kv_heads,
    head_dim
).transpose(1, 2)

# ---------------------------------------------------------
# 4. GQA
# ---------------------------------------------------------

repeat_factor = num_q_heads // num_kv_heads

k = k.repeat_interleave(
    repeat_factor,
    dim=1
)

v = v.repeat_interleave(
    repeat_factor,
    dim=1
)

# ---------------------------------------------------------
# 5. RoPE
# ---------------------------------------------------------

theta = model.config.rope_parameters["rope_theta"]


def apply_rope(x, theta):

    dim = x.shape[-1]

    inv_freq = 1.0 / (
        theta ** (
            torch.arange(
                0,
                dim,
                2,
                device=x.device,
                dtype=torch.float32
            ) / dim
        )
    )

    positions = torch.arange(
        x.shape[-2],
        device=x.device,
        dtype=torch.float32
    )

    freqs = torch.outer(
        positions,
        inv_freq
    )

    cos = torch.cos(freqs).to(x.dtype)
    sin = torch.sin(freqs).to(x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = (
        x_even * cos -
        x_odd * sin
    )

    rotated_odd = (
        x_even * sin +
        x_odd * cos
    )

    return torch.stack(
        [rotated_even, rotated_odd],
        dim=-1
    ).flatten(-2)


q = apply_rope(q, theta)
k = apply_rope(k, theta)

# ---------------------------------------------------------
# 6. Attention
# ---------------------------------------------------------

q = q.float()
k = k.float()
v = v.float()

scores = torch.matmul(
    q,
    k.transpose(-2, -1)
) / (head_dim ** 0.5)

# Causal mask
mask = torch.triu(
    torch.ones(
        seq_len,
        seq_len,
        dtype=torch.bool
    ),
    diagonal=1
)

scores = scores.masked_fill(
    mask,
    float("-inf")
)

attention_weights = torch.softmax(
    scores,
    dim=-1
)

# ---------------------------------------------------------
# 7. V weighted sum
# ---------------------------------------------------------

attention_output = torch.matmul(
    attention_weights,
    v
)

print("\nPer-head attention output:")
print(attention_output.shape)

# [batch, heads, seq, head_dim]
# [1, 9, 7, 64]

# ---------------------------------------------------------
# 8. Concatenate the 9 heads
# ---------------------------------------------------------

attention_output = attention_output.transpose(
    1,
    2
)

# [batch, seq, heads, head_dim]

attention_output = attention_output.reshape(
    batch,
    seq_len,
    num_q_heads * head_dim
)

print("\nAfter concatenating heads:")
print(attention_output.shape)

# ---------------------------------------------------------
# 9. Output projection
# ---------------------------------------------------------

print("\nOutput projection:")
print("O projection weight:")
print(attn.o_proj.weight.shape)

projected = torch.matmul(
    attention_output,
    attn.o_proj.weight.float().T
)

if attn.o_proj.bias is not None:
    projected += attn.o_proj.bias.float()

print("Projected attention output:")
print(projected.shape)

# ---------------------------------------------------------
# 10. Residual connection
# ---------------------------------------------------------

residual = x.float() + projected

print("\nResidual connection:")
print("X:", x.shape)
print("Attention output:", projected.shape)
print("Residual:", residual.shape)

# ---------------------------------------------------------
# 11. Compare before / after residual
# ---------------------------------------------------------

token_index = -1

x_token = x[0, token_index].float()
projected_token = projected[0, token_index]
residual_token = residual[0, token_index]

print("\nFinal token:", tokens[token_index])

print("\nFirst 10 values:")

print("Original X:")
print(x_token[:10])

print("\nAttention projected:")
print(projected_token[:10])

print("\nResidual X + Attention:")
print(residual_token[:10])

# ---------------------------------------------------------
# 12. Statistics
# ---------------------------------------------------------

print("\nStatistics:")

print(
    "||X|| =",
    torch.norm(x_token).item()
)

print(
    "||Attention|| =",
    torch.norm(projected_token).item()
)

print(
    "||Residual|| =",
    torch.norm(residual_token).item()
)

# ---------------------------------------------------------
# 13. LayerNorm after residual
# ---------------------------------------------------------

# NOTE:
# SmolLM2/Llama architecture uses RMSNorm rather than
# classic LayerNorm.

post_attention_norm = layer.post_attention_layernorm

normalized = post_attention_norm(
    residual.to(x.dtype)
)

normalized = normalized.float()

print("\nAfter post-attention RMSNorm:")
print(normalized.shape)

print("\nFinal token after RMSNorm:")
print(normalized[0, token_index, :10])