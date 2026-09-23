import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "./output/domain-model"
TEXT = "A Value Object is defined by its"

# ------------------------------------------------
# Load model
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

inputs = tokenizer(TEXT, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

num_q_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim

layer = model.model.layers[0]

# ------------------------------------------------
# Calculate attention
# ------------------------------------------------

with torch.no_grad():

    hidden = model.model.embed_tokens(inputs["input_ids"])
    hidden = layer.input_layernorm(hidden)

    Q = layer.self_attn.q_proj(hidden)
    K = layer.self_attn.k_proj(hidden)

    Q = Q.float()
    K = K.float()

    # Split into heads
    Q = Q.view(
        1, -1, num_q_heads, head_dim
    ).transpose(1, 2)

    K = K.view(
        1, -1, num_kv_heads, head_dim
    ).transpose(1, 2)

    # GQA: 3 KV heads -> 9 heads
    repeat_factor = num_q_heads // num_kv_heads

    K = K.repeat_interleave(
        repeat_factor,
        dim=1
    )

    # Attention scores
    scores = Q @ K.transpose(-2, -1)

    scores = scores / (head_dim ** 0.5)

    # Causal mask
    seq_len = scores.shape[-1]

    mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
    ).bool()

    scores = scores.masked_fill(
        mask,
        float("-inf")
    )

    # Softmax
    attention = torch.softmax(
        scores,
        dim=-1
    )

# ------------------------------------------------
# Inspect attention for "its"
# ------------------------------------------------

token_index = len(tokens) - 1

print("\nTokens:")
for i, token in enumerate(tokens):
    print(f"{i}: {token}")

print("\n========================================")
print("Attention to 'its' across all 9 heads")
print("========================================")

for head in range(num_q_heads):

    weights = attention[
        0,
        head,
        token_index
    ]

    print(f"\nHEAD {head}")

    for token, weight in zip(tokens, weights):

        print(
            f"{token:12s} {weight.item():.4f}"
        )