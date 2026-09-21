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

print("Tokens:")
for i, token in enumerate(tokens):
    print(f"{i}: {token}")

# ------------------------------------------------
# Model configuration
# ------------------------------------------------

num_q_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim

print("\nModel configuration:")
print("Q heads:", num_q_heads)
print("KV heads:", num_kv_heads)
print("Head dimension:", head_dim)

# ------------------------------------------------
# Get first Transformer layer
# ------------------------------------------------

layer = model.model.layers[0]

with torch.no_grad():

    # Embeddings
    hidden = model.model.embed_tokens(inputs["input_ids"])

    # Layer normalization
    hidden = layer.input_layernorm(hidden)

    # ------------------------------------------------
    # Create Q, K, V
    # ------------------------------------------------

    Q = layer.self_attn.q_proj(hidden)
    K = layer.self_attn.k_proj(hidden)
    V = layer.self_attn.v_proj(hidden)

    # Convert to FP32 for analysis
    Q = Q.float()
    K = K.float()
    V = V.float()

    print("\nRaw projections:")
    print("Q:", Q.shape)
    print("K:", K.shape)
    print("V:", V.shape)

    # ------------------------------------------------
    # Split into heads
    # ------------------------------------------------

    Q = Q.view(
        1,
        -1,
        num_q_heads,
        head_dim
    ).transpose(1, 2)

    K = K.view(
        1,
        -1,
        num_kv_heads,
        head_dim
    ).transpose(1, 2)

    V = V.view(
        1,
        -1,
        num_kv_heads,
        head_dim
    ).transpose(1, 2)

    print("\nAfter splitting:")
    print("Q:", Q.shape)
    print("K:", K.shape)
    print("V:", V.shape)

    # ------------------------------------------------
    # Grouped Query Attention
    # ------------------------------------------------

    repeat_factor = num_q_heads // num_kv_heads

    K = K.repeat_interleave(
        repeat_factor,
        dim=1
    )

    V = V.repeat_interleave(
        repeat_factor,
        dim=1
    )

    print("\nAfter GQA:")
    print("Q:", Q.shape)
    print("K:", K.shape)
    print("V:", V.shape)

    # ------------------------------------------------
    # Calculate attention scores
    # ------------------------------------------------

    scores = Q @ K.transpose(-2, -1)

    scores = scores / (head_dim ** 0.5)

    # ------------------------------------------------
    # Causal mask
    # ------------------------------------------------

    seq_len = scores.shape[-1]

    mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
    ).bool()

    scores = scores.masked_fill(
        mask,
        float("-inf")
    )

    # ------------------------------------------------
    # Softmax
    # ------------------------------------------------

    attention = torch.softmax(
        scores,
        dim=-1
    )

    print("\nAttention shape:")
    print(attention.shape)

    # ------------------------------------------------
    # HEAD 0 - attention matrix
    # ------------------------------------------------

    print("\n========== HEAD 0 ==========")

    head = 0
    matrix = attention[0, head]

    for i, token in enumerate(tokens):

        values = matrix[i].tolist()

        print(
            f"{token:12s} -> "
            + " ".join(
                f"{v:.3f}" for v in values
            )
        )

    # ------------------------------------------------
    # Calculate V-weighted output for "its"
    # ------------------------------------------------

    token_index = 6

    weights = attention[
        0,
        head,
        token_index
    ]

    values = V[0, head]

    output = weights @ values

    print("\n========== V-WEIGHTED OUTPUT ==========")

    print("\nAttention weights for 'its':")

    for token, weight in zip(tokens, weights):

        print(
            f"{token:12s}: {weight:.4f}"
        )

    print("\nV matrix shape:")
    print(values.shape)

    print("\nAttention output shape:")
    print(output.shape)

    print("\nFirst 10 values of attention output:")

    for i, value in enumerate(output[:10]):

        print(
            f"{i}: {value.item():.6f}"
        )