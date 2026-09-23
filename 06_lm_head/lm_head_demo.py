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

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)

# ---------------------------------------------------------
# 1. Final hidden state
# ---------------------------------------------------------

hidden = outputs.hidden_states[-1]

print("\nFinal hidden state:")
print(hidden.shape)

# Expected:
# [1, 7, 576]

# ---------------------------------------------------------
# 2. Final RMSNorm
# ---------------------------------------------------------

final_norm = model.model.norm(hidden)

print("\nAfter final RMSNorm:")
print(final_norm.shape)

# ---------------------------------------------------------
# 3. LM Head
# ---------------------------------------------------------

lm_head = model.lm_head

print("\nLM Head weight:")
print(lm_head.weight.shape)

# Expected:
# [49152, 576]

# ---------------------------------------------------------
# 4. Convert hidden representation into logits
# ---------------------------------------------------------

logits = lm_head(final_norm)

print("\nLogits:")
print(logits.shape)

# Expected:
# [1, 7, 49152]

# We care about the FINAL token.
next_token_logits = logits[0, -1]

print("\nNext-token logits:")
print(next_token_logits.shape)

# ---------------------------------------------------------
# 5. Find highest scoring tokens
# ---------------------------------------------------------

top_k = 20

values, indices = torch.topk(
    next_token_logits,
    top_k
)

print(f"\nTop {top_k} next-token candidates:")

for rank, (value, index) in enumerate(
    zip(values, indices),
    start=1
):
    token = tokenizer.decode([index.item()])

    print(
        f"{rank:2d}. "
        f"{repr(token):20s} "
        f"logit={value.item():.4f}"
    )

# ---------------------------------------------------------
# 6. Softmax
# ---------------------------------------------------------

probabilities = torch.softmax(
    next_token_logits.float(),
    dim=-1
)

print("\nProbability sum:")
print(probabilities.sum().item())

# ---------------------------------------------------------
# 7. Top probabilities
# ---------------------------------------------------------

values, indices = torch.topk(
    probabilities,
    top_k
)

print(f"\nTop {top_k} probabilities:")

for rank, (prob, index) in enumerate(
    zip(values, indices),
    start=1
):
    token = tokenizer.decode([index.item()])

    print(
        f"{rank:2d}. "
        f"{repr(token):20s} "
        f"probability={prob.item():.6f}"
    )

# ---------------------------------------------------------
# 8. Greedy next token
# ---------------------------------------------------------

best_index = torch.argmax(
    probabilities
)

best_token = tokenizer.decode(
    [best_index.item()]
)

print("\nPredicted next token:")
print(repr(best_token))

# ---------------------------------------------------------
# 9. Inspect actual hidden vector
# ---------------------------------------------------------

print("\nFinal hidden representation")
print("First 10 dimensions:")

print(
    final_norm[0, -1, :10]
)

print("\nNumber of logits:")
print(next_token_logits.numel())

print("\nVocabulary size:")
print(model.config.vocab_size)