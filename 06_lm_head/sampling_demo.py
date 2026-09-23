import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token
model.eval()

prompt = "A Value Object is defined by"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Last token's logits
logits = outputs.logits[0, -1, :]

print("Logits shape:")
print(logits.shape)

# --------------------------------------------------
# Temperature
# --------------------------------------------------

temperature = 1.0

scaled_logits = logits / temperature

# --------------------------------------------------
# Softmax
# --------------------------------------------------

probabilities = torch.softmax(scaled_logits, dim=-1)

print("\nProbability sum:")
print(probabilities.sum().item())

# --------------------------------------------------
# Top 10
# --------------------------------------------------

top_probs, top_ids = torch.topk(
    probabilities,
    k=10
)

print("\nTop 10 next-token candidates:\n")

for rank, (prob, token_id) in enumerate(
    zip(top_probs, top_ids),
    start=1
):
    token = tokenizer.decode([token_id.item()])

    print(
        f"{rank:2d}. "
        f"{repr(token):20s} "
        f"{prob.item():.6f}"
    )

# --------------------------------------------------
# Greedy choice
# --------------------------------------------------

greedy_id = torch.argmax(probabilities)

print("\nGreedy choice:")
print(repr(tokenizer.decode([greedy_id.item()])))

# --------------------------------------------------
# Sampling
# --------------------------------------------------

sampled_id = torch.multinomial(
    probabilities,
    num_samples=1
)

print("\nSampled choice:")
print(repr(tokenizer.decode([sampled_id.item()])))