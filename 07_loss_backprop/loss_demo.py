import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.eval()

# ---------------------------------------------------------
# Training example
# ---------------------------------------------------------

text = "A Value Object is defined by its identity"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print("Tokens:")
print(tokens)

# ---------------------------------------------------------
# 1. Run model
# ---------------------------------------------------------

with torch.no_grad():

    outputs = model(
        input_ids=input_ids
    )

logits = outputs.logits

print("\nLogits shape:")
print(logits.shape)

# [batch, sequence, vocabulary]
#
# [1, N, 49152]

# ---------------------------------------------------------
# 2. Next-token prediction
# ---------------------------------------------------------

# For every position:
#
# input:
#   A
#
# target:
#   Value
#
# input:
#   A Value
#
# target:
#   Object
#
# etc.

shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]

print("\nShifted logits:")
print(shift_logits.shape)

print("\nTarget tokens:")
print(shift_labels.shape)

# ---------------------------------------------------------
# 3. Cross entropy loss
# ---------------------------------------------------------

loss = F.cross_entropy(
    shift_logits.reshape(-1, model.config.vocab_size),
    shift_labels.reshape(-1)
)

print("\nOverall loss:")
print(loss.item())

# ---------------------------------------------------------
# 4. Inspect one prediction
# ---------------------------------------------------------

position = 0

logits_at_position = shift_logits[0, position]

target_id = shift_labels[0, position]

target_token = tokenizer.decode(
    [target_id.item()]
)

probabilities = torch.softmax(
    logits_at_position.float(),
    dim=-1
)

target_probability = probabilities[
    target_id
]

target_loss = -torch.log(
    target_probability
)

print("\n------------------------------------------------")
print("Prediction at position 0")
print("------------------------------------------------")

print("Input token:")
print(repr(tokens[position]))

print("Correct next token:")
print(repr(target_token))

print("Target token ID:")
print(target_id.item())

print("Probability assigned to correct token:")
print(target_probability.item())

print("Loss for this prediction:")
print(target_loss.item())

# ---------------------------------------------------------
# 5. Top predictions
# ---------------------------------------------------------

values, indices = torch.topk(
    probabilities,
    10
)

print("\nTop 10 predictions:")

for rank, (prob, index) in enumerate(
    zip(values, indices),
    start=1
):

    token = tokenizer.decode(
        [index.item()]
    )

    marker = ""

    if index.item() == target_id.item():
        marker = "  <-- CORRECT"

    print(
        f"{rank:2d}. "
        f"{repr(token):15s} "
        f"{prob.item():.6f}"
        f"{marker}"
    )

# ---------------------------------------------------------
# 6. Demonstrate loss mathematically
# ---------------------------------------------------------

print("\n------------------------------------------------")
print("Loss examples")
print("------------------------------------------------")

for p in [0.9, 0.5, 0.1, 0.01]:

    example_loss = -torch.log(
        torch.tensor(p)
    )

    print(
        f"P(correct) = {p:4.2f}"
        f"  -> loss = {example_loss.item():.4f}"
    )