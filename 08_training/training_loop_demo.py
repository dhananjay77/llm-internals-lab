import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token
model.train()

text = """
A Value Object is defined by its attributes rather than identity.
An Entity is defined by a stable identity over time.
An Aggregate maintains consistency boundaries in a domain model.
Bounded Contexts separate different domain models.
"""

inputs = tokenizer(
    text,
    return_tensors="pt"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)

print("Starting training...\n")

for step in range(20):

    # -----------------------------
    # Forward pass
    # -----------------------------
    outputs = model(
        input_ids=inputs["input_ids"],
        labels=inputs["input_ids"]
    )

    loss = outputs.loss

    # -----------------------------
    # Backpropagation
    # -----------------------------
    optimizer.zero_grad()

    loss.backward()

    # -----------------------------
    # Weight update
    # -----------------------------
    optimizer.step()

    print(
        f"Step {step + 1:02d} | "
        f"Loss: {loss.item():.6f}"
    )

print("\nTraining complete.")