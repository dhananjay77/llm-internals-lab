import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

model.train()

text = "A Value Object is defined by its identity"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

# --------------------------------------------------
# 1. Loss BEFORE update
# --------------------------------------------------

outputs = model(
    input_ids=inputs["input_ids"],
    labels=inputs["input_ids"]
)

loss_before = outputs.loss

print("Loss BEFORE update:", loss_before.item())


# --------------------------------------------------
# 2. Backpropagation
# --------------------------------------------------

model.zero_grad()

loss_before.backward()


# --------------------------------------------------
# 3. Save one parameter BEFORE update
# --------------------------------------------------

parameter = model.model.layers[0].self_attn.v_proj.weight

weight_before = parameter.detach().clone()

gradient = parameter.grad.detach().clone()

print("\nGradient norm:", gradient.norm().item())


# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)


# --------------------------------------------------
# 5. UPDATE
# --------------------------------------------------

optimizer.step()


# --------------------------------------------------
# 6. Compare weights
# --------------------------------------------------

weight_after = parameter.detach()

difference = weight_after - weight_before

print("\nWeight BEFORE:", weight_before.flatten()[0].item())
print("Weight AFTER :", weight_after.flatten()[0].item())

print("\nWeight change:")
print("  mean abs:", difference.abs().mean().item())
print("  max abs :", difference.abs().max().item())
print("  norm    :", difference.norm().item())


# --------------------------------------------------
# 7. Loss AFTER update
# --------------------------------------------------

with torch.no_grad():

    outputs_after = model(
        input_ids=inputs["input_ids"],
        labels=inputs["input_ids"]
    )

    loss_after = outputs_after.loss

print("\nLoss AFTER update:", loss_after.item())

print("\nLoss change:", loss_after.item() - loss_before.item())