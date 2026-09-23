import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------
# Add LoRA
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.train()

model.print_trainable_parameters()

# --------------------------------------------------
# Training data
# --------------------------------------------------

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

# --------------------------------------------------
# Capture parameters BEFORE training
# --------------------------------------------------

base_parameter = model.base_model.model.model.layers[
    0
].self_attn.q_proj.base_layer.weight

lora_A = model.base_model.model.model.layers[
    0
].self_attn.q_proj.lora_A.default.weight

lora_B = model.base_model.model.model.layers[
    0
].self_attn.q_proj.lora_B.default.weight

base_before = base_parameter.detach().clone()
A_before = lora_A.detach().clone()
B_before = lora_B.detach().clone()

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

# --------------------------------------------------
# Training loop
# --------------------------------------------------

print("\nTraining...\n")

for step in range(20):

    outputs = model(
        input_ids=inputs["input_ids"],
        labels=inputs["input_ids"]
    )

    loss = outputs.loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Step {step + 1:02d} | "
        f"Loss: {loss.item():.6f}"
    )

# --------------------------------------------------
# Compare parameters AFTER training
# --------------------------------------------------

base_after = base_parameter.detach()
A_after = lora_A.detach()
B_after = lora_B.detach()

print("\n--------------------------------")
print("PARAMETER CHANGES")
print("--------------------------------")

print(
    "\nBase Q projection change:"
)
print(
    (base_after - base_before).abs().mean().item()
)

print(
    "\nLoRA A change:"
)
print(
    (A_after - A_before).abs().mean().item()
)

print(
    "\nLoRA B change:"
)
print(
    (B_after - B_before).abs().mean().item()
)