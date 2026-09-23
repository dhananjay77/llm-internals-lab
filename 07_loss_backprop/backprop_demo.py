import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/domain-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.train()

text = "A Value Object is defined by its identity"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]

tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)

print("Tokens:")
print(tokens)

# ---------------------------------------------------------
# 1. Forward pass
# ---------------------------------------------------------

outputs = model(
    input_ids=input_ids,
    labels=input_ids
)

loss = outputs.loss

print("\nLoss:")
print(loss.item())

# ---------------------------------------------------------
# 2. Clear existing gradients
# ---------------------------------------------------------

model.zero_grad()

# ---------------------------------------------------------
# 3. BACKPROPAGATION
# ---------------------------------------------------------

loss.backward()

print("\nBackward pass complete!")

# ---------------------------------------------------------
# 4. Inspect gradients
# ---------------------------------------------------------

print("\nGradient examples:")
print("--------------------------------")

parameters_to_check = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "lm_head.weight",
]

for name in parameters_to_check:

    parameters = {
    "embed_tokens.weight": model.model.embed_tokens.weight,
    "q_proj.weight": model.model.layers[0].self_attn.q_proj.weight,
    "v_proj.weight": model.model.layers[0].self_attn.v_proj.weight,
    "gate_proj.weight": model.model.layers[0].mlp.gate_proj.weight,
    "down_proj.weight": model.model.layers[0].mlp.down_proj.weight,
}

for name, parameter in parameters.items():
    print(f"\n{name}")
    print("  shape:", tuple(parameter.shape))
    print("  grad mean:", parameter.grad.mean().item())
    print("  grad std:", parameter.grad.std().item())
    print("  grad mean abs:", parameter.grad.abs().mean().item())
    print("  grad norm:", parameter.grad.norm().item())