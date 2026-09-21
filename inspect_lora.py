from safetensors.torch import load_file
import os

LORA_MODEL = "./output/final-model"

adapter_path = os.path.join(LORA_MODEL, "adapter_model.safetensors")

weights = load_file(adapter_path)

total = 0

print("LoRA tensors:\n")

for name, tensor in weights.items():
    count = tensor.numel()
    total += count
    print(f"{name:70} {tuple(tensor.shape)}  {count:,}")

print("\n" + "=" * 70)
print(f"Total LoRA parameters: {total:,}")