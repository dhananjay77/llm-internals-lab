import torch
from safetensors.torch import load_file

path = "./output/final-model/adapter_model.safetensors"

weights = load_file(path)

for name, tensor in weights.items():
    if "layers.0.self_attn.q_proj" in name:
        print("\n", name)
        print("Shape:", tuple(tensor.shape))
        print("Mean:", tensor.mean().item())
        print("Std :", tensor.std().item())
        print("Min :", tensor.min().item())
        print("Max :", tensor.max().item())

        print("\nFirst 5 values:")
        print(tensor.flatten()[:5])
