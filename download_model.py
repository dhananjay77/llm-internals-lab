from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "HuggingFaceTB/SmolLM2-135M"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL)

print("Model loaded successfully!")
print(f"Parameters: {model.num_parameters():,}")