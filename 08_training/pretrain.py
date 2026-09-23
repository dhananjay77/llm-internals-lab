from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

MODEL = "HuggingFaceTB/SmolLM2-135M"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# SmolLM2 has no pad token by default
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)

# Load our domain corpus
with open("data/pretrain.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = Dataset.from_dict({"text": [text]})

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
    )

tokenized = dataset.map(
    tokenize,
    remove_columns=["text"]
)

# Causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./output/domain-model",

    num_train_epochs=10,
    per_device_train_batch_size=1,

    logging_steps=1,
    save_strategy="epoch",

    learning_rate=5e-5,

    report_to="none",

    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("\nStarting continued pre-training...\n")

trainer.train()

trainer.save_model("./output/domain-model")
tokenizer.save_pretrained("./output/domain-model")

print("\nTraining complete!")
print("Model saved to ./output/domain-model")