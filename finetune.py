import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

BASE_MODEL = "./output/domain-model"
OUTPUT_DIR = "./output/final-model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading domain-pretrained model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# --------------------------------------------------
# 1. Configure LoRA
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

print("\nLoRA parameter information:")
model.print_trainable_parameters()

# --------------------------------------------------
# 2. Load instruction data
# --------------------------------------------------

examples = []

with open("data/finetune.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        text = (
            "### Instruction:\n"
            + item["instruction"]
            + "\n\n"
            + "### Response:\n"
            + item["response"]
            + tokenizer.eos_token
        )

        examples.append(text)

# --------------------------------------------------
# 3. Tokenize
# --------------------------------------------------

def tokenize(text):
    result = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding="max_length",
    )

    result["labels"] = result["input_ids"].copy()

    return result


tokenized_data = [tokenize(x) for x in examples]


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            key: torch.tensor(value)
            for key, value in self.data[index].items()
        }


dataset = SimpleDataset(tokenized_data)

# --------------------------------------------------
# 4. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=5,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=1,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none",

    use_cpu=True,
)

# --------------------------------------------------
# 5. Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("\nStarting LoRA fine-tuning...\n")

trainer.train()

# --------------------------------------------------
# 6. Save adapter
# --------------------------------------------------

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFine-tuning complete!")
print("LoRA adapter saved to:", OUTPUT_DIR)