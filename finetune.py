import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

model_id = "deepseek-ai/DeepSeek-V3.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

dataset = load_dataset("proteingpt2")

def format_example(example):
    return {
        "instruction": "Generate a protein sequence.",
        "input": "",
        "output": example["text"]
    }

dataset = dataset.map(format_example)

def tokenize(example):
    prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput:"
    target = example["output"]
    full = prompt + " " + target
    tokens = tokenizer(full, truncation=True, padding="max_length", max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=False, remove_columns=dataset["train"].column_names)

args = TrainingArguments(
    output_dir="./deepseek-protein-finetune",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    fp16=True,
    logging_steps=100,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"]
)

trainer.train()

trainer.save_model("./deepseek-protein-finetune-final")
tokenizer.save_pretrained("./deepseek-protein-finetune-final")
