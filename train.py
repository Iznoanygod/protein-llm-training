import os
import sys
import time
os.environ["HF_TOKEN"] = ""
os.environ["HF_HOME"] = "/anvil/scratch/x-apark4/cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = 128004
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model.generation_config = GenerationConfig.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

dataset = load_dataset("qwedsacf/competition_math", split="train")
def to_prompt_completion(example):
    return {
        "prompt": (
            example['problem']
        ),
        "ground_truth": str(example["solution"]).strip(),
    }
mapped = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)

import re
def get_answer(expr: str):
    match = re.search(r"\\boxed\{(.+?)\}", expr)
    if match:
        return match.group(1).strip()
    return None

def rewards_func(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for prompt, completion, ground in zip(prompts, completions, ground_truth):
        print(completion)
        c = get_answer(completion)
        g = get_answer(ground)
        reward = 1.0 if g == c else 0
        rewards.append(reward)
    return rewards

training_args = GRPOConfig(output_dir="llamamath")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=rewards_func,
    args=training_args,
    train_dataset=mapped,
)
trainer.train()