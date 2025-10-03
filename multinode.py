import re
import os
os.environ["HF_TOKEN"] = ""
os.environ["HF_HOME"] = "/anvil/scratch/x-apark4/cache"
import torch
import re
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import GRPOConfig, GRPOTrainer
max_seq_length = 2048
lora_rank = 32

model_name = "meta-llama/Llama-3.2-3B-Instruct"

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
)
lora_model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
You must use LaTeX to format mathematical expressions, and you must use \\boxed{...} to indicate the final answer.
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def to_prompt_completion(example):
    return {
        "prompt": [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['problem']}
            
        ],
        "ground_truth": str(example["solution"]).strip(),
    }

dataset = load_dataset("qwedsacf/competition_math", split="train")
mapped = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)

def get_answer(expr: str):
    match = re.search(r"\\boxed\{(.+?)\}", expr)
    if match:
        return match.group(1).strip()
    return None

def correctness_reward_func(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for prompt, completion, ground in zip(prompts, completions, ground_truth):
        c = get_answer(completion[0]["content"])
        g = get_answer(ground)
        reward = 2.0 if g == c else 0
        rewards.append(reward)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

max_prompt_length = 512

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    generation_batch_size = 8,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = lora_model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = mapped,
)
trainer.train()