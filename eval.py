#!/usr/bin/env python
import argparse, os, re, csv, math, sys, time
from typing import List, Optional
os.environ["HF_TOKEN"] = ""
os.environ["HF_HOME"] = "/work/hdd/bdyk/apark4/huggingface"
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



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

def get_answer(expr: str):
    match = re.search(r"\\boxed\{(.+?)\}", expr)
    if match:
        return match.group(1).strip()
    return None

def correctness_func(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for prompt, completion, ground in zip(prompts, completions, ground_truth):
        c = get_answer(completion[0]["content"])
        g = get_answer(ground)
        reward = 2.0 if g == c else 0
        rewards.append(reward)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model id or path (base or merged).")
    args = parser.parse_args()


if __name__ == "__main__":
    main()