import logging
logging.basicConfig(
    filename='checkpoint_eval.log', 
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")
logger = logging.getLogger("checkpoint_eval")
logger.info("checkpoint evaluation started")

import os
import glob
import json
import torch
import re
import gc
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify
import matplotlib.pyplot as plt

# Configuration
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
lora_dir = "math_lora"  # Directory containing checkpoints

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('hf_token')
except:
    os.environ["HF_TOKEN"] = ""
    os.environ["HF_HOME"] = "/work/nvme/bdyk/apark4/huggingface"

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
Approach this problem systematically:
1. Identify what is given and what you need to find
2. State any relevant theorems, formulas, or definitions
3. Set up the problem (equations, diagrams, etc.)
4. Work through the calculations step-by-step
5. Verify your answer makes sense

Show all work clearly. Each step should follow logically from the previous one.
Use LaTeX to format all mathematical expressions: use $...$ for inline math and $$...$$ for displayed equations.
</reasoning>

<answer>
State your final answer clearly. Simplify fully and use the simplest form.

Use \\boxed{...} for your final answer. Examples:
- For a number: \\boxed{42} or \\boxed{\\frac{1}{2}} or \\boxed{3.14}
- For a variable: \\boxed{x = 5}
- For multiple solutions: \\boxed{x = 1, 2, 3}
- For ranges: \\boxed{0 < x < 5}
- For no solution: \\boxed{\\text{no solution}}
- For expressions: \\boxed{2x + 1}

Rules:
- Simplify all fractions to lowest terms
- Use exact forms (fractions, radicals) unless decimals are required
- Do not include units in the boxed answer unless specified in the problem
- If the answer is undefined or indeterminate, state that clearly
- Only box the final answer, not intermediate steps
</answer>
"""

def get_answer(expr: str):
    if expr is None:
        return None
    
    start_pattern = r"\\boxed\{"
    matches = list(re.finditer(start_pattern, expr))
    if not matches:
        return None
    
    last_match = matches[-1]
    start_idx = last_match.end()
    brace_count = 1
    idx = start_idx
    
    while idx < len(expr) and brace_count > 0:
        if expr[idx] == '{':
            brace_count += 1
        elif expr[idx] == '}':
            brace_count -= 1
        idx += 1
    
    if brace_count == 0:
        return expr[start_idx:idx-1].strip()
    
    return None

def correctness_reward_func(completions, ground_truth):
    rewards = []
    for completion, ground in zip(completions, ground_truth):
        c = get_answer(completion)
        g = get_answer(ground)
        try:
            reward = 1.0 if verify(parse(c), parse(g)) else 0.0
        except:
            reward = 0.0
        rewards.append(reward)
    return rewards

def to_prompt_completion(example):
    return {
        "prompt": [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['problem']}
        ],
        "ground_truth": str(example["solution"]).strip(),
    }

def get_all_checkpoints(lora_dir):
    """Find all checkpoint directories and sort by iteration number."""
    checkpoint_pattern = os.path.join(lora_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    checkpoint_info = []
    for cp in checkpoints:
        match = re.search(r"checkpoint-(\d+)", cp)
        if match:
            iteration = int(match.group(1))
            checkpoint_info.append((iteration, cp))
    
    # Sort by iteration number
    checkpoint_info.sort(key=lambda x: x[0])
    return checkpoint_info

def evaluate_checkpoint(checkpoint_path, base_model, tokenizer, test_questions, batch_size=8):
    """Evaluate a single checkpoint on the test questions."""
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
    model.eval()
    
    total_correct = 0
    
    for q_idx, question in enumerate(test_questions):
        messages = [question["prompt"]] * batch_size
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_length = inputs.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=4096,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        ground_truths = [question["ground_truth"]] * batch_size
        rewards = correctness_reward_func(responses, ground_truths)
        
        question_correct = any(r > 0 for r in rewards)
        if question_correct:
            total_correct += 1
        
        del inputs, outputs, generated_tokens
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    pass_at_k = total_correct / len(test_questions)
    return pass_at_k

def evaluate_base_model(base_model, tokenizer, test_questions, batch_size=8):
    """Evaluate the base model without any LoRA adapter."""
    logger.info("Evaluating base model (no LoRA)")
    
    base_model.eval()
    total_correct = 0
    
    for q_idx, question in enumerate(test_questions):
        messages = [question["prompt"]] * batch_size
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
        ).to(base_model.device)
        
        input_length = inputs.shape[1]
        
        with torch.no_grad():
            outputs = base_model.generate(
                inputs,
                max_new_tokens=4096,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        ground_truths = [question["ground_truth"]] * batch_size
        rewards = correctness_reward_func(responses, ground_truths)
        
        question_correct = any(r > 0 for r in rewards)
        if question_correct:
            total_correct += 1
        
        del inputs, outputs, generated_tokens
    
    torch.cuda.empty_cache()
    gc.collect()
    
    pass_at_k = total_correct / len(test_questions)
    return pass_at_k

def evaluate_all_checkpoints(lora_dir=lora_dir, num_questions=50, batch_size=8, eval_base=True):
    """
    Evaluate all checkpoints and plot the progression.
    
    Args:
        lora_dir: Directory containing checkpoint-* folders
        num_questions: Number of questions to test on
        batch_size: Number of responses per question (k in pass@k)
        eval_base: Whether to also evaluate the base model (iteration 0)
    
    Returns:
        Dictionary with iterations and their pass@k scores
    """
    # Load tokenizer and base model once
    logger.info(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load and prepare test dataset (use same seed for consistency)
    logger.info("Loading dataset...")
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    mapped = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)
    shuffled = mapped.shuffle(seed=42)
    test_questions = shuffled.select(range(min(num_questions, len(shuffled))))
    
    # Get all checkpoints
    checkpoints = get_all_checkpoints(lora_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints: {[cp[0] for cp in checkpoints]}")
    
    results = {}
    
    # Optionally evaluate base model first
    if eval_base:
        logger.info("Evaluating base model...")
        base_score = evaluate_base_model(base_model, tokenizer, test_questions, batch_size)
        results[0] = base_score
        logger.info(f"Base model (iteration 0): pass@{batch_size} = {base_score:.4f}")
        print(f"Iteration 0 (base): pass@{batch_size} = {base_score:.4f}")
    
    # Evaluate each checkpoint
    for iteration, checkpoint_path in checkpoints:
        logger.info(f"Evaluating iteration {iteration}...")
        score = evaluate_checkpoint(checkpoint_path, base_model, tokenizer, test_questions, batch_size)
        results[iteration] = score
        logger.info(f"Iteration {iteration}: pass@{batch_size} = {score:.4f}")
        print(f"Iteration {iteration}: pass@{batch_size} = {score:.4f}")
    
    # Clean up
    del base_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save results to JSON
    results_file = "checkpoint_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Plot results
    plot_results(results, batch_size)
    
    return results

def plot_results(results, batch_size):
    """Plot the pass@k progression over iterations."""
    iterations = sorted(results.keys())
    scores = [results[i] for i in iterations]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, scores, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel(f'pass@{batch_size}', fontsize=12)
    plt.title(f'Model Performance Over Training (pass@{batch_size})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(iterations, scores)):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig('checkpoint_progression.png', dpi=150)
    plt.savefig('checkpoint_progression.pdf')
    logger.info("Plots saved to checkpoint_progression.png and checkpoint_progression.pdf")
    print("Plots saved to checkpoint_progression.png and checkpoint_progression.pdf")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and plot progression")
    parser.add_argument("--lora_dir", type=str, default="math_lora",
                       help="Directory containing checkpoint-* folders")
    parser.add_argument("--num_questions", type=int, default=50,
                       help="Number of questions to test on")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of responses per question (k in pass@k)")
    parser.add_argument("--skip_base", action="store_true",
                       help="Skip evaluating the base model")
    args = parser.parse_args()
    
    results = evaluate_all_checkpoints(
        lora_dir=args.lora_dir,
        num_questions=args.num_questions,
        batch_size=args.batch_size,
        eval_base=not args.skip_base
    )
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for iteration in sorted(results.keys()):
        print(f"Iteration {iteration}: {results[iteration]:.4f}")
