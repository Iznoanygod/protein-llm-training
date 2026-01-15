import logging
logging.basicConfig(
    filename='romeupdate.log', 
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")
logger = logging.getLogger("romeupdate")
logger.info("update started")
run_name="roserun"
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('hf_token')
    userdata.get('hf')
except:
    os.environ["HF_TOKEN"] = ""
    os.environ["HF_HOME"] = "/work/nvme/bdyk/apark4/huggingface"
#os.environ["UNSLOTH_VLLM_STANDBY"] = "0" # [NEW] Extra 30% context lengths!
#from unsloth import FastLanguageModel, FastModel
import torch
import re
import random
import time
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from math_verify import parse, verify
import trackio
import gc

max_seq_length = 4096
#max_prompt_length = 2048
lora_rank = 16

base_model_id="meta-llama/Llama-3.1-8B-Instruct"
lora_id = "math_lora"
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
    
    # Find the LAST \boxed{ and then match balanced braces
    start_pattern = r"\\boxed\{"
    matches = list(re.finditer(start_pattern, expr))
    if not matches:
        return None
    
    # Use the last match
    last_match = matches[-1]
    
    # Start after \boxed{
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
        # idx is now one past the closing brace
        return expr[start_idx:idx-1].strip()
    
    return None

def correctness_reward_func(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for completion, ground in zip(completions, ground_truth):
        print(completion, ground)
        
        c = get_answer(completion[0]["content"])
        g = get_answer(ground)
        reward = 2.0 if verify(parse(c), parse(g)) else 0
        rewards.append(reward)
    return rewards

def format_reward_func(prompts, completions, ground_truth, **kwargs):
    """Improved format reward function with granular partial credit."""
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    scores = []
    
    for response in responses:
        score = 0.0
        
        has_reasoning_open = "<reasoning>" in response
        if has_reasoning_open:
            score += 0.125
        
        has_reasoning_close = "</reasoning>" in response
        if has_reasoning_close:
            score += 0.125
        
        has_answer_open = "<answer>" in response
        if has_answer_open:
            score += 0.125
        
        has_answer_close = "</answer>" in response
        if has_answer_close:
            score += 0.125
        
        has_boxed = bool(re.search(r"\\boxed\{", response))
        if has_boxed:
            score += 0.2
        
        if has_reasoning_open and has_reasoning_close and has_answer_open and has_answer_close:
            reasoning_open_idx = response.find("<reasoning>")
            reasoning_close_idx = response.find("</reasoning>")
            answer_open_idx = response.find("<answer>")
            answer_close_idx = response.find("</answer>")
            
            if (reasoning_open_idx < reasoning_close_idx < answer_open_idx < answer_close_idx):
                score += 0.2
            elif (reasoning_open_idx < reasoning_close_idx and 
                  answer_open_idx < answer_close_idx):
                score += 0.1
        
        if has_boxed and has_answer_open and has_answer_close:
            answer_open_idx = response.find("<answer>")
            answer_close_idx = response.find("</answer>")
            boxed_idx = response.find("\\boxed{")
            
            if answer_open_idx < boxed_idx < answer_close_idx:
                score += 0.1
        
        scores.append(min(score, 1.0))
    
    return scores

def to_prompt_completion(example):
    return {
        "prompt": [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['problem']}
            
        ],
        "ground_truth": str(example["solution"]).strip(),
    }

#dataset = load_dataset("qwedsacf/competition_math", split="train")
#mapped = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)
def load_llama_or_latest_checkpoint(
    base_model_id: str,
    lora_id: str,
    dtype=torch.bfloat16,
    device_map="auto",
):
    """
    If `output_dir` contains checkpoints for this run, load the latest one.
    Otherwise, load the base model from Hugging Face Hub.
    """

    last_checkpoint = None
    
    if os.path.isdir(lora_id):
        last_checkpoint = get_last_checkpoint(lora_id)
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    iteration = 0
    if last_checkpoint is not None:
        logger.info(f"Found LoRA checkpoint at: {last_checkpoint}")
        logger.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device_map,
        )
        m = re.search(r"checkpoint-(\d+)", last_checkpoint)
        if m:
            iteration = int(m.group(1))
        else:
            logger.info(f"Warning: could not parse iteration from {basename}, leaving iteration=0")
        # Attach LoRA adapter weights
        logger.info("Applying LoRA adapter from checkpoint...")
        model = PeftModel.from_pretrained(base_model, last_checkpoint, is_trainable=True)
        loaded_from = last_checkpoint
    else:
        logger.info(f"No checkpoint found, loading base model: {base_model_id}")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device_map,
        )
        model = get_peft_model(model, lora_config)
        loaded_from = base_model_id

    return model, tokenizer, loaded_from, iteration
gpu_count = torch.cuda.device_count()
def memory_stats():
    logger.info(f"memory allocated: {[torch.cuda.memory_allocated(i)/1024**2 for i in range(gpu_count)]}")
    logger.info(f"memory reserved: {[torch.cuda.memory_reserved(i)/1024**2 for i in range(gpu_count)]}")
    for i in range(gpu_count):
        logger.info(torch.cuda.memory_summary(i))
        
def update():
    #need to let the previous node get cleaned up
    #time.sleep(10)
    
    model, tokenizer, loaded_from, iteration = load_llama_or_latest_checkpoint(
        base_model_id=base_model_id,
        lora_id=lora_id,
        dtype=torch.bfloat16,
    )
    
    trackio.init(project="huggingface", space_id="iznoanygod/trackio", name=f"{run_name}-{iteration}", embed=False)
    model.print_trainable_parameters()
    logger.info("Model Loaded...")
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
        bf16=True,
        gradient_checkpointing=False,
        num_generations = 8, # Decrease if out of memory
#        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length,
        num_train_epochs = 1, # Set to 1 for a full training run
        save_steps = 10,
        max_steps = iteration+50,
        max_grad_norm = 0.1,
        report_to = "trackio", # Can use Weights & Biases
        run_name=f"roserun-{iteration}",
        output_dir = lora_id,
    )
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    mapped = dataset.map(to_prompt_completion, remove_columns=dataset.column_names).shuffle()
    logger.info("Configured...")
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = mapped,
    )
    logger.info("Starting Training...")
    memory_stats()
    if loaded_from == base_model_id:
        trainer.train(resume_from_checkpoint=False)
    else:
        # resume from checkpoint needs changing max step
        trainer.train(resume_from_checkpoint=loaded_from)
    logger.info("Finished Training...")
    memory_stats()
#    del model
#    del tokenizer
#    del trainer
#    del dataset
#    del mapped
    with torch.no_grad():
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Cleaned up memory...")
    memory_stats()

if __name__ == "__main__":
    update()