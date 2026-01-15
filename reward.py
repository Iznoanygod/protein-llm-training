import logging
logging.basicConfig(
    filename='romereward.log', 
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")
logger = logging.getLogger("romereward")
logger.info("reward started")
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
#import trackio
import gc
#trackio.init(project="huggingface", space_id="iznoanygod/trackio", name=run_name, embed=False)

max_seq_length = 4096
max_prompt_length = 2048
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
    for prompt, completion, ground in zip(prompts, completions, ground_truth):
        #print(completion, ground)
        c = get_answer(completion)
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

def check_reward():
    batch_size=8
    num_questions=10
    iteration=1
    model, tokenizer, loaded_from, _ = load_llama_or_latest_checkpoint(
        base_model_id=base_model_id,
        lora_id=lora_id,
        dtype=torch.bfloat16,
    )
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    mapped = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)
    shuffled_dataset = mapped.shuffle()
    test_questions = shuffled_dataset.select(range(min(num_questions, len(shuffled_dataset))))
    
    total_correct=0.0
    logger.info(f"Starting pass@{batch_size} evaluation on {len(test_questions)} questions...")
    memory_stats()
    
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
        #print("tokenized")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=4096,
                do_sample=True,      # sampling
                top_p=0.9,
                temperature=0.7,
                pad_token_id = tokenizer.eos_token_id
                # don't usually mix beam search + sampling;
                # if you want beam search, drop top_p/temperature and set num_beams>1
            )
            
        #print("generated")
        #texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #generated_tokens = outputs[:, input_length:]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ground_truths = [question["ground_truth"]] * batch_size
        #for src, full in zip(batch, texts):
        #    print("PROMPT:", src["prompt"])
        #    print("OUTPUT:", full)
        #    print("-" * 80)
        rewards = correctness_reward_func(messages, responses, ground_truths)
        format_rewards = format_reward_func(messages, responses, ground_truths)
        logger.info("=" * 80)
        logger.info(f"Question {q_idx + 1}")
        logger.info("FULL PROMPT:")
        logger.info(f"System: {question['prompt'][0]['content']}")
        logger.info(f"User: {question['prompt'][1]['content']}")
        logger.info(f"Ground truth: {question['ground_truth']}")
        logger.info("=" * 80)
        for resp_idx, (response, correct_reward, format_reward) in enumerate(zip(responses, rewards, format_rewards)):
            extracted_answer = get_answer(response)
            logger.info(f"Response {resp_idx + 1} | Correct: {correct_reward > 0} | Format: {format_reward} | Extracted answer: {extracted_answer}")
            logger.info(response)
            logger.info("-" * 80)
        question_correct = any(r > 0 for r in rewards)
        if question_correct:
            total_correct += 1
        # Log progress periodically
        current_accuracy = total_correct / (q_idx + 1)
        logger.info(f"Progress: {q_idx + 1}/{len(test_questions)} questions, "
                   f"current pass@{batch_size} = {current_accuracy:.4f}")
        #print(str(step)+"/"+str(iteration))
#        del shuffled_dataset
#        del batch
#        del inputs
#        del outputs
#        del texts
    logger.info("Finished grading...")
    memory_stats()
    pass_at_k = total_correct / len(test_questions)
#    del model
#    del tokenizer
#    del dataset
#    del mapped
    with torch.no_grad():
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Cleaned up memory...")
    memory_stats()
    logger.info(f"pass@{batch_size}: {pass_at_k:.4f} "
               f"({total_correct}/{len(test_questions)} questions correct)")
    #logger.info(f"reward:{total_correct / (iteration*batch_size)}")
    print( pass_at_k)

if __name__ == "__main__":
    check_reward()