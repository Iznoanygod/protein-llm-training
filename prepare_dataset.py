import argparse
from datasets import load_dataset, DatasetDict
from pathlib import Path

os.environ["HF_TOKEN"] = ""
os.environ["HF_HOME"] = "/work/hdd/bdyk/apark4/huggingface"

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


def to_prompt_completion(example):
    return {
        "prompt": [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['problem']}
            
        ],
        "ground_truth": str(example["solution"]).strip(),
    }

def split_dataset(dataset, train_frac=0.8, val_frac=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    test_size = total_size - train_size - val_size

    train_dataset = dataset.select(range(0, train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

def main():
    parser = argparse.ArgumentParser(description="Prepare competition_math as prompt/completion with standard splits.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Loading dataset {args.dataset}...")

    raw = load_dataset(args.dataset)
    print("Mapping dataset...")
    mapped = raw.map(to_prompt_completion, remove_columns=raw['train'].column_names)
    print("Splitting dataset...")
    split = split_dataset(mapped['train'], train_frac=args.train_frac, val_frac=args.val_frac)

    output_dir.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(output_dir))
    print(f"Saved processed dataset with splits to: {output_dir.resolve()}")

