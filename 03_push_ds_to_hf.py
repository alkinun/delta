import json
import sys
from datasets import Dataset

def main(path="finetune.json", repo_id=None):
    if repo_id is None:
        print("Usage: python 05_push_to_hf.py <repo_id> [path]")
        print("Example: python 05_push_to_hf.py username/code-transform-dataset")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {path}")

    ds = Dataset.from_list(samples)
    ds.push_to_hub(repo_id)

    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else None
    path = sys.argv[2] if len(sys.argv) > 2 else "finetune.json"
    main(path, repo_id)
