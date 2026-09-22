"""Build the instruction-tuning dataset from raw JSONL seed tasks.

Reads data/raw/seed_tasks.jsonl (one {"instruction", "input", "response"} object
per line), validates it, and writes a train / held-out split as HuggingFace
datasets to data/processed/.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --test-size 12 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import set_seed  # noqa: E402


def load_seed_tasks(path: Path) -> list:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not obj.get("instruction") or not obj.get("response"):
                raise ValueError(f"{path}:{lineno}: 'instruction' and 'response' are required")
            tasks.append({
                "instruction": str(obj["instruction"]).strip(),
                "input": str(obj.get("input", "") or "").strip(),
                "response": str(obj["response"]).strip(),
            })
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare instruction-tuning dataset")
    parser.add_argument("--input", default="data/raw/seed_tasks.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--test-size", type=int, default=12,
                        help="Number of held-out examples (absolute count)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tasks = load_seed_tasks(Path(args.input))
    print(f"Loaded {len(tasks)} seed tasks from {args.input}")

    ds = Dataset.from_list(tasks)
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train_ds, test_ds = split["train"], split["test"]

    out = Path(args.output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(out / "train"))
    test_ds.save_to_disk(str(out / "test"))

    def stats(d):
        resp_lens = [len(r.split()) for r in d["response"]]
        return len(d), sum(resp_lens) / len(resp_lens)

    n_train, avg_train = stats(train_ds)
    n_test, avg_test = stats(test_ds)
    print(f"Train: {n_train} examples (avg response {avg_train:.1f} words)")
    print(f"Held-out test: {n_test} examples (avg response {avg_test:.1f} words)")
    print(f"Saved to {out}/train and {out}/test")


if __name__ == "__main__":
    main()
