"""Build the instruction-tuning dataset from raw JSONL seed tasks.

Reads data/raw/seed_tasks.jsonl (one {"instruction", "input", "response"} object
per line), validates it, and writes a train / held-out split as HuggingFace
datasets to data/processed/.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --test-size 12 --seed 42
"""

import argparse
import hashlib
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
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}"
                )
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


def sha256_file(path: Path) -> str:
    """Hex SHA-256 digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(directory: Path) -> str:
    """Deterministic digest of every file under a directory.

    Files are visited in sorted relative-path order; each file's relative
    path and bytes feed the digest, so any rename, addition, removal, or
    content change alters the result.
    """
    h = hashlib.sha256()
    files = sorted(
        p for p in directory.rglob("*") if p.is_file()
    )
    for p in files:
        h.update(p.relative_to(directory).as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def write_manifest(
    out: Path,
    *,
    input_path: str,
    input_sha256: str,
    input_rows: int,
    seed: int,
    test_size: int,
    train_dir: Path,
    test_dir: Path,
    train_rows: int,
    test_rows: int,
) -> Path:
    """Write a provenance manifest describing the produced dataset.

    Records the input file's hash and row count plus the seed/test-size that
    determined the split, and one digest per saved split directory. Re-running
    ``prepare_data.py`` with the same input and seed must produce an identical
    manifest -- a quick way to spot silent data drift before training.
    """
    manifest = {
        "input": {
            "path": input_path,
            "sha256": input_sha256,
            "rows": input_rows,
        },
        "seed": seed,
        "test_size": test_size,
        "splits": {
            "train": {
                "path": train_dir.name,
                "rows": train_rows,
                "sha256": dir_digest(train_dir),
            },
            "test": {
                "path": test_dir.name,
                "rows": test_rows,
                "sha256": dir_digest(test_dir),
            },
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


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

    input_path = Path(args.input)
    manifest_path = write_manifest(
        out,
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        input_rows=len(tasks),
        seed=args.seed,
        test_size=args.test_size,
        train_dir=out / "train",
        test_dir=out / "test",
        train_rows=n_train,
        test_rows=n_test,
    )
    print(f"Wrote data manifest to {manifest_path}")


if __name__ == "__main__":
    main()
