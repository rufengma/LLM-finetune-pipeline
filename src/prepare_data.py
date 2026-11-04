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


def _normalize(text: str) -> str:
    """Collapse whitespace and casefold, for dedup comparison only."""
    return " ".join(str(text).split()).casefold()


def _word_count(text: str) -> int:
    return len(str(text).split())


def filter_tasks(tasks: list, *, dedup: bool = True,
                 min_instruction_words: int = 3,
                 min_response_words: int = 3,
                 max_response_words: int = 1024) -> tuple:
    """Drop duplicates and low-quality examples, keeping original order.

    Dedup compares the normalized (whitespace-collapsed, casefolded)
    instruction+input+response triple and keeps the first occurrence, so a
    reworded-casing copy of the same example does not double-weight it in
    training. Length filters drop instructions that are too short to be
    meaningful and responses that are too short to teach anything or so
    long they are likely pasted documents. Quality filters drop responses
    that merely echo the instruction or degenerate into one repeated word.

    Returns ``(kept_tasks, report)`` where ``report`` counts how many rows
    each reason removed, so ``main()`` can print (and the manifest can
    record) exactly what the filters did.
    """
    if min_instruction_words < 0 or min_response_words < 0 or max_response_words < 0:
        raise ValueError("filter word limits must be non-negative")
    if max_response_words < min_response_words:
        raise ValueError("max_response_words must be >= min_response_words")
    report = {
        "input_rows": len(tasks),
        "duplicate": 0,
        "instruction_too_short": 0,
        "response_too_short": 0,
        "response_too_long": 0,
        "response_echoes_instruction": 0,
        "response_degenerate": 0,
        "kept_rows": 0,
    }
    seen = set()
    kept = []
    for task in tasks:
        key = (
            _normalize(task["instruction"]),
            _normalize(task["input"]),
            _normalize(task["response"]),
        )
        if dedup and key in seen:
            report["duplicate"] += 1
            continue
        seen.add(key)
        if _word_count(task["instruction"]) < min_instruction_words:
            report["instruction_too_short"] += 1
            continue
        n_resp = _word_count(task["response"])
        if n_resp < min_response_words:
            report["response_too_short"] += 1
            continue
        if n_resp > max_response_words:
            report["response_too_long"] += 1
            continue
        if _normalize(task["response"]) == _normalize(task["instruction"]):
            report["response_echoes_instruction"] += 1
            continue
        resp_words = task["response"].split()
        if len(resp_words) > 1 and len(set(w.casefold() for w in resp_words)) == 1:
            report["response_degenerate"] += 1
            continue
        kept.append(task)
    report["kept_rows"] = len(kept)
    return kept, report


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
    filters: dict | None = None,
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
    if filters is not None:
        manifest["filters"] = filters
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
    parser.add_argument("--min-instruction-words", type=int, default=3,
                        help="Drop examples whose instruction is shorter than this (words)")
    parser.add_argument("--min-response-words", type=int, default=3,
                        help="Drop examples whose response is shorter than this (words)")
    parser.add_argument("--max-response-words", type=int, default=1024,
                        help="Drop examples whose response is longer than this (words)")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Disable dedup of identical instruction+input+response rows")
    args = parser.parse_args()

    set_seed(args.seed)

    tasks = load_seed_tasks(Path(args.input))
    print(f"Loaded {len(tasks)} seed tasks from {args.input}")

    tasks, filter_report = filter_tasks(
        tasks,
        dedup=not args.keep_duplicates,
        min_instruction_words=args.min_instruction_words,
        min_response_words=args.min_response_words,
        max_response_words=args.max_response_words,
    )
    dropped = {k: v for k, v in filter_report.items()
               if k not in ("input_rows", "kept_rows") and v}
    print(f"Filters kept {filter_report['kept_rows']} of "
          f"{filter_report['input_rows']} rows"
          + (f"; dropped: {dropped}" if dropped else "; nothing dropped"))

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
        input_rows=filter_report["input_rows"],
        seed=args.seed,
        test_size=args.test_size,
        train_dir=out / "train",
        test_dir=out / "test",
        train_rows=n_train,
        test_rows=n_test,
        filters=filter_report,
    )
    print(f"Wrote data manifest to {manifest_path}")


if __name__ == "__main__":
    main()
