"""Compare the base model against the LoRA-tuned model on the held-out set.

Metrics:
  1. Token-level loss / perplexity on held-out responses (response tokens only).
  2. Side-by-side generations for N prompts -> eval/results.md
  3. Optional LLM-as-judge win rate (--judge, needs JUDGE_API_KEY in .env).

Usage:
    python src/evaluate.py --model-id gpt2 --adapter-dir outputs/smoke-test
    python src/evaluate.py --model-id gpt2 --adapter-dir outputs/smoke-test --judge
"""

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_prompt, ensure_pad_token, make_tokenize_fn, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate base vs LoRA-tuned model")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter-dir", default="outputs/qwen2.5-1.5b-lora")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output-dir", default="eval")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--num-samples", type=int, default=8,
                   help="Prompts for side-by-side generation")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge", action="store_true",
                   help="Run LLM-as-judge pairwise comparison (needs .env)")
    return p.parse_args()


@torch.no_grad()
def heldout_loss(model, tokenizer, test_ds, max_len) -> tuple:
    """Mean NLL and perplexity over response tokens of the held-out set."""
    tokenize = make_tokenize_fn(tokenizer, max_len)
    total_nll, total_tokens = 0.0, 0
    model.eval()
    for ex in test_ds:
        tok = tokenize({"instruction": ex["instruction"],
                        "input": ex["input"], "response": ex["response"]})
        input_ids = torch.tensor([tok["input_ids"]])
        labels = torch.tensor([tok["labels"]])
        if next(model.parameters()).is_cuda:
            input_ids, labels = input_ids.cuda(), labels.cuda()
        out = model(input_ids=input_ids, labels=labels)
        n_tokens = (labels != -100).sum().item()
        if n_tokens > 0:
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens
    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(min(mean_nll, 20))


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


def judge_pairwise(client, judge_model: str, instruction: str,
                   resp_a: str, resp_b: str) -> str | None:
    """Ask the judge which response is better. Returns 'A', 'B', or None."""
    system = ("You are a fair evaluator. Compare two assistant responses to the "
              "same instruction and reply with ONLY the letter A or B for the "
              "better response (helpfulness, correctness, clarity).")
    user = (f"Instruction:\n{instruction}\n\nResponse A:\n{resp_a}\n\n"
            f"Response B:\n{resp_b}\n\nWhich is better? Reply with only A or B.")
    try:
        r = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=4, temperature=0,
        )
        verdict = r.choices[0].message.content.strip().upper()
        return verdict if verdict in ("A", "B") else None
    except Exception as e:  # noqa: BLE001 - judge is best-effort
        print(f"  [judge] call failed: {e}")
        return None


def run_judge(samples: list, judge_model: str) -> dict:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.environ.get("JUDGE_API_KEY")
    if not api_key:
        raise SystemExit("JUDGE_API_KEY not set. Copy .env.example to .env first.")
    client = OpenAI(api_key=api_key,
                    base_url=os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1"))

    wins = {"tuned": 0, "base": 0, "tie": 0}
    # Present each pair in both orders to mitigate position bias; a win only
    # counts if the judge is consistent across both orderings.
    for s in samples:
        v1 = judge_pairwise(client, judge_model, s["instruction"], s["tuned"], s["base"])
        v2 = judge_pairwise(client, judge_model, s["instruction"], s["base"], s["tuned"])
        if v1 == "A" and v2 == "B":
            wins["tuned"] += 1
        elif v1 == "B" and v2 == "A":
            wins["base"] += 1
        else:
            wins["tie"] += 1
    return wins


def md_escape(text: str, limit: int = 600) -> str:
    text = text.replace("|", "\\|").replace("\n", "<br>")
    return text[:limit] + ("…" if len(text) > limit else "")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(args.model_id,
                                                               trust_remote_code=True))
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            args.model_id, trust_remote_code=True, dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)

    # NOTE: the base and tuned models must be separate instances. Wrapping a
    # model with PeftModel mutates it in place, so evaluating "base" on the
    # same object after attaching the adapter would silently measure the
    # tuned model twice.
    base_model = load_base()
    tuned_model = PeftModel.from_pretrained(load_base(), args.adapter_dir)
    tuned_model = tuned_model.merge_and_unload()  # single model for fair timing
    base_model.eval()
    tuned_model.eval()
    print("Loaded base model and merged LoRA adapter for evaluation.")

    test_ds = load_from_disk(str(Path(args.data_dir) / "test"))
    print(f"Held-out set: {len(test_ds)} examples")

    # --- 1. Perplexity ---
    base_nll, base_ppl = heldout_loss(base_model, tokenizer, test_ds, args.max_seq_length)
    tuned_nll, tuned_ppl = heldout_loss(tuned_model, tokenizer, test_ds, args.max_seq_length)
    print(f"\nHeld-out NLL  | base: {base_nll:.4f} | tuned: {tuned_nll:.4f}")
    print(f"Held-out PPL  | base: {base_ppl:.2f} | tuned: {tuned_ppl:.2f}")

    # --- Write results.md incrementally (survives interruptions) ---
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    results_path = out_dir / "results.md"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# Evaluation Results",
            "",
            f"_Generated: {ts} · base model: `{args.model_id}` · adapter: `{args.adapter_dir}`_",
            "",
            "## Held-out perplexity (response tokens only)",
            "",
            "| Model | Mean NLL | Perplexity |",
            "|-------|----------|------------|",
            f"| Base | {base_nll:.4f} | {base_ppl:.2f} |",
            f"| LoRA-tuned | {tuned_nll:.4f} | {tuned_ppl:.2f} |",
            "",
            "## Sample generations (greedy decoding)",
            "",
        ]))

    # --- 2. Side-by-side generations ---
    samples = []
    for i, ex in enumerate(test_ds.select(range(min(args.num_samples, len(test_ds)))), 1):
        prompt = build_prompt(ex["instruction"], ex["input"])
        sample = {
            "instruction": ex["instruction"] + (f"\nInput: {ex['input']}" if ex["input"] else ""),
            "reference": ex["response"],
            "base": generate(base_model, tokenizer, prompt, args.max_new_tokens),
            "tuned": generate(tuned_model, tokenizer, prompt, args.max_new_tokens),
        }
        samples.append(sample)
        print(f"  generated {len(samples)}/{args.num_samples} samples")
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("\n".join([
                f"### Sample {i}",
                "",
                f"**Instruction:** {md_escape(sample['instruction'], 300)}",
                "",
                "| | |",
                "|---|---|",
                f"| **Base** | {md_escape(sample['base'])} |",
                f"| **LoRA-tuned** | {md_escape(sample['tuned'])} |",
                f"| **Reference** | {md_escape(sample['reference'])} |",
                "",
            ]))

    # --- 3. Optional LLM judge (appended at the end) ---
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    if args.judge:
        wins = run_judge(samples, judge_model)
        total = sum(wins.values())
        print(f"\nLLM-as-judge ({judge_model}): tuned wins {wins['tuned']}/{total}, "
              f"base wins {wins['base']}/{total}, ties {wins['tie']}/{total}")
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("\n".join([
                "",
                f"## LLM-as-judge pairwise win rate (`{judge_model}`)",
                "",
                f"- Tuned wins: **{wins['tuned']}/{total}**",
                f"- Base wins: {wins['base']}/{total}",
                f"- Ties / inconsistent: {wins['tie']}/{total}",
                "",
                "> Each pair was judged in both orders; a win counts only when the "
                "judge agreed in both orderings (mitigates position bias).",
                "",
            ]))

    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
