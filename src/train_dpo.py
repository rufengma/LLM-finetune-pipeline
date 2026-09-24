"""DPO preference-tuning follow-up for the LoRA SFT checkpoint (trl + peft).

Direct Preference Optimization trains the model to prefer *chosen* responses
over *rejected* ones for the same prompt, relative to a frozen reference
policy. This is the standard RLHF-lite step that follows supervised
fine-tuning: SFT teaches the *format*, DPO teaches *which* responses are
better.

Usage:
    # 1. SFT first (train.py), then DPO on top of the SFT adapter:
    python src/train.py --model-id gpt2 --max-steps 10 --batch-size 2 \
        --max-seq-length 256 --output-dir outputs/sft-smoke
    python src/train_dpo.py --model-id gpt2 --adapter-dir outputs/sft-smoke \
        --build-from-sft --max-steps 5 --batch-size 1 --max-seq-length 256 \
        --output-dir outputs/dpo-smoke

    # 2. DPO directly from the base model (no SFT first):
    python src/train_dpo.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
        --build-from-sft --num-epochs 1 --batch-size 2 --grad-accum 4 \
        --beta 0.1 --output-dir outputs/dpo

    # 3. With your own preference pairs (JSONL with prompt/chosen/rejected):
    python src/train_dpo.py --dpo-data data/preferences.jsonl ...

Preference data note: --build-from-sft derives (prompt, chosen, rejected)
triples from the SFT train set using the true response as `chosen` and a
response to a *different* instruction as `rejected` (a "distractor"). This is
good enough for smoke tests and demos, but for real preference tuning use
genuinely ranked pairs (human or LLM-judge labels) — see data/README or your
own labeling pipeline.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (build_prompt, default_target_modules, ensure_pad_token,  # noqa: E402
                    set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO preference-tune a causal LM")
    # Model / data
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter-dir", default=None,
                   help="SFT LoRA adapter (from train.py) to continue DPO from; "
                        "merged into the base weights before applying a fresh DPO LoRA")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--dpo-data", default=None,
                   help="JSONL file with {prompt, chosen, rejected} per line; "
                        "overrides --build-from-sft")
    p.add_argument("--build-from-sft", action="store_true",
                   help="Derive preference triples from the SFT train set "
                        "(chosen=true response, rejected=distractor response)")
    p.add_argument("--save-preferences-to", default=None,
                   help="Optional path to save the derived preference pairs as JSONL")
    p.add_argument("--output-dir", default="outputs/dpo")
    p.add_argument("--max-seq-length", type=int, default=1024)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="*", default=None)
    # DPO
    p.add_argument("--beta", type=float, default=0.1,
                   help="KL penalty vs the reference policy (smaller = stronger preference push)")
    p.add_argument("--loss-type", default="sigmoid",
                   choices=["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo_hard", "nca_pair"],
                   help="DPO loss variant (trl DPOConfig.loss_type)")
    # Optimization
    p.add_argument("--lr", type=float, default=5e-5,
                   help="DPO usually needs a smaller LR than SFT LoRA")
    p.add_argument("--num-epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If > 0, overrides --num-epochs (useful for smoke tests)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # Hardware
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--load-in-8bit", action="store_true")
    return p.parse_args()


def load_preference_jsonl(path: Path) -> Dataset:
    """Load preference triples from a JSONL file with {prompt, chosen, rejected}."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("prompt", "chosen", "rejected"):
                if not obj.get(key):
                    raise ValueError(f"{path}:{lineno}: '{key}' is required")
            rows.append({"prompt": str(obj["prompt"]),
                         "chosen": str(obj["chosen"]),
                         "rejected": str(obj["rejected"])})
    if not rows:
        raise ValueError(f"No preference pairs found in {path}")
    return Dataset.from_list(rows)


def build_preferences_from_sft(train_ds, seed: int = 42) -> Dataset:
    """Derive (prompt, chosen, rejected) triples from the SFT train set.

    chosen = the example's own response; rejected = a response written for a
    *different* instruction (a deranged shuffle, so rejected never matches the
    prompt's topic). Demo-quality construction: fine for smoke tests, not a
    substitute for genuinely ranked preference data.
    """
    import random
    rng = random.Random(seed)
    n = len(train_ds)
    if n < 2:
        raise ValueError("Need at least 2 SFT examples to build preference pairs")

    # Derangement: no example is paired with its own response.
    idx = list(range(n))
    for _ in range(100):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            break
    else:  # fallback: rotate by one
        idx = [(i + 1) % n for i in range(n)]

    rows = []
    for i in range(n):
        ex, neg = train_ds[i], train_ds[idx[i]]
        rows.append({
            "prompt": build_prompt(ex["instruction"], ex["input"]),
            "chosen": ex["response"].strip(),
            "rejected": neg["response"].strip(),
        })
    return Dataset.from_list(rows)


def load_base_model(model_id: str, args) -> tuple:
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_kwargs = {"trust_remote_code": True, "dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = args.device_map
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Preference data ---
    if args.dpo_data:
        pref_ds = load_preference_jsonl(Path(args.dpo_data))
        print(f"Loaded {len(pref_ds)} preference pairs from {args.dpo_data}")
    elif args.build_from_sft:
        train_ds = load_from_disk(str(Path(args.data_dir) / "train"))
        pref_ds = build_preferences_from_sft(train_ds, seed=args.seed)
        print(f"Derived {len(pref_ds)} preference pairs from SFT train set")
        if args.save_preferences_to:
            out = Path(args.save_preferences_to)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for row in pref_ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Saved preference pairs to {out}")
    else:
        raise SystemExit("Provide --dpo-data <jsonl> or --build-from-sft")

    # --- Policy model (base, or SFT adapter merged in) ---
    model, tokenizer = load_base_model(args.model_id, args)
    if args.adapter_dir:
        print(f"Merging SFT adapter from {args.adapter_dir} into base weights")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model = model.merge_and_unload()

    # --- Reference model: frozen copy of the (possibly merged) policy start ---
    ref_model, _ = load_base_model(args.model_id, args)
    if args.adapter_dir:
        ref_model = PeftModel.from_pretrained(ref_model, args.adapter_dir)
        ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # --- DPO LoRA config (fresh adapter on top of the policy) ---
    target_modules = args.target_modules or default_target_modules(model.config.model_type)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    # --- Training ---
    from trl import DPOConfig, DPOTrainer

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(args.num_epochs * len(pref_ds)
                                / (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        weight_decay=0.01,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_seq_length,
        logging_steps=1 if args.max_steps > 0 else 10,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=max(1, args.max_steps) if args.max_steps > 0 else None,
        save_total_limit=2,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=pref_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    # --- Save adapter + tokenizer + run config ---
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    with open(out_dir / "dpo_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Saved DPO LoRA adapter, tokenizer, and config to {out_dir}")


if __name__ == "__main__":
    main()
