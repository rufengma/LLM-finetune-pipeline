"""LoRA fine-tuning for causal language models (peft + transformers Trainer).

Trains a LoRA adapter on the instruction dataset prepared by prepare_data.py.
Loss is computed on response tokens only (instruction tokens are masked),
which is standard practice for instruction tuning.

Quick smoke test (CPU, tiny model, ~10 steps):
    python src/train.py --model-id gpt2 --max-steps 10 --batch-size 2 \\
        --max-seq-length 256 --output-dir outputs/smoke-test

Recommended real run (single GPU, ~16GB VRAM):
    python src/train.py --model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --num-epochs 3 --batch-size 2 --grad-accum 8 --lr 2e-4 --bf16 \\
        --output-dir outputs/qwen2.5-1.5b-lora

QLoRA run (single GPU, fits in ~8GB VRAM via 4-bit NF4 quantization):
    python src/train.py --model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --load-in-4bit --bf16 --num-epochs 3 --batch-size 2 --grad-accum 8 \\
        --lr 2e-4 --output-dir outputs/qwen2.5-1.5b-qlora
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (default_target_modules, ensure_pad_token, make_collate_fn,  # noqa: E402
                    make_tokenize_fn, set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune a causal LM")
    # Model / data
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output-dir", default="outputs/qwen2.5-1.5b-lora")
    p.add_argument("--max-seq-length", type=int, default=1024)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="*", default=None,
                   help="Override auto-detected LoRA target modules")
    # Optimization
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If > 0, overrides --num-epochs (useful for smoke tests)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # Hardware
    p.add_argument("--bf16", action="store_true", help="Use bf16 (Ampere+ GPUs)")
    p.add_argument("--fp16", action="store_true", help="Use fp16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="8-bit quantization (GPU only, needs bitsandbytes)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit QLoRA quantization: NF4 + double quantization "
                        "(GPU only, needs bitsandbytes). Mutually exclusive "
                        "with --load-in-8bit.")
    p.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"],
                   help="4-bit quantization data type (default: nf4)")
    p.add_argument("--bnb-4bit-double-quant", action="store_true", default=True,
                   help="Use nested (double) quantization for 4-bit "
                        "(default: on; pass --no-bnb-4bit-double-quant to disable)")
    p.add_argument("--no-bnb-4bit-double-quant", action="store_false",
                   dest="bnb_4bit_double_quant",
                   help="Disable nested quantization for 4-bit")
    return p.parse_args()


def build_quantization_config(args):
    """Return (quantization_config, needs_kbit_training_prep) or (None, False).

    4-bit and 8-bit are mutually exclusive; bitsandbytes is imported lazily so
    the base requirements stay CPU-friendly without it installed.
    """
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("error: --load-in-4bit and --load-in-8bit are mutually "
                         "exclusive; pick one.")
    if not (args.load_in_4bit or args.load_in_8bit):
        return None, False
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise SystemExit(
            "error: quantization requested but bitsandbytes is not installed. "
            "Install it with: pip install bitsandbytes")
    if args.load_in_4bit:
        compute_dtype = (torch.bfloat16 if args.bf16
                         else torch.float16 if args.fp16 else torch.bfloat16)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    return quantization_config, True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer & model ---
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True))
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    quantization_config, prep_for_kbit = build_quantization_config(args)
    model_kwargs = {"trust_remote_code": True, "dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = args.device_map
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
    elif quantization_config is not None:
        # bitsandbytes k-bit kernels are CUDA-only; fall back to full precision
        # on CPU so smoke tests and CPU dev runs still work.
        print("WARNING: bitsandbytes quantization requires a CUDA GPU; "
              "running on CPU, ignoring the quantization flag and training "
              "in full precision.")
        quantization_config, prep_for_kbit = None, False
    # On CPU, device_map is left unset so Trainer handles placement.
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if prep_for_kbit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing

    # --- LoRA ---
    target_modules = args.target_modules or default_target_modules(model.config.model_type)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Data ---
    train_ds = load_from_disk(str(Path(args.data_dir) / "train"))
    train_ds = train_ds.map(make_tokenize_fn(tokenizer, args.max_seq_length),
                            remove_columns=train_ds.column_names)
    print(f"Tokenized {len(train_ds)} training examples "
          f"(seq length <= {args.max_seq_length})")

    # --- Training ---
    # transformers>=5 removed `warmup_ratio`; convert the ratio to steps here.
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(args.num_epochs * len(train_ds)
                                / (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    training_args = TrainingArguments(
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=make_collate_fn(tokenizer),
    )
    trainer.train()

    # --- Save adapter + tokenizer + run config ---
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Saved LoRA adapter, tokenizer, and config to {out_dir}")


if __name__ == "__main__":
    main()
