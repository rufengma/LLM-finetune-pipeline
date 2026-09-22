"""Interactive / single-prompt inference with the base model + LoRA adapter.

Usage:
    # Tuned model, single prompt
    python src/inference.py --model-id gpt2 --adapter-dir outputs/smoke-test \\
        --prompt "Explain what a REST API is."

    # Base model only, interactive chat
    python src/inference.py --model-id gpt2 --interactive
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_prompt, ensure_pad_token  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate with base / LoRA-tuned model")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter-dir", default=None,
                   help="Path to LoRA adapter; omit for base model only")
    p.add_argument("--prompt", default=None, help="Single instruction to answer")
    p.add_argument("--interactive", action="store_true", help="REPL mode")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def load(model_id: str, adapter_dir: str | None):
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(model_id,
                                                               trust_remote_code=True))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        print(f"Loaded LoRA adapter from {adapter_dir} (merged)")
    else:
        print("Loaded base model (no adapter)")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def answer(model, tokenizer, instruction: str, max_new_tokens: int,
           temperature: float, top_p: float) -> str:
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id,
        # only meaningful for sampling; passing them with do_sample=False
        # triggers a transformers warning
        **({"temperature": temperature, "top_p": top_p} if temperature > 0 else {}),
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


def main() -> None:
    args = parse_args()
    model, tokenizer = load(args.model_id, args.adapter_dir)

    if args.interactive or not args.prompt:
        print("Interactive mode — type 'quit' to exit.\n")
        while True:
            try:
                instruction = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if instruction.lower() in ("quit", "exit"):
                break
            if not instruction:
                continue
            print("\nAssistant:", answer(model, tokenizer, instruction,
                                        args.max_new_tokens, args.temperature,
                                        args.top_p), "\n")
    else:
        print(answer(model, tokenizer, args.prompt, args.max_new_tokens,
                     args.temperature, args.top_p))


if __name__ == "__main__":
    main()
