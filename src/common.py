"""Shared helpers for the llm-finetune-pipeline project."""

import random

import numpy as np
import torch

# Alpaca-style template. Deliberately model-agnostic: it works with any causal
# LM, including models like GPT-2 that ship without a chat template.
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""


def build_prompt(instruction: str, input_text: str = "") -> str:
    """Render the prompt prefix (everything before the assistant's response)."""
    text = f"### Instruction:\n{instruction.strip()}"
    if input_text and input_text.strip():
        text += f"\n\n### Input:\n{input_text.strip()}"
    text += "\n\n### Response:\n"
    return text


def format_full_text(instruction: str, input_text: str, response: str, eos_token: str) -> str:
    """Render a complete training example: prompt + response + EOS."""
    return build_prompt(instruction, input_text) + response.strip() + eos_token


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_target_modules(model_type: str) -> list:
    """Sensible LoRA target modules per architecture.

    Rule of thumb: adapt the attention projections and (for LLaMA-style
    models) the MLP projections. Adapting more modules increases capacity
    at the cost of more trainable parameters.
    """
    model_type = model_type.lower()
    if "gpt2" in model_type:
        # GPT-2 uses Conv1D layers named c_attn / c_proj / c_fc
        return ["c_attn", "c_proj"]
    if any(k in model_type for k in ("llama", "mistral", "qwen", "gemma", "phi")):
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    # Fallback: attention projections, the safest generic choice
    return ["q_proj", "v_proj"]


def ensure_pad_token(tokenizer):
    """GPT-style tokenizers often lack a pad token; reuse EOS so batching works."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def make_collate_fn(tokenizer):
    """Batch collator that pads input_ids / attention_mask / labels.

    We intentionally do NOT use transformers' DataCollatorForLanguageModeling:
    in transformers>=5 it no longer pads a precomputed ``labels`` field and
    unconditionally overwrites it with ``input_ids`` when mlm=False, which
    destroys response-only loss masking. This collator pads labels with -100
    so padding never contributes to the loss.
    """
    pad_id = tokenizer.pad_token_id
    padding_side = tokenizer.padding_side or "right"

    def collate_fn(examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for e in examples:
            pad_len = max_len - len(e["input_ids"])
            if padding_side == "right":
                batch["input_ids"].append(e["input_ids"] + [pad_id] * pad_len)
                batch["attention_mask"].append(e["attention_mask"] + [0] * pad_len)
                batch["labels"].append(e["labels"] + [-100] * pad_len)
            else:
                batch["input_ids"].append([pad_id] * pad_len + e["input_ids"])
                batch["attention_mask"].append([0] * pad_len + e["attention_mask"])
                batch["labels"].append([-100] * pad_len + e["labels"])
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return collate_fn


def make_tokenize_fn(tokenizer, max_len):
    """Build a mapping function that tokenizes examples with response-only labels.

    Instruction tokens are masked with -100 so the loss is computed on the
    response only -- standard practice for instruction tuning.
    """

    def tokenize_fn(example):
        prompt = build_prompt(example["instruction"], example["input"])
        full_text = prompt + example["response"].strip() + tokenizer.eos_token

        # Tokenize prompt and full text consistently so the prompt is an exact
        # token prefix of the full sequence.
        prompt_ids = tokenizer(prompt, add_special_tokens=True,
                               truncation=True, max_length=max_len)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=True,
                         truncation=True, max_length=max_len)

        n_prompt = min(len(prompt_ids), len(full["input_ids"]))
        labels = [-100] * n_prompt + full["input_ids"][n_prompt:]
        full["labels"] = labels
        return full

    return tokenize_fn
