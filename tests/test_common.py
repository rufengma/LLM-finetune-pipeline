"""Tests for the chat-template and formatting helpers in src/common.py.

``src/common.py`` imports torch and numpy at module level, but the functions
covered here (build_prompt, format_full_text, default_target_modules) are pure
Python. Heavy deps are stubbed out with MagicMock so the suite runs on a
plain interpreter -- no torch/numpy required.

Run with:  pytest tests/ -q
"""

import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

# Stub heavy training deps before importing src.common. Only the pure helper
# functions are exercised here; tokenizer-dependent helpers (make_collate_fn,
# make_tokenize_fn) and set_seed intentionally stay out of scope because they
# need a real torch/tokenizer.
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("numpy", MagicMock())

from src.common import (  # noqa: E402
    PROMPT_TEMPLATE,
    build_prompt,
    default_target_modules,
    format_full_text,
    maybe_init_wandb,
)


class TestPromptTemplate:
    def test_template_renders_with_format(self):
        rendered = PROMPT_TEMPLATE.format(instruction="Say hi", response="Hello")
        assert "Say hi" in rendered
        assert "Hello" in rendered

    def test_template_uses_alpaca_style_sections(self):
        assert "### Instruction:" in PROMPT_TEMPLATE
        assert "### Response:" in PROMPT_TEMPLATE


class TestBuildPrompt:
    def test_instruction_only_exact_shape(self):
        assert build_prompt("Classify this text") == (
            "### Instruction:\nClassify this text\n\n### Response:\n"
        )

    def test_with_input_text(self):
        assert build_prompt("Summarize", "Long article here") == (
            "### Instruction:\nSummarize\n\n"
            "### Input:\nLong article here\n\n"
            "### Response:\n"
        )

    def test_whitespace_only_input_is_ignored(self):
        assert build_prompt("Answer", "   \n  ") == build_prompt("Answer", "")

    def test_leading_trailing_whitespace_is_stripped(self):
        prompt = build_prompt("  Trim me  ", "\n Pad me \n")
        assert "  Trim me" not in prompt
        assert "Pad me \n" not in prompt
        assert "### Instruction:\nTrim me\n" in prompt
        assert "### Input:\nPad me\n" in prompt

    def test_prompt_ends_with_response_marker(self):
        assert build_prompt("x").endswith("### Response:\n")


class TestFormatFullText:
    def test_full_text_is_prompt_plus_response_plus_eos(self):
        text = format_full_text("Translate", "bonjour", "hello", "</s>")
        assert text == (
            build_prompt("Translate", "bonjour") + "hello" + "</s>"
        )

    def test_response_whitespace_stripped_before_eos(self):
        text = format_full_text("Reply", "", "  yes  ", "</s>")
        assert text.endswith("yes</s>")
        assert "  yes" not in text

    def test_eos_token_appended_verbatim(self):
        assert format_full_text("i", "", "r", "<|endoftext|>").endswith("<|endoftext|>")

    def test_with_input_text_round_trip(self):
        text = format_full_text("QA", "What is AI?", "A field of CS.", "</s>")
        assert "### Input:\nWhat is AI?\n" in text
        assert text.endswith("A field of CS.</s>")


class TestDefaultTargetModules:
    def test_gpt2_family(self):
        assert default_target_modules("gpt2") == ["c_attn", "c_proj"]
        assert default_target_modules("gpt2-xl") == ["c_attn", "c_proj"]

    def test_llama_family_gets_attention_and_mlp(self):
        modules = default_target_modules("llama")
        assert modules == [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    def test_other_supported_families(self):
        for model_type in ("mistral-7b", "Qwen2-7B", "gemma-2-9b", "phi-3-mini"):
            assert default_target_modules(model_type) == [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ], model_type

    def test_matching_is_case_insensitive(self):
        assert default_target_modules("LLaMA-2-7B") == default_target_modules("llama")

    def test_unknown_model_falls_back_to_q_v_proj(self):
        assert default_target_modules("bert-base-uncased") == ["q_proj", "v_proj"]


class TestMaybeInitWandb:
    def _no_wandb(self, monkeypatch):
        """Remove wandb from sys.modules and make `import wandb` fail.

        sys.modules["wandb"] = None makes the import raise ImportError
        deterministically, whether or not wandb is installed in the env.
        """
        monkeypatch.setitem(sys.modules, "wandb", None)

    def test_no_project_returns_false_without_importing_wandb(self, monkeypatch):
        self._no_wandb(monkeypatch)
        args = Namespace(wandb_project=None, wandb_run_name=None, wandb_entity=None)
        assert maybe_init_wandb(args) is False
        # import was never attempted: the None sentinel stays untouched
        assert sys.modules["wandb"] is None

    def test_missing_project_attr_also_returns_false(self, monkeypatch):
        self._no_wandb(monkeypatch)
        assert maybe_init_wandb(Namespace()) is False

    def test_project_without_wandb_installed_exits_with_hint(self, monkeypatch):
        self._no_wandb(monkeypatch)
        args = Namespace(wandb_project="llm-finetune", wandb_run_name=None,
                         wandb_entity=None)
        with pytest.raises(SystemExit) as exc:
            maybe_init_wandb(args)
        assert "pip install wandb" in str(exc.value)

    def test_project_inits_run_and_returns_true(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        args = Namespace(wandb_project="llm-finetune", wandb_run_name="r1",
                         wandb_entity="team", model_id="gpt2", lr=1e-4)
        assert maybe_init_wandb(args) is True
        fake.init.assert_called_once()
        _, kwargs = fake.init.call_args
        assert kwargs["project"] == "llm-finetune"
        assert kwargs["name"] == "r1"
        assert kwargs["entity"] == "team"
        assert kwargs["config"]["model_id"] == "gpt2"
        assert kwargs["config"]["lr"] == 1e-4

    def test_defaults_become_none_when_run_name_entity_unset(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        args = Namespace(wandb_project="llm-finetune", wandb_run_name=None,
                         wandb_entity=None, model_id="gpt2", lr=2e-4)
        assert maybe_init_wandb(args) is True
        _, kwargs = fake.init.call_args
        assert kwargs["name"] is None
        assert kwargs["entity"] is None
