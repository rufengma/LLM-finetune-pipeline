"""Tests for the offline-testable helpers in src/evaluate.py.

``src/evaluate.py`` imports torch, datasets, peft and transformers at module
level, but the helpers covered here (``md_escape``,
``build_results_header``, ``judge_pairwise``, ``run_judge``) never touch a
model. Heavy deps are stubbed with MagicMock so the suite runs on a plain
interpreter -- no torch/GPU required, and the judge path is exercised with a
scripted fake OpenAI client instead of a real API.

``heldout_loss`` / ``generate`` / ``main`` intentionally stay out of scope:
they need a real model, tokenizer and held-out dataset.

Run with:  pytest tests/ -q
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub heavy eval deps before importing src.evaluate. Only the pure helpers
# are exercised here.
for _mod in ("torch", "numpy", "datasets", "peft", "transformers"):
    sys.modules.setdefault(_mod, MagicMock())

from src.evaluate import (build_gallery_table, build_results_header,
                          gallery_cell, judge_pairwise, md_escape,
                          most_repeated_sentence, run_judge)  # noqa: E402


# --- Fake OpenAI client ---------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Returns scripted verdicts in order; records every call."""

    def __init__(self, verdicts, fail=False):
        self._verdicts = list(verdicts)
        self.fail = fail
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("API down")
        return _FakeResponse(self._verdicts.pop(0))


class _FakeClient:
    def __init__(self, verdicts, fail=False):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(verdicts, fail))


def _make_sample(instruction="Do X", tuned="tuned text", base="base text"):
    return {"instruction": instruction, "tuned": tuned, "base": base}


# --- md_escape ------------------------------------------------------------

class TestMdEscape:
    def test_plain_text_unchanged(self):
        assert md_escape("hello world") == "hello world"

    def test_escapes_pipe_for_markdown_tables(self):
        assert md_escape("a | b") == "a \\| b"

    def test_newline_becomes_br(self):
        assert md_escape("line1\nline2") == "line1<br>line2"

    def test_empty_string(self):
        assert md_escape("") == ""

    def test_truncates_at_default_limit_with_ellipsis(self):
        text = "x" * 700
        out = md_escape(text)
        assert out == "x" * 600 + "…"
        assert len(out) == 601

    def test_exactly_at_limit_has_no_ellipsis(self):
        assert md_escape("y" * 600) == "y" * 600

    def test_custom_limit(self):
        assert md_escape("abcdefghij", limit=4) == "abcd…"

    def test_limit_counts_escaped_text(self):
        # A pipe counts as two chars after escaping, so truncation can cut
        # inside the escaped form -- the function truncates, not re-escapes.
        assert md_escape("a|b", limit=2) == "a\\…"


# --- gallery helpers ------------------------------------------------------

LOOP = ("I am sorry for the inconvenience. " * 8).strip()


class TestMostRepeatedSentence:
    def test_detects_a_sentence_loop(self):
        assert most_repeated_sentence(LOOP) == (
            "I am sorry for the inconvenience.", 8)

    def test_below_threshold_returns_none(self):
        text = "Same sentence here. Same sentence here. Something else."
        assert most_repeated_sentence(text) is None

    def test_exactly_at_threshold_counts(self):
        text = "Echo. Echo. Echo."
        assert most_repeated_sentence(text) == ("Echo.", 3)

    def test_matching_is_case_insensitive_and_whitespace_collapsed(self):
        text = "Hello  world. HELLO WORLD. hello world."
        assert most_repeated_sentence(text) == ("Hello world.", 3)

    def test_newline_separated_repeats_count(self):
        text = "First line.\nSecond line.\nSecond line.\nSecond line."
        assert most_repeated_sentence(text) == ("Second line.", 3)

    def test_picks_the_most_repeated_sentence(self):
        text = ("Alpha. Beta. Alpha. Beta. Alpha. Gamma. Gamma. Gamma. Gamma.")
        assert most_repeated_sentence(text) == ("Gamma.", 4)

    def test_empty_and_single_sentence_return_none(self):
        assert most_repeated_sentence("") is None
        assert most_repeated_sentence("Just one sentence.") is None

    def test_fragments_without_terminal_punctuation_still_split_on_newlines(self):
        text = "### Response:\n### Response:\n### Response:\nDone."
        assert most_repeated_sentence(text) == ("### Response:", 3)


class TestGalleryCell:
    def test_plain_output_is_just_word_count(self):
        assert gallery_cell("One two three four.") == "4 words"

    def test_looping_output_gets_repetition_flag(self):
        assert gallery_cell(LOOP) == (
            '48 words · repeats “I am sorry for the inconvenience.” ×8')

    def test_long_repeated_sentence_is_truncated(self):
        text = ("The gradient descent algorithm is a simple, but powerful, "
                "way to learn. ") * 3
        out = gallery_cell(text)
        assert out.startswith("36 words · repeats “The gradient descent algorithm is a s…")
        assert out.endswith("×3")

    def test_empty_output(self):
        assert gallery_cell("") == "0 words"


class TestBuildGalleryTable:
    def _samples(self):
        return [
            _make_sample(instruction="Write a short apology email",
                         base=LOOP, tuned="Dear Ana, sorry about the delay."),
            _make_sample(instruction="What is gradient descent?",
                         base="The following is a response.",
                         tuned="Gradient descent steps downhill."),
        ]

    def test_has_intro_and_header_rows(self):
        out = build_gallery_table(self._samples())
        assert out.startswith("> Gallery: one row per sample")
        assert "| Sample | Instruction | Base | LoRA-tuned |" in out

    def test_one_row_per_sample_with_anchor_links(self):
        out = build_gallery_table(self._samples())
        assert "| [1](#sample-1) | Write a short apology email |" in out
        assert "| [2](#sample-2) | What is gradient descent? |" in out
        assert "48 words · repeats" in out
        assert "6 words" in out  # tuned sample 1: "Dear Ana, sorry about the delay."
        assert "5 words" in out  # base sample 2: "The following is a response."

    def test_long_instruction_is_truncated(self):
        sample = _make_sample(instruction="x" * 100, base="b", tuned="t")
        out = build_gallery_table([sample])
        assert "x" * 60 + "…" in out

    def test_empty_samples_is_just_intro_and_header(self):
        out = build_gallery_table([])
        assert "| Sample | Instruction | Base | LoRA-tuned |" in out
        assert "[1](#sample-1)" not in out


# --- build_results_header -------------------------------------------------

def _header():
    return build_results_header("2026-11-12 15:00 UTC", "gpt2",
                                "outputs/smoke-test",
                                3.6752, 39.46, 3.4998, 33.11)


class TestBuildResultsHeader:
    def test_names_model_adapter_and_timestamp(self):
        out = _header()
        assert out.startswith("# Evaluation Results\n")
        assert "2026-11-12 15:00 UTC" in out
        assert "`gpt2`" in out
        assert "`outputs/smoke-test`" in out

    def test_perplexity_table_values(self):
        out = _header()
        assert "| Base | 3.6752 | 39.46 |" in out
        assert "| LoRA-tuned | 3.4998 | 33.11 |" in out
        assert "## Held-out perplexity (response tokens only)" in out

    def test_reading_note_explains_snapshot_and_rouge(self):
        out = _header()
        assert "single eval run" in out
        assert "--rouge" in out
        assert "greedy decoding" in out

    def test_ends_with_sample_generations_section(self):
        assert _header().endswith("## Sample generations (greedy decoding)\n")


# --- judge_pairwise -------------------------------------------------------

class TestJudgePairwise:
    def test_returns_a_verdict(self):
        client = _FakeClient(["A"])
        assert judge_pairwise(client, "gpt-4o-mini", "Do X", "resp A", "resp B") == "A"

    def test_lowercase_verdict_is_uppercased(self):
        client = _FakeClient(["b"])
        assert judge_pairwise(client, "gpt-4o-mini", "Do X", "resp A", "resp B") == "B"

    def test_verdict_with_surrounding_whitespace(self):
        client = _FakeClient(["  A\n"])
        assert judge_pairwise(client, "gpt-4o-mini", "Do X", "resp A", "resp B") == "A"

    def test_non_ab_verdict_returns_none(self):
        for verdict in ("tie", "C", "", "A or B", "Both are good"):
            client = _FakeClient([verdict])
            assert judge_pairwise(client, "gpt-4o-mini", "Do X", "a", "b") is None

    def test_api_failure_returns_none(self):
        client = _FakeClient([], fail=True)
        assert judge_pairwise(client, "gpt-4o-mini", "Do X", "a", "b") is None

    def test_request_carries_instruction_and_both_responses(self):
        client = _FakeClient(["A"])
        judge_pairwise(client, "judge-model", "Summarize this", "tuned", "base")
        call = client.chat.completions.calls[0]
        assert call["model"] == "judge-model"
        assert call["temperature"] == 0
        user_text = call["messages"][1]["content"]
        assert "Summarize this" in user_text
        assert "tuned" in user_text
        assert "base" in user_text
        assert call["messages"][0]["role"] == "system"


# --- run_judge ------------------------------------------------------------

def _install_fake_openai(monkeypatch, verdicts):
    """Point `openai` / `dotenv` at fakes; run_judge imports both lazily."""
    client = _FakeClient(verdicts)

    # run_judge calls OpenAI(...) directly and uses the return value as the
    # client, so the factory must return the client instance.
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda *a, **k: client
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    return client


class TestRunJudge:
    def test_counts_tuned_base_and_tie_wins(self, monkeypatch):
        # Sample 1: judge picks A (tuned) then B (tuned in 2nd order) -> tuned
        # Sample 2: B then A -> base
        # Sample 3: A then A (inconsistent) -> tie
        _install_fake_openai(monkeypatch, ["A", "B", "B", "A", "A", "A"])
        monkeypatch.setenv("JUDGE_API_KEY", "test-key")
        samples = [_make_sample(f"Q{i}") for i in range(3)]
        assert run_judge(samples, "gpt-4o-mini") == {"tuned": 1, "base": 1, "tie": 1}

    def test_judge_failure_counts_as_tie(self, monkeypatch):
        # judge_pairwise returns None on API failure; None/None -> tie.
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = lambda *a, **k: _FakeClient([], fail=True)
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
        monkeypatch.setenv("JUDGE_API_KEY", "test-key")
        assert run_judge([_make_sample()], "gpt-4o-mini") == {
            "tuned": 0, "base": 0, "tie": 1,
        }

    def test_missing_api_key_exits(self, monkeypatch):
        _install_fake_openai(monkeypatch, [])
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="JUDGE_API_KEY not set"):
            run_judge([_make_sample()], "gpt-4o-mini")

    def test_empty_samples_returns_zero_counts(self, monkeypatch):
        _install_fake_openai(monkeypatch, [])
        monkeypatch.setenv("JUDGE_API_KEY", "test-key")
        assert run_judge([], "gpt-4o-mini") == {"tuned": 0, "base": 0, "tie": 0}
