"""Tests for the CLI surface of src/train.py.

``src/train.py`` imports torch, datasets, peft and transformers at module
level, but the argument-parsing path under test here (`parse_args`) never
touches them. Heavy deps are stubbed with MagicMock so the suite runs on a
plain interpreter -- no torch/transformers required.

Run with:  pytest tests/ -q
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy training deps before importing src.train. Only parse_args is
# exercised here; main() intentionally stays out of scope because it needs a
# real model, dataset and GPU environment.
for _mod in ("torch", "numpy", "datasets", "peft", "transformers"):
    sys.modules.setdefault(_mod, MagicMock())

from src.train import build_eval_kwargs, parse_args  # noqa: E402


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return parse_args()


class TestNeftuneNoiseAlpha:
    def test_defaults_to_none(self, monkeypatch):
        args = _parse(monkeypatch, [])
        assert args.neftune_noise_alpha is None

    def test_explicit_value_is_parsed_as_float(self, monkeypatch):
        args = _parse(monkeypatch, ["--neftune-noise-alpha", "5.0"])
        assert args.neftune_noise_alpha == 5.0

    def test_cli_flag_maps_to_trainingarguments_kwarg_name(self, monkeypatch):
        # TrainingArguments takes `neftune_noise_alpha`; the dest name must
        # match so argparse output can be passed straight through.
        args = _parse(monkeypatch, ["--neftune-noise-alpha", "10"])
        assert "neftune_noise_alpha" in vars(args)

    def test_non_numeric_value_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, ["--neftune-noise-alpha", "nope"])

    def test_yaml_config_accepts_the_key(self, monkeypatch, tmp_path):
        cfg = tmp_path / "neft.yaml"
        cfg.write_text("neftune_noise_alpha: 5.0\n")
        args = _parse(monkeypatch, ["--config", str(cfg)])
        assert args.neftune_noise_alpha == 5.0


class TestEarlyStoppingArgs:
    def test_disabled_by_default(self, monkeypatch):
        args = _parse(monkeypatch, [])
        assert args.early_stopping_patience is None
        assert build_eval_kwargs(args) == {}

    def test_explicit_patience_is_parsed_as_int(self, monkeypatch):
        args = _parse(monkeypatch, ["--early-stopping-patience", "3"])
        assert args.early_stopping_patience == 3

    def test_non_numeric_value_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, ["--early-stopping-patience", "soon"])

    def test_eval_split_defaults_to_test(self, monkeypatch):
        args = _parse(monkeypatch, ["--early-stopping-patience", "2"])
        assert args.eval_split == "test"

    def test_yaml_config_accepts_all_keys(self, monkeypatch, tmp_path):
        cfg = tmp_path / "es.yaml"
        cfg.write_text("early_stopping_patience: 4\neval_split: val\neval_steps: 7\n")
        args = _parse(monkeypatch, ["--config", str(cfg)])
        assert args.early_stopping_patience == 4
        assert args.eval_split == "val"
        assert args.eval_steps == 7


class TestBuildEvalKwargs:
    def test_max_steps_run_evals_every_fifth_of_budget(self, monkeypatch):
        args = _parse(monkeypatch, ["--early-stopping-patience", "2",
                                    "--max-steps", "10"])
        kw = build_eval_kwargs(args)
        assert kw["eval_strategy"] == "steps"
        assert kw["eval_steps"] == 2  # 10 // 5
        assert kw["load_best_model_at_end"] is True
        assert kw["metric_for_best_model"] == "eval_loss"
        assert kw["greater_is_better"] is False

    def test_explicit_eval_steps_wins(self, monkeypatch):
        args = _parse(monkeypatch, ["--early-stopping-patience", "2",
                                    "--max-steps", "10", "--eval-steps", "3"])
        assert build_eval_kwargs(args)["eval_steps"] == 3

    def test_epoch_run_evals_per_epoch(self, monkeypatch):
        args = _parse(monkeypatch, ["--early-stopping-patience", "2"])
        kw = build_eval_kwargs(args)
        assert kw["eval_strategy"] == "epoch"
        assert kw["eval_steps"] is None  # TrainingArguments handles per-epoch
        assert kw["load_best_model_at_end"] is True

    def test_disabled_returns_empty(self, monkeypatch):
        args = _parse(monkeypatch, ["--max-steps", "10"])
        assert build_eval_kwargs(args) == {}

    def test_zero_patience_disables(self, monkeypatch):
        # patience < 1 is rejected in main(), so here it simply disables
        args = _parse(monkeypatch, ["--early-stopping-patience", "0"])
        assert build_eval_kwargs(args) == {}
