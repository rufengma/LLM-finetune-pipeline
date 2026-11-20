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

from src.train import build_eval_kwargs, parse_args, resolve_checkpoint  # noqa: E402


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


class TestPromptTemplateArgs:
    def test_defaults_to_alpaca(self, monkeypatch):
        args = _parse(monkeypatch, [])
        assert args.prompt_template == "alpaca"

    def test_explicit_chatml_is_accepted(self, monkeypatch):
        args = _parse(monkeypatch, ["--prompt-template", "chatml"])
        assert args.prompt_template == "chatml"

    def test_unknown_template_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, ["--prompt-template", "llama3"])

    def test_yaml_config_accepts_the_key(self, monkeypatch, tmp_path):
        cfg = tmp_path / "tmpl.yaml"
        cfg.write_text("prompt_template: chatml\n")
        args = _parse(monkeypatch, ["--config", str(cfg)])
        assert args.prompt_template == "chatml"

    def test_cli_flag_overrides_yaml(self, monkeypatch, tmp_path):
        cfg = tmp_path / "tmpl.yaml"
        cfg.write_text("prompt_template: chatml\n")
        args = _parse(monkeypatch, ["--config", str(cfg),
                                    "--prompt-template", "alpaca"])
        assert args.prompt_template == "alpaca"


def _fake_checkpoint(parent, name, with_state=True):
    d = parent / name
    d.mkdir(parents=True)
    if with_state:
        (d / "trainer_state.json").write_text('{"global_step": 10}')
    return d


class TestResolveCheckpoint:
    def test_none_means_fresh_run(self, tmp_path):
        assert resolve_checkpoint(None, tmp_path) is None

    def test_latest_picks_highest_step(self, tmp_path):
        _fake_checkpoint(tmp_path, "checkpoint-5")
        _fake_checkpoint(tmp_path, "checkpoint-10")
        _fake_checkpoint(tmp_path, "checkpoint-3")
        assert resolve_checkpoint("latest", tmp_path) == str(tmp_path / "checkpoint-10")

    def test_latest_ignores_dirs_without_trainer_state(self, tmp_path):
        _fake_checkpoint(tmp_path, "checkpoint-5")
        _fake_checkpoint(tmp_path, "checkpoint-99", with_state=False)
        assert resolve_checkpoint("latest", tmp_path) == str(tmp_path / "checkpoint-5")

    def test_latest_with_no_checkpoints_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            resolve_checkpoint("latest", tmp_path)

    def test_latest_with_only_invalid_dirs_errors(self, tmp_path):
        _fake_checkpoint(tmp_path, "checkpoint-7", with_state=False)
        with pytest.raises(SystemExit):
            resolve_checkpoint("latest", tmp_path)

    def test_explicit_existing_path_returned_as_given(self, tmp_path):
        ckpt = _fake_checkpoint(tmp_path, "checkpoint-10")
        assert resolve_checkpoint(str(ckpt), tmp_path) == str(ckpt)

    def test_bare_name_resolves_under_output_dir(self, tmp_path):
        _fake_checkpoint(tmp_path, "checkpoint-10")
        assert resolve_checkpoint("checkpoint-10", tmp_path) == str(tmp_path / "checkpoint-10")

    def test_missing_path_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            resolve_checkpoint("checkpoint-42", tmp_path)

    def test_dir_without_trainer_state_errors(self, tmp_path):
        _fake_checkpoint(tmp_path, "checkpoint-10", with_state=False)
        with pytest.raises(SystemExit):
            resolve_checkpoint(str(tmp_path / "checkpoint-10"), tmp_path)

    def test_plain_file_errors(self, tmp_path):
        f = tmp_path / "not-a-checkpoint"
        f.write_text("nope")
        with pytest.raises(SystemExit):
            resolve_checkpoint(str(f), tmp_path)


class TestResumeFromCheckpointArgs:
    def test_defaults_to_none(self, monkeypatch):
        args = _parse(monkeypatch, [])
        assert args.resume_from_checkpoint is None

    def test_explicit_path_is_parsed(self, monkeypatch):
        args = _parse(monkeypatch, ["--resume-from-checkpoint",
                                    "outputs/run/checkpoint-10"])
        assert args.resume_from_checkpoint == "outputs/run/checkpoint-10"

    def test_latest_keyword_is_parsed(self, monkeypatch):
        args = _parse(monkeypatch, ["--resume-from-checkpoint", "latest"])
        assert args.resume_from_checkpoint == "latest"

    def test_yaml_config_accepts_the_key(self, monkeypatch, tmp_path):
        cfg = tmp_path / "resume.yaml"
        cfg.write_text("resume_from_checkpoint: latest\n")
        args = _parse(monkeypatch, ["--config", str(cfg)])
        assert args.resume_from_checkpoint == "latest"

    def test_cli_flag_overrides_yaml(self, monkeypatch, tmp_path):
        cfg = tmp_path / "resume.yaml"
        cfg.write_text("resume_from_checkpoint: latest\n")
        args = _parse(monkeypatch, ["--config", str(cfg),
                                    "--resume-from-checkpoint",
                                    "outputs/run/checkpoint-20"])
        assert args.resume_from_checkpoint == "outputs/run/checkpoint-20"
