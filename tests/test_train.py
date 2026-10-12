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

from src.train import parse_args  # noqa: E402


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
