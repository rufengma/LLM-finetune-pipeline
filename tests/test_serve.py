"""Tests for the FastAPI inference server (src/serve.py).

The HTTP layer is tested with a stub model/tokenizer so no real weights
are loaded: the stub records what generation was asked to do and returns
a canned reply.

Note on stubbing: like the other test modules, heavy deps are stubbed
with MagicMock. The real ``src/inference.py`` decorates ``answer()``
with ``@torch.no_grad()``, which a mocked torch turns into a MagicMock,
so the tests inject a small fake ``inference`` module that re-implements
the same contract (render prompt -> generate -> decode -> strip prompt)
on top of the real ``common.build_prompt``. The real ``answer()`` path
is exercised by the live smoke run (tiny random model + curl) documented
in the README.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Stub heavy deps before importing src.serve (same convention as the
# other test modules).
for _mod in ("torch", "numpy", "datasets", "peft", "transformers"):
    sys.modules.setdefault(_mod, MagicMock())

from src.common import build_prompt  # noqa: E402

_fake_inference = types.ModuleType("inference")


def _fake_answer(model, tokenizer, instruction, max_new_tokens,
                 temperature, top_p, template="alpaca"):
    prompt = build_prompt(instruction, template=template)
    inputs = tokenizer(prompt, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id,
        **({"temperature": temperature, "top_p": top_p} if temperature > 0 else {}),
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


_fake_inference.answer = _fake_answer
_fake_inference.load = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("load() must not be called in HTTP tests"))
sys.modules["inference"] = _fake_inference

from src.serve import create_app, parse_args  # noqa: E402


class StubTokenizer:
    """Echoes the rendered prompt back plus a canned reply, so the
    ``answer()`` slicing (``decoded[len(prompt):]``) yields the reply."""

    def __init__(self):
        self.pad_token_id = 0
        self.last_prompt = ""

    def __call__(self, text, return_tensors=None):
        self.last_prompt = text
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def decode(self, ids, skip_special_tokens=True):
        return self.last_prompt + "stub reply"


class StubModel:
    """Records generate kwargs; ``is_cuda=False`` keeps tensors on CPU."""

    def __init__(self):
        self.generate_kwargs = None

    def parameters(self):
        return iter([SimpleNamespace(is_cuda=False)])

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]

    def eval(self):
        return self


def _client(**app_kwargs):
    model = StubModel()
    tokenizer = StubTokenizer()
    app = create_app(model, tokenizer, model_id="stub-model",
                     adapter_dir="outputs/smoke-test", **app_kwargs)
    return TestClient(app), model, tokenizer


def test_health_reports_loaded_model():
    client, _, _ = _client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "stub-model",
                        "adapter": "outputs/smoke-test"}


def test_generate_returns_stub_reply_with_server_defaults():
    client, model, _ = _client()
    r = client.post("/generate", json={"instruction": "Hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "stub reply"
    assert body["model"] == "stub-model"
    assert body["adapter"] == "outputs/smoke-test"
    assert body["prompt_template"] == "alpaca"
    # Server defaults (256 / 0.7 / 0.9) reach generate() when unset
    assert model.generate_kwargs["max_new_tokens"] == 256
    assert model.generate_kwargs["temperature"] == 0.7
    assert model.generate_kwargs["top_p"] == 0.9
    assert model.generate_kwargs["do_sample"] is True


def test_generate_per_request_overrides_win():
    client, model, _ = _client()
    r = client.post("/generate", json={
        "instruction": "Hi", "max_new_tokens": 32,
        "temperature": 0, "top_p": 0.5, "prompt_template": "chatml"})
    assert r.status_code == 200
    assert r.json()["prompt_template"] == "chatml"
    assert model.generate_kwargs["max_new_tokens"] == 32
    # temperature 0 -> greedy, so no sampling kwargs are passed through
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs


def test_generate_rejects_blank_instruction():
    client, _, _ = _client()
    assert client.post("/generate", json={"instruction": ""}).status_code == 422
    assert client.post("/generate", json={"instruction": "   "}).status_code == 422


def test_generate_rejects_out_of_range_sampling_params():
    client, _, _ = _client()
    assert client.post(
        "/generate", json={"instruction": "Hi", "temperature": 2.5}
    ).status_code == 422
    assert client.post(
        "/generate", json={"instruction": "Hi", "top_p": 1.5}
    ).status_code == 422
    assert client.post(
        "/generate", json={"instruction": "Hi", "max_new_tokens": 0}
    ).status_code == 422
    assert client.post(
        "/generate", json={"instruction": "Hi", "max_new_tokens": 2048}
    ).status_code == 422


def test_generate_rejects_unknown_template():
    client, _, _ = _client()
    assert client.post(
        "/generate", json={"instruction": "Hi", "prompt_template": "llama"}
    ).status_code == 422


def test_server_default_template_is_used_when_request_omits_it():
    client, model, tokenizer = _client(default_prompt_template="chatml")
    r = client.post("/generate", json={"instruction": "Hi"})
    assert r.status_code == 200
    assert r.json()["prompt_template"] == "chatml"
    assert tokenizer.last_prompt.startswith("<|im_start|>user")


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["serve.py"])
    args = parse_args()
    assert args.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.adapter_dir is None
    assert args.prompt_template == "alpaca"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.max_new_tokens == 256
