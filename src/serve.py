"""HTTP inference server for the base model + LoRA adapter (FastAPI).

Wraps the same loading/generation path as ``src/inference.py`` behind two
endpoints so the tuned model can be called from any HTTP client:

    GET  /health    -- liveness check + which model/adapter is loaded
    POST /generate  -- {"instruction": "..."} -> {"response": "..."}

Usage:
    # Tuned model on port 8000 (default host 127.0.0.1)
    python src/serve.py --model-id gpt2 --adapter-dir outputs/smoke-test

    # Base model only, custom port, chatml template defaults
    python src/serve.py --model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --prompt-template chatml --port 8080

    curl -X POST localhost:8000/generate \\
        -H 'Content-Type: application/json' \\
        -d '{"instruction": "Explain what a REST API is.", "temperature": 0}'

Requires the optional server dependencies (see requirements.txt):
``pip install fastapi uvicorn``. The model is loaded once at startup, so
the first request is as fast as the CLI path; generation itself is
synchronous and runs in FastAPI's threadpool.
"""

import argparse
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import answer, load  # noqa: E402

TemplateName = Literal["alpaca", "chatml"]


class GenerateRequest(BaseModel):
    instruction: str = Field(
        min_length=1, description="Instruction to answer (blank is rejected)")

    @field_validator("instruction")
    @classmethod
    def _strip_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("instruction must not be blank")
        return value
    max_new_tokens: int | None = Field(
        default=None, ge=1, le=1024,
        description="Overrides the server default when set")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0,
        description="0 = greedy; overrides the server default when set")
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Overrides the server default when set")
    prompt_template: TemplateName | None = Field(
        default=None,
        description="Prompt format; defaults to the server's "
                    "--prompt-template when omitted. Must match the template "
                    "the adapter was trained with")


class GenerateResponse(BaseModel):
    response: str
    model: str
    adapter: str | None
    prompt_template: str


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve base / LoRA-tuned model over HTTP")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter-dir", default=None,
                   help="Path to LoRA adapter; omit for base model only")
    p.add_argument("--prompt-template", default="alpaca",
                   choices=["alpaca", "chatml"],
                   help="Default prompt format for requests that do not "
                        "specify one (must match the adapter's training "
                        "template; see training_config.json)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (127.0.0.1 keeps the server local)")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def create_app(model, tokenizer, *, model_id: str, adapter_dir: str | None = None,
               default_max_new_tokens: int = 256,
               default_temperature: float = 0.7, default_top_p: float = 0.9,
               default_prompt_template: str = "alpaca") -> FastAPI:
    """Build the FastAPI app around an already-loaded model + tokenizer.

    Split out from ``main()`` so tests can inject a stub model instead of
    loading real weights.
    """
    app = FastAPI(title="llm-finetune-pipeline inference server",
                  description="Generate with the base model + LoRA adapter "
                              "over HTTP (see src/serve.py).")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model=model_id, adapter=adapter_dir)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        template = req.prompt_template or default_prompt_template
        text = answer(
            model, tokenizer, req.instruction.strip(),
            req.max_new_tokens if req.max_new_tokens is not None
            else default_max_new_tokens,
            req.temperature if req.temperature is not None
            else default_temperature,
            req.top_p if req.top_p is not None else default_top_p,
            template=template)
        return GenerateResponse(response=text, model=model_id,
                                adapter=adapter_dir,
                                prompt_template=template)

    return app


def main() -> None:
    args = parse_args()
    model, tokenizer = load(args.model_id, args.adapter_dir)
    app = create_app(
        model, tokenizer, model_id=args.model_id, adapter_dir=args.adapter_dir,
        default_max_new_tokens=args.max_new_tokens,
        default_temperature=args.temperature, default_top_p=args.top_p,
        default_prompt_template=args.prompt_template)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "error: serving requires uvicorn (pip install uvicorn)")
    print(f"Serving {args.model_id} "
          f"({args.adapter_dir or 'base model only'}) "
          f"on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
