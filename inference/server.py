"""Thin launcher for a vLLM OpenAI-compatible inference server.

Hosts any HuggingFace chat/VLM model behind the OpenAI Chat Completions API
(``/v1/chat/completions``). This is intentionally a *thin wrapper* over vLLM's own
OpenAI server — we don't reimplement the API or the engine.

    # defaults to Qwen/Qwen3.5-0.8B on 127.0.0.1:8000
    uv run --package inference python -m inference.server

    # override model / port, or pass any extra vLLM flag through:
    uv run --package inference python -m inference.server --model Qwen/Qwen3-1.7B --port 8001
    uv run --package inference python -m inference.server --max-model-len 8192

Conservative defaults (``--max-model-len 8192``, ``--gpu-memory-utilization 0.85``,
``--limit-mm-per-prompt`` off) are added so the default launch fits a ~12 GB card; pass any
of them explicitly to override.

Agents connect by pointing ``OPENAI_BASE_URL`` at ``http://<host>:<port>/v1`` (see
``.env.example`` / ``agents/config.py``). Any model vLLM supports works without agent
changes — the agent only knows "an OpenAI-compatible URL".
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = os.environ.get("INFERENCE_MODEL", "Qwen/Qwen3.5-0.8B")
DEFAULT_HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
DEFAULT_PORT = os.environ.get("INFERENCE_PORT", "8000")

# Conservative defaults so the box-standard launch fits a 12 GB card. Each is applied only
# when the user didn't already pass it, so explicit flags always win.
#   --limit-mm-per-prompt  skips multimodal cache allocation for our text-only use of the VLM.
#   --attention-backend    TRITON_ATTN needs no CUDA toolkit (Triton ships its own compiler),
#                          unlike FlashInfer (needs nvcc) or FLASH_ATTN (not installed here).
#                          TORCH_SDPA is ViT-only in vLLM v1, so it can't serve the LM.
SAFE_DEFAULTS: dict[str, str] = {
    "--max-model-len": "8192",
    "--gpu-memory-utilization": "0.85",
    "--limit-mm-per-prompt": '{"image":0,"video":0}',
    "--attention-backend": "TRITON_ATTN",
}


def _with_defaults(passthrough: list[str]) -> list[str]:
    """Append each SAFE_DEFAULTS flag the user hasn't already supplied (in any form)."""
    extra: list[str] = []
    for flag, value in SAFE_DEFAULTS.items():
        if any(tok == flag or tok.startswith(flag + "=") for tok in passthrough):
            continue
        extra += [flag, value]
    return extra


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch a vLLM OpenAI-compatible server.",
        epilog="Unrecognized flags are passed straight through to `vllm serve`.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id to serve")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    args, passthrough = parser.parse_known_args(argv)

    # No CUDA toolkit (nvcc) on this box, so the FlashInfer *sampler* (which JIT-compiles)
    # can't start either — turn it off. The attention backend is chosen via the
    # `--attention-backend TRITON_ATTN` SAFE_DEFAULT above (a CLI flag in vLLM 0.22, not an
    # env var). `setdefault` lets you override the sampler choice via the environment.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # Force vLLM's v1 model runner. The v2 runner (default-on for plain CausalLM archs like
    # Qwen3ForCausalLM) allocates UVA buffers, which require pinned host memory — and WSL
    # forces `pin_memory=False`, so v2 dies at device init with `RuntimeError: UVA is not
    # available`. The v1 runner has no such dependency and serves the same models fine.
    # `setdefault` lets you re-enable v2 (`VLLM_USE_V2_MODEL_RUNNER=1`) on a non-WSL box.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    cmd = [
        "vllm", "serve", args.model,
        "--host", args.host,
        "--port", str(args.port),
        *passthrough,
        *_with_defaults(passthrough),
    ]
    print(f"[inference] launching: {' '.join(cmd)}", file=sys.stderr)
    # Replace this process with vLLM's server (it owns the lifecycle, signals, logs).
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
