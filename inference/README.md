# Inference

A **thin launcher** over vLLM's built-in OpenAI-compatible server, so any HuggingFace
chat/VLM model is reachable at `/v1/chat/completions`. Agents don't know or care which
model is behind it — they only know "an OpenAI-compatible URL" (`OPENAI_BASE_URL`).

## Start the server

```bash
# defaults to Qwen/Qwen3.5-0.8B on 127.0.0.1:8000
uv run --package inference python -m inference.server

# override model / port; any extra flag is passed straight to `vllm serve`
uv run --package inference python -m inference.server --model Qwen/Qwen3-1.7B --port 8001
uv run --package inference python -m inference.server --max-model-len 8192

# a Wordle-fine-tuned model plays much better than the base model:
uv run --package inference python -m inference.server --model saketh-chervu/qwen3-06b-wordle-sft-phase1-best
```

**Overridable defaults.** To make the box-standard launch fit a ~12 GB card, the launcher adds
these unless you pass them yourself (an explicit flag always wins):

| Flag | Default | Why |
|------|---------|-----|
| `--max-model-len` | `8192` | cap context → smaller KV cache |
| `--gpu-memory-utilization` | `0.85` | leave headroom on a 12 GB card |
| `--limit-mm-per-prompt` | `{"image":0,"video":0}` | text-only use of a VLM → skip multimodal cache |
| `--attention-backend` | `TRITON_ATTN` | no-CUDA-toolkit-needed attention (see [Troubleshooting](#troubleshooting)) |

First run downloads the model from HuggingFace and loads it onto the GPU. **Startup is slow the
first time** (vLLM compiles CUDA graphs with `torch.compile` — a few minutes on WSL; the result is
cached, so later starts are fast). The server is ready once it logs
`Application startup complete` / `Uvicorn running on http://<host>:<port>`. Check it:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Point agents at it

In `.env` (see `.env.example`):

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
INFERENCE_MODEL=Qwen/Qwen3.5-0.8B
OPENAI_API_KEY=EMPTY            # vLLM accepts any non-empty key
```

Then, with the server running:

```bash
# watch the agent play one game on a colored board
uv run --package agents python -m agents.run --demo

# headless eval, prints a win-rate summary
uv run --package agents python -m agents.run --episodes 20
```

## Notes

- **Single GPU, single model.** `server.py` just `exec`s `vllm serve`; vLLM owns the
  engine, batching (continuous batching across concurrent requests), lifecycle, and logs.
- **VRAM.** A 0.8B model fits comfortably on a 12 GB card. For larger models pass
  `--gpu-memory-utilization`, `--max-model-len`, or `--dtype` straight through.
- **This server has no game/agent knowledge.** It is a pure model host; the game loop
  lives in `agents/` (see `agents/Readme.md`).

## Troubleshooting

**`Could not find nvcc` / `EngineCore failed to start` (FlashInfer).** vLLM's default attention
backend (FlashInfer) JIT-compiles CUDA kernels at startup, which needs the **CUDA toolkit**
(`nvcc`) — the *driver* alone isn't enough, and on a brand-new CUDA (e.g. cu13) there are often no
prebuilt FlashInfer/flash-attn wheels yet. That's why the launcher defaults to
**`--attention-backend TRITON_ATTN`** and `VLLM_USE_FLASHINFER_SAMPLER=0`: Triton ships with
PyTorch and brings its own compiler (`ptxas`), so it needs **no CUDA toolkit**. (Note `TORCH_SDPA`
is ViT-only in vLLM v1 and can't serve the language model.) If you later install the toolkit and
want FlashInfer's faster kernels, override with `--attention-backend FLASHINFER`.

**CUDA out of memory.** Lower `--gpu-memory-utilization` (e.g. `0.80`) or shorten context with
`--max-model-len 4096`; your explicit flag overrides the default.

**First start hangs for minutes.** Expected — `torch.compile` is building CUDA graphs (watch for
`Compiling a graph … takes N s`). The compiled graphs are cached under `~/.cache/vllm`, so
subsequent starts skip it.
