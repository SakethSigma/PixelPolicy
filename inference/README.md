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

## Metrics / evaluation — score checkpoints across all games

Beyond the demo, this package evaluates **trained checkpoints** behaviorally: play a fixed,
seeded held-out test set of every game with a served checkpoint and report per-game accuracy /
win-rate. It reuses the generic per-game wiring in `distillation/registry.py::GAMES`
(`make_agent` / `make_env(target)` / `sample_targets(n,"val",rng)` / `good_status`) and the episode
driver `agents/rollout.py::run_eval` — no per-game code.

```
metrics.py          pure scoring: solved/n, Wilson 95% CI, per-game + aggregate (single/multi/reasoning)
evaluate.py         play all games × N seeded val instances against a server → eval_results/<label>.json
run_checkpoints.py  orchestrate base + epoch checkpoints: launch server, wait ready, evaluate, teardown
analysis/viz_eval.py blog-quality plots from the result JSONs (PEP 723 inline deps; no vLLM/torch)
```

**Run it (local, one GPU, sequential).** One command evaluates the base model + every epoch
checkpoint of a variant (it launches/awaits/tears-down the vLLM server per checkpoint, and is
resumable — skips any `<out>/<label>.json` that already exists):

```bash
uv run --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-sft-wordle --epochs 1,2,3,4 --base \
  --games all --n 300 --seed 0 --out eval_results/
```

Or evaluate a single already-running server:

```bash
uv run --package inference python -m inference.server --model saketh-chervu/word-games-sft-wordle --revision epoch-3 &
uv run --package inference python -m inference.evaluate --label wordle-e3 --games all --n 300 --seed 0 --out eval_results/
```

**Eval set.** `--n` distinct held-out (`mode="val"`) instances per game, drawn with a fixed `--seed`
so every checkpoint sees the *identical* test set. Sampling is frozen for fairness
(`temperature 0.6, top_p 0.95, enable_thinking=True`). A "solved" episode is `final.status ==
good_status` (`"won"` for wordle/codebreaker/bullscows; `"correct"` for the single-turn games).
Checkpoints exist only at whole-epoch boundaries (training used `save_strategy="epoch"`), so the
x-axis is `{base, e1, e2, e3, e4}`.

**Raw predictions are the source of truth.** Every episode is persisted *incrementally* (appended +
flushed as it finishes) to `eval_results/raw/<label>/<game>.jsonl` — target, outcome, and each turn's
raw reply + parsed action. So a crash never loses completed episodes, and **any new metric is
recomputed offline with no re-inference** (`--no-store-raw` to opt out). To derive metrics from
stored raw:

```bash
# recompute metrics for every label from raw (after editing metrics.py, or just to rebuild the JSONs):
uv run --package inference python -m inference.recompute --raw eval_results/raw --out eval_results
# a single checkpoint:
uv run --package inference python -m inference.recompute --raw eval_results/raw/wordle-e4 --out eval_results
```
Adding a metric = edit `inference/metrics.py::game_metrics` (it sees each episode's turns: `action`,
`response`, `status`), then run `recompute` — the expensive generation is never repeated.

**Plots (blog).** Render all figures from the results — no GPU needed:

```bash
uv run --no-project inference/analysis/viz_eval.py --results eval_results --out eval_plots
```

Produces: a **game × checkpoint accuracy heatmap**, **per-game accuracy** small-multiples (with
Wilson CI bands), **aggregate** (macro / single-turn / multi-turn / reasoning) across epochs, the
**wordle headline** (win-rate ± CI + solved-by-round), **base→best delta per game** (the transfer
story), and **format discipline** (action-parse / `<think>` rate). PNG + SVG.

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
