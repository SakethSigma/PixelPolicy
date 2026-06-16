# RunPod metrics / eval commands (temp — delete when done)

Run **checkpoint evaluation** (play all games, score win-rate/accuracy) on a remote GPU, push the
results to HF, and analyze on local. Separate from training (see `_RUNPOD_COMMANDS.md`).

**Why remote:** a 0.8B model barely uses an A100 80 GB, so you can run `--concurrency 32` (vs the
12 GB-local `4`) → ~8× the in-flight episodes → ~an hour instead of many. Split work across machines
(e.g. e1/e2 local, **e3/e4 remote**); each checkpoint is an independent `<label>.json` + `raw/<label>/`,
so you merge the dirs locally.

---

## 1. One-time setup on the eval pod

Fresh pods are bare — install the basics first (**git is usually missing**):

```bash
apt-get update && apt-get install -y git tmux curl procps      # procps = `watch`; tmux = survive disconnects
curl -LsSf https://astral.sh/uv/install.sh | sh ; source $HOME/.local/bin/env
git clone https://github.com/SakethSigma/PixelPolicy.git && cd PixelPolicy
git log --oneline -1                                           # confirm latest pushed code (>= 7602785, the crash-safe fix)
export HF_TOKEN=hf_xxxxxxxx                                    # needed to DOWNLOAD the (private) checkpoints
```

Then get vLLM in place. **Prefer 1A** (a template that already has a working vLLM) — it sidesteps the
whole CUDA-version dance.

### 1A. RECOMMENDED — start from a vLLM template, keep its vLLM, add ONLY our deps

Qwen3.5 needs vLLM **>= 0.17**, and the latest vLLM ships CUDA-13 wheels that die on a 12.8 driver
(`libcudart.so.13`). Easiest fix: pick a RunPod template that ALREADY ships a working vLLM, then
install OUR pure-Python deps **without touching its vllm/torch**. The repo's packages are
`package = false` (imported via the workspace PYTHONPATH, not built), so you only need their
third-party deps (pydantic, httpx, openai, huggingface-hub, datasets, …).

```bash
which vllm; python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
# if the template's vLLM lives in a venv, ACTIVATE it first:  source /path/to/that/venv/bin/activate

uv sync --package inference --active --inexact \
  --no-install-package vllm --no-install-package torch         # install OUR deps INTO the template env; keep its vllm/torch
uv run --no-sync --active --package inference vllm --version   # template vllm, unchanged (no libcudart error)
```
**With path 1A, add `--active` to EVERY `uv run` below** (so it uses the template env, not a fresh
`.venv`). Fallback if `--active` doesn't catch the env (conda/system Python, not a venv):
`uv pip install --python "$(which python3)" "pydantic>=2" python-dotenv "httpx>=0.27" "openai>=1" "huggingface-hub>=0.20" "datasets>=2" "anthropic>=0.40"`.

### 1B. FALLBACK — plain pod (no usable vLLM), install a cu128 vLLM yourself

```bash
uv sync --package inference                                     # eval harness deps (agents + distillation + vllm)
# vLLM has a compiled CUDA extension, so vLLM AND torch must be the SAME CUDA (12.8). Hit a WINDOW:
#   1. Qwen3.5 (Qwen3_5ForConditionalGeneration) needs vLLM >= 0.17.0 (older errors "not supported").
#   2. vLLM >= ~0.21 defaults to a CUDA-13 wheel → "libcudart.so.13" on a 12.8 driver. And
#      --torch-backend=cu128 only fixes TORCH's index, NOT vllm's compiled _C (vllm#43435).
# => Target vLLM 0.17–0.20. 0.19.0 is CONFIRMED working (cu128, Qwen3.5 loads):
uv pip install --reinstall "vllm==0.19.0" --torch-backend=cu128
#   if it fails the `vllm --version` test below, walk DOWN: 0.19.0 → 0.18.0 → 0.17.0 (0.20+ may pull cu13)
# Still cu13? install the explicit cu128 RELEASE wheel (some URLs 404 → next minor, vllm#37847):
#   uv pip install https://github.com/vllm-project/vllm/releases/download/v0.18.0/vllm-0.18.0+cu128-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128

# REAL test — loads the compiled _C extension (NOT just `import vllm`); must print a version, no libcudart:
uv run --no-sync --package inference vllm --version
```

## 2. Run the eval — ONE command, fully hands-off (all epochs + base, all games, auto-push)

**ALWAYS launch inside tmux** (so a dropped SSH connection can't SIGINT the run):

```bash
tmux new -s eval        # or reattach: tmux attach -t eval

# TRAINED checkpoints FIRST (base is the slow one — defer it, see §2.1):
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-sft-wordle --epochs 1,2,3,4 \
  --games all --n 300 --seed 0 --concurrency 300 --max-num-seqs 300 \
  --out /workspace/eval_results_v2/ \
  --push-results-repo saketh-chervu/word-games-eval --push-results-revision main
```
Detach (leave it running): **`Ctrl-b` then `d`**.

- Evaluates **epoch-1..4** on all 13 games (300 each), writes metrics + raw to the persistent
  volume, and **auto-uploads the whole eval dir to `saketh-chervu/word-games-eval` after each
  checkpoint** (a *dedicated* repo so model weights aren't mixed in). No manual upload.
- **Run BASE LAST, not first.** Base never solves → every episode burns all 6 turns (slowest run);
  don't block the trained results behind it. `_checkpoints()` always puts `--base` FIRST, so DON'T
  pass `--base` here — run it as a separate final command (§2.1). It merges into the same dir.
- **Concurrency: the ceiling is `--n` (300).** Games run one at a time and each has only `n`
  episodes, so in-flight tops out at `min(concurrency, n) = 300`. Setting either knob above 300 is
  wasted. `--concurrency` = games the client plays in parallel; `--max-num-seqs` raises vLLM's 256
  default so all 300 actually run (else overflow queues — `Waiting: N` in the log).
- **NO `--enforce-eager` on the A100** (it's a WSL/no-nvcc local-only workaround that kills
  CUDA-graph speed — a 0.8B is launch-overhead-bound, so eager ~10×'d our latency).
- **Crash-safe (commit a7dbb60+):** every episode is flushed to disk the instant it finishes, and
  resume skips by *which targets are done*. Re-run the *same* command after any crash/disconnect —
  it loses nothing and continues. If the A100 OOMs (it won't at 0.8B), lower `--concurrency` / add
  `--max-model-len 4096`.
- `--games wordle` = ~13× faster if you only want the headline.

### 2.1 Base model — run LAST (separate command, same out dir)

```bash
# after the trained run finishes; base merges in as base.json + raw/base/
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-sft-wordle --base --epochs "" \
  --games all --n 300 --seed 0 --concurrency 300 --max-num-seqs 300 \
  --out /workspace/eval_results_v2/ \
  --push-results-repo saketh-chervu/word-games-eval --push-results-revision main
```

### 2.2 Monitor, reconnect, resume (run these in a SECOND terminal / pane)

```bash
# RECONNECT after an SSH/internet drop — the run survives inside tmux:
tmux attach -t eval        # if "no session": it died → just re-run the §2 command (resumes from disk)
tmux ls                    # list sessions

# PROGRESS — episodes completed per checkpoint (each *.jsonl climbs to 300; 13 games each).
# (plain bash loop; `watch -n10 '...'` also works now that §1 installs procps)
while true; do clear; date; \
  wc -l /workspace/eval_results_v2/raw/*/*.jsonl 2>/dev/null | grep -v total; \
  echo "--- finished checkpoints (pushed to HF) ---"; \
  ls /workspace/eval_results_v2/*.json 2>/dev/null || echo none; \
  sleep 10; done

# LIVE engine load (is it actually working, or wedged?) — running count + throughput:
curl -s http://127.0.0.1:8000/metrics | grep -E 'num_requests_(running|waiting)'
nvidia-smi                 # GPU util high = working; ~0% + no progress = wedged

# ONE-OFFs:
wc -l /workspace/eval_results_v2/raw/*/wordle.jsonl        # wordle progress, any checkpoint
find /workspace/eval_results_v2/raw -name '*.jsonl' | xargs wc -l | tail -1   # grand total
```

## 3. On LOCAL — fetch + analyze (no GPU, nothing manual on the pod)

```bash
cd /mnt/d/Projects/PixelPolicy
hf download saketh-chervu/word-games-eval --repo-type model --local-dir ./eval_results_v2   # `huggingface-cli` is dead — use `hf`
# (run it again anytime to pull newer checkpoints as they finish — it's incremental.
#  for a non-default results branch, e.g. full-v2:  --revision full-v2)

uv run --no-project inference/analysis/viz_eval.py --results ./eval_results_v2 --out eval_plots_v2
# new metric later? edit inference/metrics.py, then recompute from raw — NO re-inference:
uv run --package inference python -m inference.recompute --raw ./eval_results_v2/raw --out ./eval_results_v2
```

---

## Notes
- **`--no-sync` on every `uv run`** (torch stays cu128). **HF_TOKEN required** (private checkpoints).
- Raw generations are the source of truth → `recompute.py` derives any new metric offline, free.
- Sampling is frozen for fairness: `temperature 0.6, top_p 0.95, enable_thinking`, `max_tokens 4096`.
- **TODO (zero manual steps):** add `--push-results-repo` to `run_checkpoints` so it uploads
  `eval_results_v2/` to HF after each checkpoint (like training's grad-probe auto-push).
