# RunPod metrics / eval commands (temp — delete when done)

Run **checkpoint evaluation** (play all games, score win-rate/accuracy) on a remote GPU, push the
results to HF, and analyze on local. Separate from training (see `_RUNPOD_COMMANDS.md`).

**Why remote:** a 0.8B model barely uses an A100 80 GB, so you can run `--concurrency 32` (vs the
12 GB-local `4`) → ~8× the in-flight episodes → ~an hour instead of many. Split work across machines
(e.g. e1/e2 local, **e3/e4 remote**); each checkpoint is an independent `<label>.json` + `raw/<label>/`,
so you merge the dirs locally.

---

## 1. One-time setup on the eval pod (A100, template: Runpod Pytorch 2.4.0)

```bash
apt-get update && apt-get install -y tmux ; tmux new -s eval     # survive disconnects
curl -LsSf https://astral.sh/uv/install.sh | sh ; source $HOME/.local/bin/env
git clone https://github.com/SakethSigma/PixelPolicy.git && cd PixelPolicy
git reset --hard origin/main                                     # exact pushed code (fetch is implicit on fresh clone)

uv sync --package inference                                      # eval harness deps (agents + distillation + vllm)
# vLLM has a compiled CUDA extension, so vLLM AND torch must be the SAME CUDA (12.8). Two
# constraints pull in opposite directions, so you must hit a WINDOW — not just "pin low":
#   1. Qwen3.5 (Qwen3_5ForConditionalGeneration) needs vLLM >= 0.17.0. Older vllm (e.g. 0.10.2)
#      errors: "Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported".
#   2. vLLM >= ~0.21 defaults to a CUDA-13 wheel → "libcudart.so.13" on a 12.8 driver. And
#      --torch-backend=cu128 only fixes TORCH's index, NOT vllm's own compiled _C — so even with
#      cu128 torch, a cu13 vllm wheel still dies (vllm#43435).
# => Target vLLM 0.17–0.20: new enough for Qwen3.5, old enough to still ship a cu128 wheel.
# 0.19.0 is CONFIRMED working on the A100 pod (cu128, Qwen3.5 loads). Use it:
uv pip install --reinstall "vllm==0.19.0" --torch-backend=cu128
#   if it ever fails the `vllm --version` test below, walk DOWN: 0.19.0 → 0.18.0 → 0.17.0
#   (0.20.0+ may drag in cu13 → "libcudart.so.13")
# If a version still drags in cu13, install its explicit cu128 RELEASE wheel directly (skips the
# PyPI default; some URLs 404 → try the next minor, vllm#37847):
#   uv pip install https://github.com/vllm-project/vllm/releases/download/v0.18.0/vllm-0.18.0+cu128-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128
# CLEANER if you control the image: use a CUDA-13 RunPod template (driver >= 580) instead of the
# "PyTorch 2.4.0" (CUDA 12.1) one, then plain `uv pip install vllm` (latest, cu13) just works.

export HF_TOKEN=hf_xxxxxxxx                                      # needed to DOWNLOAD the (private) checkpoints
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
# (plain bash loop — `watch` needs `apt-get install -y procps` which the pod lacks)
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
huggingface-cli download saketh-chervu/word-games-eval --repo-type model --local-dir ./eval_results_v2
# (run it again anytime to pull newer checkpoints as they finish — it's incremental)

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
