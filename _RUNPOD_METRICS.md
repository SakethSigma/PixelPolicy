# RunPod metrics / eval commands (temp — delete when done)

Run **checkpoint evaluation** (play all games, score win-rate/accuracy) on a remote GPU, push the
results to HF, and analyze on local. Separate from training (see `_RUNPOD_COMMANDS.md`).

**Why remote:** a 0.8B model barely uses an A100 80 GB, so you can run `--concurrency 32` (vs the
12 GB-local `4`) → ~8× the in-flight episodes → ~an hour instead of many. Split work across machines
(e.g. e1/e2 local, **e3/e4 remote**); each checkpoint is an independent `<label>.json` + `raw/<label>/`,
so you merge the dirs locally.

---

## 1. One-time setup on the eval pod (A100, RunPod **PyTorch 2.8.0** / CUDA 12.x)

The lockfile now pins a coherent set — **vllm 0.19.0** (cu128 + Qwen3.5 window),
**prometheus-fastapi-instrumentator 7.1.0** (8.0.0 crashes vLLM's API → 500 on every request), and
**cu12 torch** — so setup is just `uv sync`. No version-walking, no `libcudart.so.13`, no prometheus 500.

```bash
# fresh pods are bare — git/tmux are usually missing:
apt-get update && apt-get install -y git tmux curl procps
curl -LsSf https://astral.sh/uv/install.sh | sh ; source $HOME/.local/bin/env

git clone https://github.com/SakethSigma/PixelPolicy.git && cd PixelPolicy
git log --oneline -1                          # MUST be >= bf5de72 (the vllm/instrumentator pins)

uv sync --package inference                    # installs the locked set: vllm 0.19.0 + instrumentator 7.1.0 + cu12 torch
export HF_TOKEN=hf_xxxxxxxx                    # private checkpoints

# verify — MUST print 0.19.0 with NO "libcudart.so.13":
uv run --no-sync --package inference vllm --version

# DECISIVE web-stack test — the endpoint that used to 500 must return 200:
uv run --no-sync --package inference vllm serve saketh-chervu/word-games-sft-full-v2 \
  --revision epoch-1 --host 127.0.0.1 --port 8000 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":0,"video":0}' &
sleep 120; curl -s -o /dev/null -w "models: %{http_code}\n" http://127.0.0.1:8000/v1/models; kill %1
```

**Only if** `vllm --version` errors with `libcudart.so.13` (an older CUDA-12.4 template / driver
mismatch), pin the cu128 wheels in ONE deterministic install — no walking:
```bash
uv pip install --reinstall "vllm==0.19.0" --torch-backend=cu128
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
- **`--no-sync` on every `uv run`** (don't let it re-resolve away the locked vllm/torch). **HF_TOKEN required** (private checkpoints).
- Raw generations are the source of truth → `recompute.py` derives any new metric offline, free.
- Sampling is frozen for fairness: `temperature 0.6, top_p 0.95, enable_thinking`, `max_tokens 4096`.
- **TODO (zero manual steps):** add `--push-results-repo` to `run_checkpoints` so it uploads
  `eval_results_v2/` to HF after each checkpoint (like training's grad-probe auto-push).
