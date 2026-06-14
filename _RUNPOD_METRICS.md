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
uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu128

export HF_TOKEN=hf_xxxxxxxx                                      # needed to DOWNLOAD the (private) checkpoints
uv run --no-sync --package inference python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # +cu128 True
```

## 2. Run the eval (writes to the persistent volume)

```bash
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-sft-wordle --epochs 3,4 \
  --games all --n 300 --seed 0 --enforce-eager --concurrency 32 \
  --out /workspace/eval_results_v2/
```
- `--epochs 3,4` → just those checkpoints (run e1/e2 elsewhere). `--games wordle` = ~13× faster if you
  only want the headline. Add `--base` for the untrained reference line.
- Outputs: `/workspace/eval_results_v2/<label>.json` (metrics) + `raw/<label>/<game>.jsonl` (raw
  generations, flushed per episode).
- **Crash-tolerant:** if the vLLM server dies mid-run, re-run the *same* command — resume-from-raw
  skips completed episodes and continues. On a tight card add `--max-model-len 4096` and lower
  `--concurrency` (total KV ≈ concurrency × seq-len).

## 3. Push results to HF (no git on the pod)

```bash
huggingface-cli upload saketh-chervu/word-games-sft-wordle \
  /workspace/eval_results_v2 eval_results_v2 --revision eval --repo-type model
```
(Uploads metrics + `raw/` to the `eval` branch of the model repo. Re-run anytime to refresh.)

## 4. On LOCAL — fetch, merge, analyze (no GPU)

```bash
cd /mnt/d/Projects/PixelPolicy
huggingface-cli download saketh-chervu/word-games-sft-wordle --revision eval \
  --include "eval_results_v2/**" --repo-type model --local-dir ./fetched
cp -rn ./fetched/eval_results_v2/* ./eval_results_v2/            # merge remote e3/e4 with local e1/e2

# plots + paste-ready table:
uv run --no-project inference/analysis/viz_eval.py --results eval_results_v2 --out eval_plots_v2
# new metric later? edit inference/metrics.py, then recompute from raw — NO re-inference:
uv run --package inference python -m inference.recompute --raw eval_results_v2/raw --out eval_results_v2
```

---

## Notes
- **`--no-sync` on every `uv run`** (torch stays cu128). **HF_TOKEN required** (private checkpoints).
- Raw generations are the source of truth → `recompute.py` derives any new metric offline, free.
- Sampling is frozen for fairness: `temperature 0.6, top_p 0.95, enable_thinking`, `max_tokens 4096`.
- **TODO (zero manual steps):** add `--push-results-repo` to `run_checkpoints` so it uploads
  `eval_results_v2/` to HF after each checkpoint (like training's grad-probe auto-push).
