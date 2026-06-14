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

## 2. Run the eval — ONE command, fully hands-off (all epochs + base, all games, auto-push)

```bash
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-sft-wordle --epochs 1,2,3,4 --base \
  --games all --n 300 --seed 0 --enforce-eager --concurrency 32 \
  --out /workspace/eval_results_v2/ \
  --push-results-repo saketh-chervu/word-games-eval --push-results-revision main
```
- Evaluates **base + epoch-1..4** on all 13 games (300 each), writes metrics + raw to the persistent
  volume, and **auto-uploads the whole eval dir to `saketh-chervu/word-games-eval` after each
  checkpoint** (created automatically; a *dedicated* repo so model weights aren't mixed in). No
  manual upload — launch it in tmux and walk away.
- **Crash-tolerant:** if the vLLM server dies, re-run the *same* command — resume-from-raw skips
  completed episodes and continues. If the A100 OOMs (it won't at 0.8B), lower `--concurrency` /
  add `--max-model-len 4096`.
- `--games wordle` = ~13× faster if you only want the headline.

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
