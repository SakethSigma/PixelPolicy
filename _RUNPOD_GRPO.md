# RunPod GRPO commands (temp — delete when done)

GRPO RL (verl) on a remote **A100 80 GB** pod. Develop locally, run here. Mirrors the discipline of
`_RUNPOD_COMMANDS.md`: **two hard rules** —
1. **Write checkpoints to `/workspace`** (persistent volume), never to the repo under `/` (ephemeral
   container disk). The trainer's `--work-dir /workspace/grpo` already does this.
2. **Always run inside `tmux`** — a dropped SSH/web terminal kills your *view*, not the job.

**Isolation:** verl gets its **own venv** (`/workspace/verl-venv`), separate from the workspace `.venv`
and from `inference/`'s pinned `vllm==0.19.0`. The two never mix; they only share the source tree via
`PYTHONPATH=.`. Eval still runs from the workspace `.venv` (unchanged).

---

## Step 0 — push main (run LOCALLY)

```bash
git push origin main
```

---

## One-time setup on the GRPO pod (A100 80 GB · RunPod PyTorch 2.x / CUDA 12.x)

```bash
# 0. tmux (survives disconnects)
apt-get update && apt-get install -y git tmux curl procps
tmux new -s grpo

# 1. clone the repo (source tree only; verl is installed separately)
git clone https://github.com/SakethSigma/PixelPolicy.git
cd PixelPolicy && git log --oneline -1

# 2. ISOLATED verl venv (does NOT touch the workspace .venv / inference vllm 0.19)
python -m venv /workspace/verl-venv
source /workspace/verl-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128    # match the cu128 driver FIRST
pip install "verl[sglang]"            # verl + its own sglang/vllm/flashinfer (fallback: pip install "verl[vllm]")
pip install pandas pyarrow huggingface-hub wandb python-dotenv pydantic transformers

# 3. secrets
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. verify verl imports + sees the GPU  → MUST print a version and True
/workspace/verl-venv/bin/python -c "import verl, torch; print(verl.__version__, torch.cuda.is_available())"
```

> If `verl[sglang]` fights the cu128 driver / flashinfer wheels, fall back to `pip install "verl[vllm]"`
> and set `actor_rollout_ref.rollout.name=vllm` (append it as a trailing override to the launch command).

---

## Step A — local smoke (no GPU): confirm the rollout/reward/packing pipeline

Runs the **full per-turn pipeline** (draw targets → roll_batch → per-turn reward → pack) with a fake
tokenizer, no verl/GPU. Do this anywhere (even your laptop) before touching the pod:
```bash
cd /workspace/PixelPolicy
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --arm b --smoke-local \
  --train-batch-size 3 --group-size 4
PYTHONPATH=. /workspace/verl-venv/bin/python -m pytest training/grpo/tests/ -q   # 29 tests
```

## Step B — complete the two VERL-VERSION seams in `training/grpo/trainer.py`

The rollout/reward/packing are done and tested; the verl optimizer glue has **two clearly-marked
seams** to wire against the *installed* verl (copy from `verl/recipe/dapo/main_dapo.py`):
- **`_build_workers(cfg)`** — spin up verl's actor/ref/rollout RayWorkerGroup (ResourcePoolManager, Role).
- **`verl_batch_generate(...)`** — `list[prompt_ids] → list[response_ids]` via the rollout engine's
  `generate_sequences`, and the final optimize step (`to_dataproto` → `compute_grpo_outcome_advantage`
  with `index=uid` → `update_actor` → save/push `step-N`).

These are localized; everything around them is verl-agnostic. (This is the one integration risk the
plan flagged — verl's worker API moves across releases.)

## Step C — PROBE (5 steps) — verify the seam + read steps/hour, in tmux

**Start with Arm B** (full-v2 init → wordle, the key transfer comparison):
```bash
cd /workspace/PixelPolicy
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --arm b \
  --work-dir /workspace/grpo --hub-model-id saketh-chervu/word-games-grpo-armb \
  --max-steps 5
```
Watch wandb (`pixelpolicy-grpo`): reward + breakdown (`novel/format/invalid`), `entropy`, `reward_std`,
`clip_ratio/high`. Read steps/hour, extrapolate (PLAN.md: ~5–12 min/step), then launch the full run.

---

## Training jobs (run each inside tmux; detach with `Ctrl-b` then `d`)

### Arm B — full-v2 init → GRPO on wordle  (FIRST RUN, the key comparison)
```bash
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --arm b \
  --work-dir /workspace/grpo --hub-model-id saketh-chervu/word-games-grpo-armb \
  --wandb-project pixelpolicy-grpo
```

### Arm A — wordle-SFT init → GRPO on wordle  (the baseline to compare B against)
```bash
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --arm a \
  --work-dir /workspace/grpo --hub-model-id saketh-chervu/word-games-grpo-arma \
  --wandb-project pixelpolicy-grpo
```

### Arm C — full-v2 init → GRPO on ALL games (dilution check)
```bash
PYTHONPATH=. /workspace/verl-venv/bin/python -m training.grpo.train --arm c \
  --work-dir /workspace/grpo --hub-model-id saketh-chervu/word-games-grpo-armc \
  --wandb-project pixelpolicy-grpo
```

**Parallelism:** the cheapest way to cut calendar time for three arms is **three single-GPU pods in
parallel** (one arm each) — no FSDP comms tax, same total GPU-hours. A single 2× A100 pod with
`--n-gpus 2` ~halves one arm's wall-clock but costs ~2×; use it only if one arm alone is too slow.

**Collapse showing? (declining `reward_std`/`entropy` on wandb):** first lever is KL removal
`--kl-loss-coef 0` (DAPO), then StarPO-S filtering `--filter-top-variance-frac 0.5`. Clip-higher is on
by default (the primary guard).

**Knobs if a step runs hot (check the probe first):** `--group-size 4` (was 8), `--train-batch-size 32`,
`--no-thinking` (last resort — diverges from eval), or append verl overrides like
`actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.actor.fsdp_config.param_offload=true`.

---

## Monitor / reconnect (second pane)

```bash
tmux attach -t grpo            # reconnect after a drop; "no session" → it died, re-run the job
nvidia-smi                     # GPU util high = working
# wandb (project pixelpolicy-grpo) shows reward + the per-component breakdown (novel/format/invalid/win),
# KL, and steps/sec. Watch that eval win-rate (below), not just shaped reward, climbs.
```

Checkpoints push to `saketh-chervu/word-games-grpo-<arm>@step-N` as they save (weights-only), and also
live under `/workspace/grpo/ckpt` on the persistent volume — a pod death never strands them.

---

## Evaluate the GRPO checkpoints (reuse the frozen harness — see `_RUNPOD_METRICS.md`)

From the **workspace `.venv`** (vllm 0.19, untouched), on this or another pod:

```bash
uv sync --package inference        # the locked vllm 0.19 eval stack
export HF_TOKEN=hf_xxxxxxxx
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-grpo-arma --revisions step-25,step-50,step-75 \
  --games wordle --n 300 --seed 0 --concurrency 300 --max-num-seqs 300 \
  --out /workspace/eval_results_grpo/ \
  --push-results-repo saketh-chervu/word-games-eval --push-results-revision grpo
```
Sampling is frozen (`temperature 0.6, top_p 0.95, enable_thinking`, max_tokens 4096) on the **val**
pool, which training never touches. Pull + plot on LOCAL with
`inference/analysis/viz_eval.py --results ./eval_results_grpo --out eval_plots_grpo`.

**The headline comparison:** Arm A vs Arm B wordle win-rate across `step-N` — does broad SFT (B) give a
better RL starting point than narrow SFT (A)? Plus both vs the SFT-only baselines.

---

## Notes
- **`--no-sync` on every workspace `uv run`** (don't let it re-resolve away vllm 0.19). The verl venv is
  separate and uses its own `python` directly.
- **verl version:** the two worker/optimizer seams in `training/grpo/trainer.py` and a few config keys
  are version-sensitive (marked `# VERL-VERSION`). Complete them against the installed verl (copy from
  `verl/recipe/dapo/main_dapo.py`); `--smoke-local` validates everything that doesn't touch verl.
- **Crash recovery:** verl writes full checkpoints under `/workspace/grpo/ckpt`; re-launch with verl's
  `trainer.resume_mode=auto` (append as an override) to continue from the latest.
```
