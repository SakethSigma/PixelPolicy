# RunPod training commands (temp — delete when done)

Push the latest `main` first. **Two hard rules that this saga taught us:**
1. **Write checkpoints to `/workspace`** (the persistent volume), NEVER to the repo under `/` (the
   ephemeral *container disk* — it dies with the container / can't be migrated). `--output-dir
   /workspace/runs/<variant>`.
2. **Always run inside `tmux`** — a web-terminal/internet drop kills your *view*, not the job, and
   `tmux attach` gets the session back. (Losing your internet does NOT stop the RunPod container.)

Plus: torch must be the cu128 wheel (`uv pip install … cu128`) and every `uv run` uses `--no-sync`
(plain `uv sync`/`uv run` re-resolve torch → cu130 "driver too old" or missing `libcudnn.so.9`).

**Container disk vs volume:** a pod = a container (`/`, ephemeral, ~dies on stop/migrate/recreate)
**+** a persistent volume (`/workspace`, survives). The repo/`.venv` can live on `/` (re-cloneable);
**checkpoints must live on `/workspace`** so a pod death never strands them.

---

## Step 0 — push main (run LOCALLY, in /mnt/d/Projects/PixelPolicy)

```bash
git push origin main
```

---

## One-time setup on EACH RunPod pod (template: Runpod Pytorch 2.4.0 · GPU: A100 80 GB)

```bash
# 0. ALWAYS work inside tmux (survives terminal/internet drops; reconnect with `tmux attach`)
apt-get update && apt-get install -y tmux       # not preinstalled in the RunPod image
tmux new -s train

# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. clone + enter
git clone https://github.com/SakethSigma/PixelPolicy.git
cd PixelPolicy

# 3. deps, THEN install the cu128 torch manually (pulls the nvidia CUDA libs incl. cudnn)
uv sync --package training
uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu128

# 4. secrets. Do NOT set WANDB_PROJECT (trainer forces it).
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. verify  → MUST print  <ver>+cu128 True
uv run --no-sync --package training python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **If step 5 fails** (`+cu130` / missing `libcudnn.so.9`): re-run the `uv pip install torch … cu128`
> line, then always use `--no-sync`.

Batch 32 × `--grad-accum 2` = effective 64, `--lr 3e-5`. **Keep batch/accum/lr identical across the
three jobs** (else the comparison is confounded). Smaller GPU? keep effective batch 64 (e.g. `16 × 4`).

---

## Training jobs — FRESH runs (start a new run here)

> A brand-new training run (from the base model) = run the matching Job below. **Don't** use the
> Crash-recovery section for a fresh run — that's only for continuing an interrupted one.

## Job 1 — wordle-only baseline

```bash
uv run --no-sync --package training python -m training.sft.train --variant wordle \
  --output-dir /workspace/runs/wordle --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 \
  --bf16 --gradient-checkpointing \
  --report-to wandb --run-name wordle --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-wordle --hub-per-epoch
```

## Job 2 — full set, no curriculum

> The original `…-sft-full` repo holds the dead run's epoch-1/2 — **don't overwrite it.** Push this
> clean re-run to a NEW id (`…-sft-full-v2`) and use a distinct `--run-name`/`--output-dir`.

```bash
uv run --no-sync --package training python -m training.sft.train --variant full \
  --output-dir /workspace/runs/full-v2 --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 \
  --bf16 --gradient-checkpointing \
  --report-to wandb --run-name full-v2 --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-full-v2 --hub-per-epoch \
  --game-probe-steps 250
```
> `--game-probe-steps 250` → every 250 steps, probe each game's per-layer/component grad signature
> to `/workspace/runs/full-v2/grad_probe.jsonl` (which game drives which layer — for offline analysis).
> Cheap, off-GPU-memory-neutral (runs after the step, batch `k=8` < training batch). 0 disables.

## Job 3 — full set, curriculum (widening)

```bash
uv run --no-sync --package training python -m training.sft.train --variant curriculum \
  --curriculum-strategy widening --output-dir /workspace/runs/curriculum --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 \
  --bf16 --gradient-checkpointing \
  --report-to wandb --run-name curriculum-widening --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-curriculum --hub-per-epoch \
  --game-probe-steps 250
```
(wordle Job 1 is single-game, so the per-game probe is trivial there — omit it, or use a small
`--game-probe-steps 50` since wordle has few steps.)

---

## Crash recovery — resume an INTERRUPTED run (NOT for fresh runs)

> Use this **only** to continue a run that already produced ≥1 checkpoint and then died. For a
> brand-new run, use the Job commands above. (A run started with the *old* code has no `resume`
> branch and can't be resumed — start it fresh instead.)

Each epoch now pushes TWO things to HF:
- `epoch-1..4` — weights-only (for inference), and
- **`resume`** — the **FULL** checkpoint (optimizer + scheduler + RNG + trainer_state), overwritten
  each epoch. So even if the pod and its `/workspace` are both gone, the last completed epoch is
  recoverable from HF.

On a fresh GPU pod (after the one-time setup above):
```bash
cd /PixelPolicy
mkdir -p /workspace/runs/<variant>
huggingface-cli download saketh-chervu/word-games-sft-<variant> --revision resume \
  --local-dir /workspace/runs/<variant>/resume-ckpt

# resume — restores optimizer/scheduler/step/epoch and finishes the remaining epochs:
uv run --no-sync --package training python -m training.sft.train --variant <variant> \
  --output-dir /workspace/runs/<variant> --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 --bf16 --gradient-checkpointing \
  --report-to wandb --run-name <name> --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-<variant> --hub-per-epoch \
  --resume-from /workspace/runs/<variant>/resume-ckpt
```
(If `/workspace` on the *same* pod survived, skip the download and just add `--resume` — it
auto-finds the latest local checkpoint.)

---

## Evaluating on a remote GPU (A100) — faster, + push results back to local

A 0.8B model is tiny on an A100 80 GB, so the 12 GB-local `--concurrency 4` bottleneck is gone —
run `--concurrency 32` (≈8× the in-flight episodes → ~hour, not many hours). Split work across
machines (e.g. e1/e2 local, **e3/e4 remote**); each checkpoint is an independent `<label>.json` +
`raw/<label>/`, so you just merge the dirs locally.

**On the A100 pod** (same one-time setup as above: tmux, uv, clone, cu128 torch, `--no-sync`,
`export HF_TOKEN=…`; the eval needs the inference deps):
```bash
cd /PixelPolicy
uv sync --package inference     # agents + distillation + vllm for the eval harness
uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu128

# run e3,e4 on all games — write to the PERSISTENT volume:
uv run --no-sync --package inference python -m inference.run_checkpoints \
  --repo saketh-chervu/word-games-sft-wordle --epochs 3,4 \
  --games all --n 300 --seed 0 --enforce-eager --concurrency 32 \
  --out /workspace/eval_results_v2/

# push the whole eval dir (metrics + raw/) to a branch of the model repo:
huggingface-cli upload saketh-chervu/word-games-sft-wordle \
  /workspace/eval_results_v2 eval_results_v2 --revision eval --repo-type model
```

**On LOCAL — fetch the remote results and merge with your local ones:**
```bash
huggingface-cli download saketh-chervu/word-games-sft-wordle --revision eval \
  --include "eval_results_v2/**" --repo-type model --local-dir ./fetched
cp -rn ./fetched/eval_results_v2/* ./eval_results_v2/   # merge e3/e4 into the local e1/e2 dir
uv run --no-project inference/analysis/viz_eval.py --results eval_results_v2 --out eval_plots_v2
```
Resumable + crash-tolerant: re-run the same `run_checkpoints` command and it skips completed episodes
(resume-from-raw). (TODO if you want zero manual steps: add a `--push-results-repo` flag so the
orchestrator uploads `eval_results_v2/` to HF after each checkpoint — say the word.)

---

## Notes

- **`--no-sync` on EVERY `uv run`** — non-negotiable; install torch once (step 3), never let uv touch it.
- **OOM:** Qwen3.5's ~248k vocab → huge fp32 cross-entropy logits; trainer defaults to
  `loss_type=chunked_nll`. Over-length rows are dropped. Still OOM? lower batch, raise grad-accum.
- **Checkpoints:** `epoch-1..4` (weights-only, inference) + `resume` (full, recovery) on the Hub, AND
  local full checkpoints under `/workspace/runs/<variant>/`.
- **Tracking:** wandb project `pixelpolicy-sft`; per-layer `gradnorm/*` + `updnorm/*` panels now also
  split per block into `attn_NN` / `mlp_NN` / `norm_NN` (plus the whole-block `layer_NN`).
- **Push everything first:** `git log origin/main..main` should be empty after Step 0.
- **`grad_probe.jsonl` exfils automatically — no manual step.** With `--game-probe-steps > 0` and
  `--push-to-hub`, the probe JSONL is pushed to **`<hub-model-id>@probe/grad_probe.jsonl`** each epoch
  AND synced to the wandb run. (It also lives on `/workspace`, which persists.) Pull it to LOCAL for
  analysis from either:
  ```bash
  huggingface-cli download saketh-chervu/word-games-sft-full-v2 --revision probe \
    --include grad_probe.jsonl --local-dir ./probe
  # …or grab it from the wandb run's Files tab.
  ```
  So you can sync the branch, launch, and sleep — the per-game grad data lands on HF on its own.
```
