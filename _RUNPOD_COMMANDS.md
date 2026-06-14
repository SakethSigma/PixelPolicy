# RunPod training commands (temp — delete when done)

Push the latest `main` first (this env has no GitHub creds). **torch must be the cu128 wheel
installed manually** — do NOT rely on `uv sync` for torch, and **run everything with `--no-sync`**
(plain `uv sync`/`uv run` re-resolve torch and break it: either cu130 = "driver too old", or missing
`libcudnn.so.9`).

---

## Step 0 — push main (run LOCALLY, in /mnt/d/Projects/PixelPolicy)

```bash
git push origin main
```

---

## One-time setup on EACH RunPod pod (template: Runpod Pytorch 2.4.0 · GPU: A100 80 GB · disk ≥50 GB)

> **GPU: A100 80 GB** (PCIe ~$1.39/hr, or SXM). 80 GB removes the large-vocab memory pressure — batch
> 16–32 fits comfortably. (A40 48 GB also works *with* the default `chunked_nll` loss; A100 just ends
> the OOM fights. `chunked_nll` is identical math to `nll`, just lower memory — TRL is making it the
> default in 1.7.)

```bash
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. clone + enter
git clone https://github.com/SakethSigma/PixelPolicy.git
cd PixelPolicy

# 3. deps, THEN install the cu128 torch manually (pulls the nvidia CUDA libs incl. cudnn)
uv sync --package training
uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu128

# 4. secrets (replace). Do NOT set WANDB_PROJECT (trainer forces it).
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. verify  → MUST print  <ver>+cu128 True   (note --no-sync)
uv run --no-sync --package training python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 6. (OPTIONAL) smoke test — longest sequences + chunked_nll loss; prints max seq len + batch fit.
#     Only honest once the latest main is pushed/pulled (older healthcheck runs plain nll → false OOM).
#     You can SKIP this and go straight to training: train.py already defaults to chunked_nll.
uv run --no-sync --package training python -m training.sft.healthcheck --variant full --bf16 \
  --gradient-checkpointing --sweep 4,8,16,32 --max-seq-len 4096
```

> **If step 5 fails** (`+cu130`, or `libcudnn.so.9` missing): you ran a plain `uv sync`/`uv run`
> which reverted torch. Re-run the `uv pip install torch … cu128` line, then always use `--no-sync`.

A100 80 GB fits **batch 32** (the jobs below use it). Effective batch = 32 × `--grad-accum 2` = **64**,
with `--lr 3e-5` scaled for it. **Keep batch + grad-accum + lr IDENTICAL across all three jobs** — a
different effective batch or LR would confound the wordle/full/curriculum comparison. Run each in
`tmux`; epochs push to the Hub as they finish.

---

## Job 1 — wordle-only baseline

```bash
uv run --no-sync --package training python -m training.sft.train --variant wordle \
  --output-dir ./runs/wordle --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 \
  --bf16 --gradient-checkpointing \
  --report-to wandb --run-name wordle --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-wordle --hub-per-epoch
```

## Job 2 — full set, no curriculum

```bash
uv run --no-sync --package training python -m training.sft.train --variant full \
  --output-dir ./runs/full --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 \
  --bf16 --gradient-checkpointing \
  --report-to wandb --run-name full --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-full --hub-per-epoch
```

## Job 3 — full set, curriculum (widening)

```bash
uv run --no-sync --package training python -m training.sft.train --variant curriculum \
  --curriculum-strategy widening --output-dir ./runs/curriculum --epochs 4 --lr 3e-5 \
  --per-device-batch-size 32 --grad-accum 2 --max-seq-len 4096 \
  --bf16 --gradient-checkpointing \
  --report-to wandb --run-name curriculum-widening --wandb-project pixelpolicy-sft \
  --push-to-hub --hub-model-id saketh-chervu/word-games-sft-curriculum --hub-per-epoch
```

---

## Notes

- **`--no-sync` on EVERY `uv run`** — non-negotiable. Plain `uv sync`/`uv run` re-resolve torch and
  break CUDA. Install torch once (step 3) and never let uv touch it again.
- **OOM fix:** Qwen3.5's ~248k vocab → huge fp32 cross-entropy logits. Trainer defaults to
  `loss_type=chunked_nll` (chunks the loss). Rows over `--max-seq-len` are dropped, not truncated.
  Still OOM? lower `--per-device-batch-size`, raise `--grad-accum` to keep the effective batch.
- **Tracking:** wandb project `pixelpolicy-sft`; runs `wordle` / `full` / `curriculum-widening`;
  per-layer `gradnorm/*` + `updnorm/*` panels. Don't set `WANDB_PROJECT`.
- **Checkpoints:** `saketh-chervu/word-games-sft-<variant>`, revisions `epoch-1..4` (+ final on
  `main`), weights-only — for per-checkpoint inference eval (`vllm … --revision epoch-N`).
- **Smoke train** first (optional): append `--max-steps 5 --epochs 1 --eval-samples-per-game 8 --eval-samples-all 16`.
- **Disk:** pod volume ≥50 GB (torch + model + 4 epoch checkpoints).
- **Push everything first:** Step 0 must include the latest commits (cu128 pin removed, over-length
  filter, healthcheck uses chunked_nll). `git log origin/main..main` should be empty after pushing.
```
