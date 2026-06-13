# Training — SFT of Qwen3.5-0.8B on the word-games dataset

Supervised fine-tuning of `Qwen/Qwen3.5-0.8B` on
[`saketh-chervu/word-games-distillation`](https://huggingface.co/datasets/saketh-chervu/word-games-distillation)
via HuggingFace **TRL `SFTTrainer`**. Three recipes share one trainer:

| `--variant` | data | sampler |
|-------------|------|---------|
| `wordle` | only `game_name=="wordle"` rows | shuffled |
| `full` | all valid games | shuffled |
| `curriculum` | all valid games, difficulty-aware order | curriculum (see below) |

Designed to run on a **separate training machine with no git checkout** — the **only output channel
is the HuggingFace Hub**: every epoch checkpoint is pushed to its own revision (`epoch-N`) so you can
later host each one and measure downstream accuracy. Run comparison is via **Weights & Biases** (free,
cloud) so runs from different machines land in one project.

## Layout

```
training/
├── pyproject.toml
├── README.md
├── CURRICULUM_NOTES.md          # research + hypotheses behind the curriculum (brainstorm doc)
└── sft/
    ├── format.py                # row → (prompt, completion); the ONLY chat-template seam
    ├── data_flat.py             # loader #1: wordle-only + full (non-curriculum)
    ├── data_curriculum.py       # loader #2: widening | sorted | weighted
    ├── train.py                 # SFTTrainer entrypoint (4 epochs, save+push every checkpoint)
    ├── stats.py                 # sequence-length stats (CPU-only; pick --max-seq-len)
    ├── healthcheck.py           # pre-flight: model/data load, train stable, batch-size sweep
    ├── dynamics.py              # per-layer grad/update-norm callback → wandb (--gradlog-steps)
    └── upload.py                # push one checkpoint dir → Hub revision
```

## Setup (training machine)

```bash
# from the repo root (copy/clone the repo once; git not otherwise needed)
uv sync --package training       # installs torch, transformers, trl, accelerate, hub, wandb

# .env (or real env vars):
#   HF_TOKEN=hf_...               # write token — pushes checkpoints to the Hub
#   WANDB_API_KEY=...             # free personal account — cross-machine run comparison
```

VRAM: a 0.8B full fine-tune fits a ~24 GB card at `--per-device-batch-size 4 --grad-accum 8
--max-seq-len 4096 --bf16 --gradient-checkpointing`. Smaller card → lower batch size, raise
`--grad-accum` to keep the effective batch.

## Data format & filtering

- Each dataset row stores `messages` (the conversation prefix, **already including the system turn**)
  and `completion` (the assistant reply, with `<think>` kept). `format.build_example` renders the
  prompt with the model's chat template (`add_generation_prompt=True, enable_thinking=True`) exactly
  as the vLLM inference server does, then hands TRL a `{prompt, completion}` pair → **completion-only
  loss** (prompt tokens masked; `packing=False`).
- **Only `valid==True` rows are trained** (the quality gate; for Wordle this is `<think>` format
  compliance, already re-derived upstream). The dry run prints how many invalid rows were dropped.

## Sequence-length stats (run anytime — CPU only, no GPU)

```bash
uv run --package training python -m training.sft.stats                 # full train split
uv run --package training python -m training.sft.stats --games wordle  # one game
```

Measured over all **85,959 valid train rows** (prompt+completion tokens, Qwen3.5 tokenizer):
`p50=271, p90=473, p95=506, p99=863, max=5553`. **99.4% fit in ≤1024 tokens; 95.6% in ≤512.** Wordle
(multi-turn) is by far the longest (mean 770, p99 1425); every other game is short. We use
**`--max-seq-len 4096`** (the default) so no data is lost — every row fits except one Wordle outlier
(5553 tokens) whose tail is truncated. SFTTrainer *truncates* over-length rows, it never filters
them, so the row count is unchanged regardless of `--max-seq-len`.

## Health check / batch-size sweep (GPU; run on the training machine)

```bash
# load model+data, run a few steps, assert loss is finite
uv run --package training python -m training.sft.healthcheck --variant full --bf16 --steps 10

# find the largest per-device batch that fits + peak VRAM per size
uv run --package training python -m training.sft.healthcheck --variant full --bf16 \
  --gradient-checkpointing --sweep 1,2,4,8,16 --max-seq-len 4096
```

## Tokenizer & loss (important details)

- **Completion-only loss.** We hand TRL a `{prompt, completion}` pair, so `SFTTrainer` masks the
  prompt tokens and trains the loss on the completion only (`packing=False` keeps that masking
  intact). We never teach the prompt. This is the prompt-completion path in `format.build_example`.
- **Padding side.** Qwen tokenizers default to `padding_side="left"` (right for batched
  *generation*). For causal-LM SFT we set **`padding_side="right"`** in `train.py` so completion
  tokens stay contiguous and next-token loss aligns with the attention mask (pads are masked out of
  the loss). Inference/vLLM handles its own left padding — separate concern.
- **`enable_thinking=True`** in the chat template, matching the vLLM inference server, so the prompt
  is byte-identical to inference and the `<think>` completions train correctly.

## Dry run first (verify data loading, no model)

```bash
uv run --package training python -m training.sft.data_flat       --variant wordle --dry-run
uv run --package training python -m training.sft.data_flat       --variant full   --dry-run
uv run --package training python -m training.sft.data_curriculum --strategy widening --dry-run
```

Each prints: total vs valid (dropped-invalid) counts, per-game breakdown, prompt/completion token-
length percentiles, and 1–2 fully rendered examples. The curriculum dry run also prints the per-
bucket stage/reasoning composition so you can see the ordering. Add `--no-tokenize` for counts only.

## Train

```bash
# 1) wordle-only baseline
uv run --package training python -m training.sft.train --variant wordle \
  --output-dir ./runs/wordle --epochs 4 --lr 2e-5 --per-device-batch-size 4 --grad-accum 8 \
  --max-seq-len 4096 --bf16 --gradient-checkpointing --report-to wandb --run-name wordle \
  --push-to-hub --hub-model-id <you>/word-games-sft-wordle --hub-per-epoch

# 2) full set, no curriculum
uv run --package training python -m training.sft.train --variant full \
  --output-dir ./runs/full --epochs 4 --lr 2e-5 --per-device-batch-size 4 --grad-accum 8 \
  --max-seq-len 4096 --bf16 --gradient-checkpointing --report-to wandb --run-name full \
  --push-to-hub --hub-model-id <you>/word-games-sft-full --hub-per-epoch

# 3) full set, curriculum (default widening; try --curriculum-strategy sorted|weighted)
uv run --package training python -m training.sft.train --variant curriculum \
  --curriculum-strategy widening --output-dir ./runs/curriculum --epochs 4 --lr 2e-5 \
  --per-device-batch-size 4 --grad-accum 8 --max-seq-len 4096 --bf16 --gradient-checkpointing \
  --report-to wandb --run-name curriculum-widening \
  --push-to-hub --hub-model-id <you>/word-games-sft-curriculum --hub-per-epoch
```

Multi-GPU: prefix with `uv run --package training accelerate launch -m training.sft.train …`.
Smoke test: add `--max-steps 5 --epochs 1 --eval-samples-per-game 8 --eval-samples-all 16`.

All hyperparameters are flags: `--lr --per-device-batch-size --grad-accum --max-seq-len
--warmup-ratio --weight-decay --lr-scheduler --epochs --seed` (+ `--bf16/--fp16`,
`--gradient-checkpointing`).

## Checkpoints & the Hub

- `save_strategy="epoch"`, `save_total_limit=None` → **all 4 epoch checkpoints are kept**.
- With `--push-to-hub --hub-per-epoch`, each epoch checkpoint is pushed **as it is saved** to Hub
  revision `epoch-N` (weights-only by default — optimizer state skipped, all inference needs). The
  final weights also go to `main`. Completed epochs survive even if the run later dies.
- Re-push or push one checkpoint by hand:
  ```bash
  uv run --package training python -m training.sft.upload ./runs/full/checkpoint-1234 \
    --hub-model-id <you>/word-games-sft-full --revision epoch-2
  ```

## Evaluation

- **In-training:** `eval_dataset` is a **dict** → the built-in Trainer→wandb path logs an aggregated
  `eval_all_loss` **and** a per-game `eval_<game>_loss` each epoch (no custom metric code; everything
  stays on the training box + wandb). Tune with `--eval-samples-all` / `--eval-samples-per-game`
  (set both to 0 to disable). **Loss is only a coarse sanity signal.**
- **The real metric — done later, separately:** host each checkpoint and measure **downstream task
  accuracy** on a fixed test set. Because every epoch is its own Hub revision, you can do:
  ```bash
  uv run --package inference python -m inference.server \
    --model <you>/word-games-sft-curriculum --revision epoch-3
  # then run agents/run.py (Wordle) / the per-game eval harness against that server
  ```
  (That metrics harness is not in this package — it's the separate inference step.)

## Curriculum

See [CURRICULUM_NOTES.md](CURRICULUM_NOTES.md) for the research, hypotheses, and the open design
questions. In short: strict easy→hard (`sorted`) is the *risky* arm (it can erode the fragile
reasoning behavior over 4 epochs); the default `widening` strategy introduces difficulty progressively
but keeps the 4 chain-of-thought games (wordle, anagram, crossword, mistakeid) present throughout plus
a small replay slice, which is the arm most likely to beat a plain shuffle without losing reasoning.
