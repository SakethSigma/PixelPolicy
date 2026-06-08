# Distillation

Generate **teacher trajectories** with Claude, turn them into SFT data, combine across
games, and push to the HuggingFace Hub. The student (a small open model, e.g.
`Qwen/Qwen3.5-0.8B`) is later trained to imitate the teacher.

> This package only **drives** the existing layers — it adds no game/agent/model logic.
> A teacher is just an `LLMBackend`, so the *same* inference game loop (`agents.rollout.run_eval`)
> records its trajectories. See **[PLAN.md](PLAN.md)** for the full design and rationale.

## Pipeline

```
AnthropicBackend (agents/)  ─▶  run_eval  ─▶  Trajectories
   teacher, <think>…</think><guess>…             │
                                                 ▼
 generate.py ─▶ data/raw/<game>.jsonl  ─▶  dataset.py (keep solved + explode per move)
                                                 │
                                                 ▼
                                      data/sft/<game>.jsonl
                                                 │
                                           push.py ─▶ Hub
```

- **Raw is the source of truth.** `data/raw/<game>.jsonl` holds full `Trajectory` records
  (format-neutral). SFT shaping happens later, so changing the training format never
  requires re-running (expensive) Claude rollouts.
- **One sample per move, loss on the completion only.** Each `Turn` already stores the
  exact prompt the agent built (`turn.messages`), so a sample is
  `{messages: turn.messages, completion: turn.response}` — byte-identical to inference.
- **Quality gate = solved episodes only** (`final.status == "won"`).

## Usage

```bash
# 1. teacher rollouts -> raw trajectories (uses ANTHROPIC_API_KEY, TEACHER_MODEL from .env)
uv run --package distillation python -m distillation.run generate --game wordle --n 500 --mode train

# 2. filter solved + explode into per-move SFT samples
uv run --package distillation python -m distillation.run build --game wordle

# 3. combine all games + push (uses HF_HUB_REPO_ID, HF_TOKEN); try --dry-run first
uv run --package distillation python -m distillation.run push --dry-run
uv run --package distillation python -m distillation.run push
```

## Adding a game

A new game's teacher data needs **only** a `GameSpec` entry in
[`registry.py`](registry.py) (its agent already lives in `agents/<game>/`). `generate.py`,
`dataset.py`, and `push.py` are game-agnostic and don't change.

## Files

| File | Role |
|------|------|
| [`config.py`](config.py) | `DistillConfig.from_env()` — teacher model/effort, Hub repo id, data paths |
| [`registry.py`](registry.py) | `GAMES` — per-game wiring (the only place a new game is added) |
| [`generate.py`](generate.py) | teacher rollouts via `run_eval` → raw `Trajectory` JSONL |
| [`dataset.py`](dataset.py) | filter solved + explode → per-move SFT JSONL |
| [`push.py`](push.py) | combine games → `datasets.Dataset` → `push_to_hub` |
| [`run.py`](run.py) | CLI: `generate` / `build` / `push` |

## Config (`.env`)

```bash
ANTHROPIC_API_KEY=...                 # teacher; read by the Anthropic SDK
TEACHER_MODEL=claude-opus-4-8
HF_HUB_REPO_ID=your-username/pixelpolicy-distill
HF_TOKEN=...                          # read by huggingface_hub for push_to_hub
```
