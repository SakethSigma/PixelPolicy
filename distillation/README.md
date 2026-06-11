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

## Unified SFT schema (one row shape for every game)

Every row — Wordle (Claude-distilled) and the single-turn word-skill games (programmatic) —
shares the same columns, defined once in [`schema.py`](schema.py):

| column | meaning |
|--------|---------|
| `game_name` / `game_no` | `"wordle"` / `0`, `"charcount"` / `1`, … (numbering per [`games/DATA_SOURCING.md`](../games/DATA_SOURCING.md)) |
| `round` | move number within the episode (single-turn games are always `1`) |
| `valid` | passed the solved gate (`status == good_status`): `won` for Wordle, `correct` for single-turn |
| `target` / `system` / `messages` / `completion` | the answer, prompt, and reply (byte-identical to inference) |
| `completion_no_think` / `has_think` | reply with `<think>` stripped + a flag |
| `episode` | per-run episode index |

`push.py` upgrades any *legacy* row (the original Wordle SFT, whose `game` field was the
episode index) on load, so old and new files combine without re-running rollouts.

## Two producers

- **Claude-distilled** (reasoning games, e.g. Wordle): [`batch_play.py`](batch_play.py) →
  raw `*_raw.json` + SFT `*_sft.jsonl`. Raw is the source of truth; re-shape SFT anytime with
  [`reexport.py`](reexport.py) (no Claude re-run).
- **Programmatic** (no-reasoning games, e.g. charcount): [`programmatic.py`](programmatic.py)
  formats the env's gold answer straight into the completion — zero API cost, fully reproducible.

```bash
# programmatic game (charcount): generate SFT rows (seeded, self-checked).
# default = 14k rows: >=4k Wordle-vocab words + the rest from WordNet (lengths 3-20).
uv run --package distillation python -m distillation.programmatic

# re-shape existing Claude raw dumps into the current SFT schema (no Claude)
uv run --package distillation python -m distillation.reexport distillation/data/batch_*_raw.json

# combine every game + push. --overwrite wipes the Hub repo's old shards/card first
# (required after a schema change); --dry-run reports stats without a token.
uv run --package distillation python -m distillation.push --dry-run
uv run --package distillation python -m distillation.push --overwrite
```

Example combined dataset:
**[saketh-chervu/word-games-distillation](https://huggingface.co/datasets/saketh-chervu/word-games-distillation)**
— 17,078 rows (3,078 Wordle + 14,000 charcount).

## Adding a game

A new game's data needs **only** a `GameSpec` entry in [`registry.py`](registry.py) (its agent
already lives in `agents/<game>/`) plus its `game_no` in `GAME_NUMBERS`. `batch_play.py`,
`programmatic.py`, `schema.py`, and `push.py` are game-agnostic and don't change. Reasoning
games go through `batch_play.py`; programmatic games add a small loop like charcount's.

## Files

| File | Role |
|------|------|
| [`schema.py`](schema.py) | the unified SFT row schema; `sft_row` + `normalize_legacy` |
| [`registry.py`](registry.py) | `GAMES` + `GAME_NUMBERS` — per-game wiring (the only place a new game is added) |
| [`batch_play.py`](batch_play.py) | lockstep Claude Batch-API rollouts → raw + SFT (reasoning games) |
| [`programmatic.py`](programmatic.py) | no-Claude "synthetic teacher" → SFT (programmatic games, e.g. charcount) |
| [`reexport.py`](reexport.py) | re-shape a raw dump → current SFT schema (no Claude re-run) |
| [`push.py`](push.py) | combine games → `datasets.Dataset` → `push_to_hub` (`--overwrite` for schema changes) |
| [`cost_probe.py`](cost_probe.py) | measure teacher cost at a given reasoning effort |

## Config (`.env`)

```bash
ANTHROPIC_API_KEY=...                 # teacher; read by the Anthropic SDK
TEACHER_MODEL=claude-opus-4-8
HF_HUB_REPO_ID=your-username/pixelpolicy-distill
HF_TOKEN=...                          # read by huggingface_hub for push_to_hub
```
