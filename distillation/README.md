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
- **Quality gate = solved episodes only** — `status == good_status` (`"correct"` for the
  single-turn games, `"won"` for the multi-turn deduction games codebreaker/bullscows). The
  programmatic teachers pass by construction. **Wordle is the exception**: its `valid` flag is
  re-derived from **format compliance** (does the move carry a `<think>` block) rather than the
  win/loss outcome — see the `valid` row in the schema table below.

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
| `game_name` / `game_no` | `"wordle"` / `0`, `"charcount"` / `1`, `"validity"` / `2`, `"anagram"` / `3`, `"endstart"` / `4`, `"rhyme"` / `5`, `"crossword"` / `6`, `"charset"` / `7`, `"mistakeid"` / `8`, `"tower"` / `9`, `"codebreaker"` / `10`, `"bullscows"` / `11`, `"consistency"` / `12` (numbering per [`games/DATA_SOURCING.md`](../games/DATA_SOURCING.md); all 13 built) |
| `round` | move number within the episode (single-turn games are always `1`; multi-turn games — Wordle, codebreaker, bullscows — increment per turn, one row per move) |
| `valid` | the rejection gate. **Single-turn games** gate on **correctness** (`status == good_status` — `correct`). **Wordle** gates on **format**: `valid = has_think` (the move carries a `<think>` block), regardless of win/loss — re-derived in `push.py`'s `load_rows`, so a well-formed reasoned move counts even from a lost game and a move with no `<think>` is dropped even from a won one. |
| `target` / `system` / `messages` / `completion` | the answer, prompt, and reply (byte-identical to inference) |
| `completion_no_think` / `has_think` | reply with `<think>` stripped + a flag |
| `episode` | per-run episode index |

`push.py` upgrades any *legacy* row (the original Wordle SFT, whose `game` field was the
episode index) on load, so old and new files combine without re-running rollouts.

## Two producers

- **Claude-distilled** (reasoning games — Wordle, anagram, crossword, mistakeid): [`batch_play.py`](batch_play.py) →
  raw `*_raw.json` + SFT `*_sft.jsonl`. Raw is the source of truth; re-shape SFT anytime with
  [`reexport.py`](reexport.py) (no Claude re-run).
- **Programmatic** (no-reasoning games — charcount, validity, endstart, rhyme, charset, tower,
  consistency, and the multi-turn codebreaker / bullscows): [`programmatic.py`](programmatic.py)
  formats the env's gold answer straight into the completion — zero API cost, fully reproducible.
  Pick the game with `--game` (choices:
  `charcount|validity|rhyme|charset|tower|endstart|codebreaker|bullscows|consistency`).

```bash
# programmatic games: generate SFT rows (seeded, self-checked). --game selects the game.
uv run --package distillation python -m distillation.programmatic --game charcount  # 14k: >=4k Wordle-vocab + rest WordNet (len 3-20)
uv run --package distillation python -m distillation.programmatic --game validity   # 13,254: 6,627 valid + 6,627 invalid (50/50)
uv run --package distillation python -m distillation.programmatic --game endstart   # 6k: MCQ, 1 matching candidate + 4 distractors (shuffled)
uv run --package distillation python -m distillation.programmatic --game rhyme      # 10k: 5k MCQ + 5k free
uv run --package distillation python -m distillation.programmatic --game charset    # 12k: 2-4 words each (1 five-letter Wordle word + non-five-letter words)
uv run --package distillation python -m distillation.programmatic --game tower      # 5k: deduction puzzles (~3,343 single-solution + ~1,657 two-solution)
uv run --package distillation python -m distillation.programmatic --game consistency # 10k: 5k yes + 5k no (reuses Wordle's scorer)

# programmatic MULTI-TURN games: an unbiased solver (random opening + a uniformly random
# consistent code) is replayed via agents/rollout.py::run_episode, one SFT row per turn ($0, no
# Claude). --max-rows caps output AT AN EPISODE BOUNDARY (whole episodes kept), so the round
# distribution stays unbiased (never "only first/last turns").
uv run --package distillation python -m distillation.programmatic --game codebreaker --episodes 5000 --max-rows 10000  # 10k rows (~2,726 episodes, ~3.7 turns/ep)
uv run --package distillation python -m distillation.programmatic --game bullscows   --max-rows 10000                  # 10k rows (~1,823 episodes, ~5.5 turns/ep)

# Claude-distilled reasoning games (anagram, crossword, mistakeid): single-turn Batch API +
# rejection, so require_think keeps only correct traces that carry a <think> block (see below).
uv run --package distillation python -m distillation.batch_play \
  --game anagram --episodes 1000 --model claude-sonnet-4-6 --effort high
uv run --package distillation python -m distillation.batch_play \
  --game crossword --episodes 1500 --model claude-sonnet-4-6 --effort high
# mistakeid reads the committed games/mistakeid/challenges.jsonl (165 mistake + 1,498 clean boards);
# 330 = a balanced 165/165 mistake/clean mix. --effort max (xhigh is NOT valid for this model).
uv run --package distillation python -m distillation.batch_play \
  --game mistakeid --episodes 330 --model claude-sonnet-4-6 --effort max

# re-shape existing Claude raw dumps into the current SFT schema (no Claude)
uv run --package distillation python -m distillation.reexport distillation/data/batch_*_raw.json

# combine every game + push. --overwrite wipes the Hub repo's old shards/card first
# (required after a schema change); --dry-run reports stats without a token.
uv run --package distillation python -m distillation.push --dry-run
uv run --package distillation python -m distillation.push --overwrite
```

> **`require_think`** — a `GameSpec` flag for the *reasoning* distilled games (anagram, crossword,
> mistakeid), distilled at high adaptive-thinking effort (mistakeid at `max`). A solved trace with
> no `<think>` block is unusable as a reasoning SFT target, so `batch_play.py` keeps a trace only
> when it is correct **and** carries a `<think>` block. (Wordle leaves the spec flag `False`;
> instead `push.py` re-derives Wordle's `valid` directly from `has_think`, so its gate is purely
> format compliance — a `<think>` block — independent of whether the game was won.)

Example combined dataset:
**[saketh-chervu/word-games-distillation](https://huggingface.co/datasets/saketh-chervu/word-games-distillation)**
— **96,162 rows** (86,545 train / 9,617 test) across **13 games**, **95,520 valid**: 3,078 Wordle
(2,602 valid) + 14,000 charcount + 13,254 validity + 1,000 anagram (932 valid) + 10,000 rhyme +
1,500 crossword (1,415 valid) + 12,000 charset + 330 mistakeid (317 valid) + 5,000 tower + 6,000
endstart + 10,000 codebreaker + 10,000 bullscows + 10,000 consistency. The **multi-turn share**
(Wordle + codebreaker + bullscows) is **~24%** (up from ~5% with Wordle alone). The `valid` flag is
the rejection gate (the dataset keeps `valid=False` rows too; the trainer filters `valid==True`).
For the correctness games it means the answer matched; for **Wordle** it means **format compliance**
(`has_think`) — so Wordle's 3,078 rows include 2,602 with a `<think>` block (476 no-think moves
dropped), regardless of win/loss. The programmatic games (including codebreaker/bullscows) are all
valid by construction.

## Adding a game

A new game's data needs **only** a `GameSpec` entry in [`registry.py`](registry.py) (its agent
already lives in `agents/<game>/`) plus its `game_no` in `GAME_NUMBERS`. `batch_play.py`,
`schema.py`, and `push.py` are game-agnostic and don't change. Claude-distilled games go through
`batch_play.py` (set `require_think=True` on the spec for reasoning games like anagram, crossword,
and mistakeid); programmatic games add a small `--game` branch in `programmatic.py` like charcount /
validity / endstart / rhyme / charset / tower / consistency. A **programmatic multi-turn** game
(codebreaker, bullscows) instead supplies a `*Solver` and is driven by `_play_multiturn`
(`agents/rollout.py::run_episode` with the solver as the `generate` callback), emitting one SFT
row per turn; its `GameSpec` sets `max_rounds > 1` and `good_status="won"`, and `--max-rows` caps
the output at an episode boundary. `push.py`'s `DEFAULT_INPUTS` lists the per-game SFT files
(mapping `crossword_sft` → `(crossword, 6)`, `charset_sft` → `(charset, 7)`, `mistakeid_sft` →
`(mistakeid, 8)`, etc.) and skips any that are missing, so you can push a subset while a game's
data is still being generated.

## Files

| File | Role |
|------|------|
| [`schema.py`](schema.py) | the unified SFT row schema; `sft_row` + `normalize_legacy` |
| [`registry.py`](registry.py) | `GAMES` + `GAME_NUMBERS` — per-game wiring (the only place a new game is added) |
| [`batch_play.py`](batch_play.py) | Claude Batch-API rollouts → raw + SFT (distilled games: wordle, anagram, crossword, mistakeid); honors `require_think` |
| [`programmatic.py`](programmatic.py) | no-Claude "synthetic teacher" → SFT (programmatic games: charcount, validity, endstart, rhyme, charset, tower, consistency; plus the multi-turn codebreaker/bullscows via `_play_multiturn` + `--max-rows`) |
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
