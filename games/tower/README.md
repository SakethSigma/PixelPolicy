# Tower deduction

Word-skill game **#9** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses one placement puzzle, `step(answer)` scores the listed placements
and ends the episode. It teaches the model to **reason from Wordle-style ✓/x feedback** — the same
deductive skill Wordle rewards, distilled into one self-contained challenge.

Same architecture as [Wordle](../wordle/README.md): a **pure core** that a human can play and an
RL trainer can step in-process by the thousands, through the same code path, so feedback and
rules never drift. The env knows nothing about models, tokens, or rewards.

## The puzzle

A tower has **3 floors** (1 = bottom, 3 = top); each floor has **two rooms, Left and Right**.
Three people each live in a different room, and **no two people share a floor** (the floor
assignment is a bijection). You are shown a *proposed* placement and, for each person, two ✓/x
flags — whether their **floor** is correct and whether their **room** is correct (the same ✓/x
symbols Wordle uses). List **every** placement consistent with the feedback.

The logic is provable and the answer set is tiny:

- **Rooms never branch.** Each room is one of two; a wrong-room flag means the person is in the
  *other* room of whatever floor they end up on. So every person's room is fixed.
- **Floors fix the bijection up to derangements.** The consistent floor assignments are the
  permutations of `(1,2,3)` whose agreement with the proposed floors matches the flags exactly.
  If **any** floor flag is ✓ there is exactly **1** solution; if **all three** floors are wrong
  there are exactly **2** (the two derangements of 3). The answer set is **always size 1 or 2 —
  never more.**

The whole distinct logic space is only **1,920** structures, so surface variety comes from a pool
of random first names (60 names), not from harder logic.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → challenge posed, no answer yet
"correct"      → the listed placements exactly match the consistent set   (the solved/"good" status)
"incorrect"    → wrong / missing / extra placement, or unparseable
```

Ground truth is pure Python (no corpora, no `<think>` requested): enumerate the 6 floor
permutations, keep those whose ✓/x pattern matches, and flip each mismatched room.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-tower python -m games.tower.play

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-tower uvicorn games.tower.server:app
uv run --package game-tower python -m games.tower.play --http http://127.0.0.1:8000

# tests (core solve/score, render round-trip, Local/HTTP parity) — 20 passing
pytest games/tower/tests/ -q
```

> No shared vocab asset needed — challenges are generated from pure Python and a built-in name
> pool, unlike the vocabulary-backed word games.

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "names": ["Alice", "Bob", "Carol"],
  "shown_floors": [2, 1, 3],              // the proposed floor per person
  "shown_rooms": [0, 1, 0],               // 0 = Left, 1 = Right
  "floor_ok": [false, true, false],       // per-person "floor correct?" flag
  "room_ok": [true, false, true],         // per-person "room correct?" flag
  "status": "correct",                    // "in_progress" | "correct" | "incorrect"
  "submitted": "solution 1:\n...",        // the raw answer step() scored (null before step)
  "solutions": [                          // every consistent placement; revealed only once status != in_progress
    [{"name": "Alice", "floor": 3, "room": "Right"}, ...]
  ]
}
```

The canonical `<answer>` block (what the synthetic teacher writes and a human types) is one
numbered `solution N:` block per consistent placement, one person per line:

```
solution 1:
Alice: floor 3, Right
Bob: floor 1, Left
Carol: floor 2, Left
```

`game.py::parse_answer` reads this format back tolerantly (any order, `Left`/`Right`/`L`/`R`,
any case, optional `solution N:` headers); the scored set of placements must **exactly equal**
the consistent set — extra, missing, or wrong placements (or an unparseable answer) score
`incorrect`, the same "malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"Alice,Bob,Carol;2L,1R,3L;01,10,00"?}` | New episode; `word` pins an encoded `names;shown;feedback` target. |
| `POST` | `/step` | `{"game_id":"...","answer":"solution 1:\n..."}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Challenge generation

`TowerBank` builds each challenge in pure Python: pick three distinct random names, a random
*shown* placement, and a random *true* placement; the per-person flags are derived by comparing
them. `sample_targets(n, mode, rng)` returns `n` distinct encoded targets — and because the name
pool decouples surface form from the 1,920 distinct logic structures, that surface space is far
larger. There is no committed split file and no vocabulary dependency; `mode` is accepted only
for parity with the other games.

## Files

```
games/tower/
├── game.py     # pure core: TowerGame, TowerBank, solve/encode/decode/parse, models, NAMES pool
├── render.py   # render_observation (the prompt) + render_solutions (canonical answer block)
├── client.py   # TowerClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http)
└── tests/      # test_game / test_render / test_client
```

## Training data

Tower is **programmatic** (the consistent set is cheap and exact, no reasoning wanted), so its SFT
data is produced by [`distillation/programmatic.py --game tower`](../../distillation/README.md) —
not Claude. The built default emits **5,000 rows**, all valid by construction: ~3,343
single-solution + ~1,657 two-solution (the ~1/3 all-floors-wrong derangement rate). See that
module for the generator and the unified dataset schema.
