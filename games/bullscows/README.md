# Bulls & Cows

Game **#11** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **multi-turn** environment
in the style of [Wordle](../wordle/README.md): `reset()` hides a secret number, then each `guess`
returns **count** feedback (bulls + cows) and the model refines its next guess until it cracks the
number. Like [codebreaker](../codebreaker/README.md) it teaches the **core Wordle loop** — parse
several turns of feedback and adjust — but the feedback is a different *representation*: aggregate
counts rather than per-position tiles, so the deduction skill is decoupled from positional cues.

Same architecture as Wordle: a **pure core** that a human can play and an RL trainer can step
in-process by the thousands, through the same code path, so feedback and rules never drift. The
env knows nothing about models, tokens, or rewards.

## The game

The secret is **4 distinct digits** (each guess must also be 4 distinct digits, 0–9). You have
**10 rounds** (`max_rounds`). After each guess you are told two **counts**:

```
bulls = digits that are correct AND in the right position
cows  = digits that are in the code but in the wrong position
```

There are no per-position tiles — just the two totals. Win when `bulls == 4`; lose if 10 rounds
pass without it.

## Status convention

Mirrors Wordle so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → still guessing
"won"          → bulls == 4                             (the solved/"good" status)
"lost"         → 10 rounds used without cracking the number
```

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-bullscows python -m games.bullscows.play

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-bullscows uvicorn games.bullscows.server:app
uv run --package game-bullscows python -m games.bullscows.play --http http://127.0.0.1:8000

# tests (core bull/cow counting, solver, render, Local/HTTP parity) — 16 passing
uv run --package game-bullscows pytest games/bullscows/tests/ -q
```

> No shared vocab asset needed — secrets are distinct-digit strings, so the package has **no vocab
> or corpus dependency** (unlike the word games).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "n_digits": 4,
  "max_rounds": 10,
  "current_round": 2,
  "status": "in_progress",          // "in_progress" | "won" | "lost"
  "rounds": [
    { "guess": "1234", "bulls": 1, "cows": 2, "error": null },
    { "guess": "1123", "bulls": 0, "cows": 0, "error": "..." }
  ],
  "secret": null                    // revealed only once status != in_progress
}
```

- A valid round carries `bulls`/`cows` counts and `error == null`. An **invalid guess** (not 4
  distinct digits) has an `error` and still consumes a round — the same "invalid costs a round"
  contract as Wordle.
- `secret` is the reveal channel and stays `null` while the game is in progress.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"1234"?}` | New game; `word` pins the secret. |
| `POST` | `/guess` | `{"game_id":"...","guess":"1234"}` | Returns updated `GameState`. Invalid guesses return **200** with an `error` round. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` guessing after the game ended.

## The teacher solver

The SFT "teacher" is `BullsCowsSolver` — a deliberately **unbiased** solver: it opens with a
**random** code, then on every later turn guesses a code drawn **uniformly at random from the set
still consistent** with all bulls/cows feedback so far. No fixed opening and no ordering bias — a
deterministic solver would teach a biased policy.

## Files

```
games/bullscows/
├── game.py     # pure core: BullsCowsGame, BullsCowsBank, BullsCowsSolver, compute_feedback, models
├── render.py   # render_observation (the prompt) + render_round (per-round feedback line)
├── client.py   # BullsCowsClient Protocol + Local/HTTP transports (verb: guess)
├── server.py   # async FastAPI: /reset /guess /state
├── play.py     # terminal play (in-process or --http)
└── tests/      # test_game / test_render / test_client
```

## Training data

Bulls & Cows is **programmatic** but **multi-turn**, so its SFT data is produced by the
programmatic multi-turn generator in
[`distillation/programmatic.py --game bullscows`](../../distillation/README.md) — it replays
`BullsCowsSolver` through the same `agents/rollout.py::run_episode` Wordle uses, emitting **one
SFT row per turn**, at **$0** and with no Claude. The built default caps at **10,000 rows**
(≈1,823 episodes, ~5.5 turns/episode) via `--max-rows 10000`; `--max-rows` caps **at an episode
boundary** so the round distribution stays unbiased. See that module for the generator and the
unified dataset schema.

Each completion begins with a short, **programmatically-generated worded rationale** that recaps
the bull/cow clues, then the `<guess>` tag:

```
From the clues so far (0932 → 0 bulls, 1 cow), 1440 numbers still fit every count; 2158 is one of them, so I'll try it.
<guess>2158</guess>
```

The rationale is **always true** — `BullsCowsSolver.move` derives it (`_reason`) from the same
counts and consistent-number set it draws the guess from, and every generated row is self-checked
(the episode must reach `won`), so the dataset never teaches false reasoning. It is
**templated/programmatic**, not chain-of-thought: it is **not** wrapped in `<think>`, so the row's
`has_think` stays `False`. Because the rationale lives only in the completion, `build_messages`
replays only the bare `<guess>NNNN</guess>` of prior turns when constructing the next prompt — the
rationale is dropped on replay (exactly as Wordle strips `<think>`), so the reasoning is a training
target without growing the context (prompt + completion stays ~400 tokens, well under 4k). The
agent system prompt now asks the model to briefly reason from the clues before the final `<guess>`
tag.
