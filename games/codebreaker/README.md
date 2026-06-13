# Codebreaker (Mastermind)

Game **#10** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **multi-turn** environment
in the style of [Wordle](../wordle/README.md): `reset()` hides a secret code, then each `guess`
returns per-position feedback and the model refines its next guess until the code is cracked (or
it runs out of rounds). It teaches the **core Wordle loop** — parse several turns of feedback and
adjust — on a non-vocabulary symbol space, isolating the deduction skill from word knowledge.

Same architecture as Wordle: a **pure core** that a human can play and an RL trainer can step
in-process by the thousands, through the same code path, so feedback and rules never drift. The
env knows nothing about models, tokens, or rewards.

## The game

The secret is **4 slots**, each one of **6 symbols** (`A B C D E F`); **symbols can repeat**. You
have **12 rounds** (`max_rounds`). After each guess every slot is scored with the same symbols
Wordle uses:

```
✓  right symbol in the right slot
-  right symbol but in the wrong slot
x  the symbol is not in the code (or all its copies are already accounted for)
```

Feedback uses **Wordle's exact two-pass duplicate rule** (`compute_feedback`): greens are marked
first and consume a copy of the symbol, then remaining copies feed the yellows — so a repeated
guess symbol against fewer copies in the secret correctly shows some `x`. Win when all four slots
are `✓`; lose if 12 rounds pass without cracking it.

## Status convention

Mirrors Wordle so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → still guessing
"won"          → all four slots ✓                       (the solved/"good" status)
"lost"         → 12 rounds used without cracking the code
```

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-codebreaker python -m games.codebreaker.play

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-codebreaker uvicorn games.codebreaker.server:app
uv run --package game-codebreaker python -m games.codebreaker.play --http http://127.0.0.1:8000

# tests (core feedback/duplicate rule, solver, render, Local/HTTP parity) — 16 passing
uv run --package game-codebreaker pytest games/codebreaker/tests/ -q
```

> No shared vocab asset needed — secrets are pure symbol strings, so the package has **no vocab
> or corpus dependency** (unlike the word games).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "code_length": 4,
  "symbols": "ABCDEF",
  "max_rounds": 12,
  "current_round": 2,
  "status": "in_progress",          // "in_progress" | "won" | "lost"
  "rounds": [
    { "guess": "ACEF", "feedback": "✓-xx", "error": null },
    { "guess": "ABCD", "feedback": "",     "error": "invalid" }
  ],
  "secret": null                    // revealed only once status != in_progress
}
```

- A valid round has a 4-character `feedback` string and `error == null`. An **invalid guess**
  (wrong length, or a symbol outside A–F) has empty `feedback` and `error == "invalid"`, and still
  consumes a round — the same "invalid costs a round" contract as Wordle.
- `secret` is the reveal channel and stays `null` while the game is in progress.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"ACEF"?}` | New game; `word` pins the secret. |
| `POST` | `/guess` | `{"game_id":"...","guess":"ACEF"}` | Returns updated `GameState`. Invalid guesses return **200** with an `error` round. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` guessing after the game ended.

## The teacher solver

The SFT "teacher" is `CodebreakerSolver` — a deliberately **unbiased** solver: it opens with a
**random** code, then on every later turn guesses a code drawn **uniformly at random from the set
still consistent** with all feedback so far (maintaining the candidate set incrementally). There
is **no fixed opening and no symbol-order bias** — a deterministic or ordered solver would teach a
biased policy, which is the opposite of what these games are for.

## Files

```
games/codebreaker/
├── game.py     # pure core: CodebreakerGame, CodebreakerBank, CodebreakerSolver, compute_feedback, models
├── render.py   # render_observation (the prompt) + render_round (per-round feedback line)
├── client.py   # CodebreakerClient Protocol + Local/HTTP transports (verb: guess)
├── server.py   # async FastAPI: /reset /guess /state
├── play.py     # terminal play (in-process or --http)
└── tests/      # test_game / test_render / test_client
```

## Training data

Codebreaker is **programmatic** but **multi-turn**, so its SFT data is produced by the
programmatic multi-turn generator in
[`distillation/programmatic.py --game codebreaker`](../../distillation/README.md) — it replays
`CodebreakerSolver` through the same `agents/rollout.py::run_episode` Wordle uses, emitting **one
SFT row per turn**, at **$0** and with no Claude. The built default caps at **10,000 rows**
(≈2,726 episodes, ~3.7 turns/episode) via `--episodes 5000 --max-rows 10000`; `--max-rows` caps
**at an episode boundary** so the round distribution stays unbiased. See that module for the
generator and the unified dataset schema.

Each completion begins with a short, **programmatically-generated worded rationale** that recaps
the deductions, then the `<guess>` tag:

```
Clues so far — fixed: slot 1=A; in the code but misplaced: none; not in the code: B, D. 64 codes still fit; AAEF is one of them, so I'll try it.
<guess>AAEF</guess>
```

The rationale is **always true** — `CodebreakerSolver.move` derives it (`_reason`) from the same
feedback and consistent-code set it draws the guess from, and every generated row is self-checked
(the episode must reach `won`), so the dataset never teaches false reasoning. It is
**templated/programmatic**, not chain-of-thought: it is **not** wrapped in `<think>`, so the row's
`has_think` stays `False`. Because the rationale lives only in the completion, `build_messages`
replays only the bare `<guess>CODE</guess>` of prior turns when constructing the next prompt — the
rationale is dropped on replay (exactly as Wordle strips `<think>`), so the reasoning is a training
target without growing the context (prompt + completion stays ~400 tokens, well under 4k). The
agent system prompt now asks the model to briefly note what the clues tell it before the final
`<guess>` tag.
