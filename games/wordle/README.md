# Wordle

A self-contained Wordle environment for PixelPolicy: a **pure game core** that a human
can play in the terminal and that an RL trainer can step in-process by the thousands —
through the **same** code path, so feedback and rules can never drift between them.

The environment knows nothing about models, tokens, or rewards. Reward is an RL-specific
concern and lives in `training/` (see [Training integration](#training-integration)).

---

## Quick start

Everything runs through the workspace's shared `uv` env. The package is `game-wordle`.

### Play in the terminal

```bash
# In-process — no server needed. Colored tiles via `rich` (the [tui] extra).
uv run --package game-wordle python -m games.wordle.play

# Pin the answer, or pick the val pool:
uv run --package game-wordle python -m games.wordle.play --word crane
uv run --package game-wordle python -m games.wordle.play --mode val
```

You get 6 guesses. Per-letter feedback: `✓` right letter & spot, `-` right letter wrong
spot, `x` not in the word. Type `q` (or Ctrl-D) to quit.

> An invalid guess **costs a round** and tells you why (`inadequate length` /
> `out of vocabulary`) — there's no free "try again". This is intentional and identical
> for a human and a model (see [Invalid guesses](#invalid-guesses)).

### Run the HTTP server

```bash
uv run --package game-wordle uvicorn games.wordle.server:app --reload
# then, in another shell, play the SAME game over HTTP:
uv run --package game-wordle python -m games.wordle.play --http http://127.0.0.1:8000
```

Run **single process** (no `--workers`): game state lives in an in-memory dict keyed by
`game_id`, and multiple workers would split that state. Concurrency comes from the async
event loop — every handler is microsecond CPU work, so one process is not a bottleneck.

### Run the tests

```bash
uv run --package game-wordle pytest games/wordle/tests/ -q   # 91 tests
```

---

## Architecture

```
            ┌─────────────────────────── games/wordle ───────────────────────────┐
            │                                                                     │
 human ───► play.py ──┐                                  ┌── server.py (FastAPI) ─┤◄── HTTP ── eval / inference
            (rich UI) │                                  │   /reset /guess /state │
                      ▼                                  ▼                        │
            client.py: WordleClient (Protocol)                                    │
              ├── LocalWordleClient  ─────────────┐      ▲                        │
 trainer ───► └── HTTPWordleClient ──► HTTP ──────┼──────┘                        │
 (in-proc)                                        ▼                               │
                                       game.py  (PURE CORE — no web, no reward)   │
                                         WordleGame · WordBank · compute_feedback │
                                         GameState · RoundResult · LetterFeedback │
            └─────────────────────────────────────────────────────────────────────┘
```

| File | Responsibility |
|------|----------------|
| [`game.py`](game.py) | **Pure core.** `WordleGame` (one episode), `WordBank` (word pools + validation), `compute_feedback`, and the Pydantic models. No FastAPI, no reward. The single source of truth. |
| [`server.py`](server.py) | Thin async FastAPI wrapper: `POST /reset`, `POST /guess`, `GET /state/{id}`. |
| [`client.py`](client.py) | `WordleClient` Protocol + `LocalWordleClient` (in-process) and `HTTPWordleClient` (httpx). One interface, two transports. |
| [`render.py`](render.py) | Dependency-free text board — the observation a human *and* an LLM read. |
| [`play.py`](play.py) | Terminal UI (`rich`), layered over the same `render.py` layout. |
| [`split.py`](split.py) | CLI to regenerate the deterministic train/val split (run deliberately). |

**Why two transports behind one Protocol:** training steps envs in-process for free
(`LocalWordleClient`); eval/inference/remote play go over HTTP (`HTTPWordleClient`). Both
delegate to the same `game.py`, so a guess behaves identically either way — proven by a
field-for-field parity test in `tests/test_client.py`.

---

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "max_rounds": 6,
  "current_round": 2,
  "status": "in_progress",          // "in_progress" | "won" | "lost"
  "rounds": [
    { "guess": "CRANE", "feedback": ["x","x","-","x","-"], "error": null },
    { "guess": "ZZZZZ", "feedback": [],                    "error": "out of vocabulary" }
  ],
  "target": null                    // revealed (uppercased) only once status != in_progress
}
```

- `feedback` symbols come from `LetterFeedback`: `CORRECT="✓"`, `WRONG_POS="-"`, `WRONG_LETTER="x"`.
- A valid round has 5 feedback entries and `error == null`. An **invalid round has empty
  `feedback` and an `error`** (`InvalidReason`: `"inadequate length"` or `"out of vocabulary"`).
- `target` is the reward-time reveal channel and stays `null` while the game is in progress.

### Feedback & duplicate letters

Two-pass scoring so duplicates are correct. Greens are marked first and consume a copy from
the target; remaining copies then feed yellows. Example — guess `PUPPY` vs target `APPLE`:
the middle `P` is green, the first `P` is yellow (one `P` left), the rest gray → `- x ✓ x x`.

### Invalid guesses

An invalid guess **consumes a round** and reports a reason, with **no per-letter feedback**
(so a non-word can't be used to probe letters for free). This rule lives in the pure core
(`WordleGame.guess`), not in the server or client, so both transports agree. Validation is
tiered: structural (length/alpha) first, then vocabulary membership.

A human and a model therefore face the exact same constraint — submitting `zzzzz` burns a
guess for both.

---

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode": "train"\|"val", "word": "apple"?}` | New game. `word` pins the target (debug/eval/GRPO). Returns empty `GameState`. |
| `POST` | `/guess` | `{"game_id": "...", "guess": "crane"}` | Returns updated `GameState`. Invalid guesses return **200** with an `error` round (they're not HTTP errors). |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

**Status codes:** `404` unknown `game_id`; `400` guessing after the game ended, or a
malformed pinned `word` on `/reset`; `422` a bad `mode` (Pydantic). Note that a bad *guess*
is **not** a 400 — it's a consumed round.

---

## Client usage

Both clients implement the same three-verb `WordleClient` Protocol and return `GameState`.
Both are context managers (the HTTP one owns its `httpx.Client`; the local one is a no-op),
so trainer/eval code can be written transport-agnostic.

```python
from games.wordle.client import LocalWordleClient, HTTPWordleClient
from games.wordle.game import WordBank
from games.wordle.render import render_observation

# In-process (training, eval harnesses, notebooks) — zero network.
bank = WordBank()                       # load the committed split once
env = LocalWordleClient(bank)
env.reset(mode="train")                 # or word="crane" to pin
print(render_observation(env.state()))  # the text an LLM policy sees
state = env.guess("crane")

# Over HTTP (remote env, cross-language agents) — same methods, same GameState.
with HTTPWordleClient("http://127.0.0.1:8000") as env:
    env.reset(mode="val")
    state = env.guess("moist")
```

`reset()` again on a handle abandons the current game and starts a fresh one.

---

## Word pools & the deterministic split

The full allowed-guess list lives in [`words.txt`](words.txt) (12,972 words). It is split
once into committed artifacts — [`train_words.txt`](train_words.txt) (10,384) and
[`val_words.txt`](val_words.txt) (2,588) — which `WordBank` loads directly.

The split is **byte-for-byte deterministic**: `assign_pool(word) = sha256(word) % 1000 < 200
→ val else train`. Pool membership depends only on a word's own bytes — not list order, list
size, Python version, or `PYTHONHASHSEED`. So a given word is *always* train or *always*
val, on every machine, every run, for as long as the files are unchanged. This is what lets
you compare many models over a month on a fixed eval set.

The only runtime randomness is `WordBank.sample(mode)` choosing a fresh target per episode.

**Regenerating** (a deliberate act — it changes which words are eval words and invalidates
cross-run comparisons; only do it after curating `words.txt`):

```bash
uv run --package game-wordle python -m games.wordle.split
```

---

## Training integration

The trainer is the **client**; the env is pure. The server never produces training data —
during rollout the client already holds the prompt, generated `token_ids`, `logprobs`, and
the returned feedback, so "breaking the episode into per-turn samples" happens naturally
client-side. `game_id` is just a rollout handle, irrelevant to the gradient step.

### Reward boundary

Reward is **not** in this package. It's a pure function of the finished trajectory plus the
answer, planned for:

```
training/rewards/wordle.py::compute_reward(trajectory, target_word) -> float
```

Keeping it out of the env means recipes can swap shaping (terminal win/lose, fewer-guesses
bonus, per-turn information gain) without touching the environment. The trainer reads the
answer from `GameState.target`, which the env reveals only once the game ends.

### In-process rollout sketch (GRPO-style)

```python
from games.wordle.game import WordBank
from games.wordle.client import LocalWordleClient
from games.wordle.render import render_observation

bank = WordBank()                                    # load split once
targets = [bank.sample("train") for _ in range(N)]   # trainer draws the answers
envs = [LocalWordleClient(bank) for _ in range(N)]
for e, t in zip(envs, targets):
    e.reset(word=t)                                  # pin → a GRPO group can share a target

# step every live env until done:
#   obs = render_observation(e.state())  →  model.generate()  →  e.guess(g)
#   record (prompt, token_ids, logprobs, feedback) for that episode

# episode end:
#   trajectory = e.state().rounds + e.state().status
#   answer     = e.state().target          # revealed on completion
#   reward     = compute_reward(trajectory, answer)   # ← in training/, later
```

**Notes & footguns**
- **GRPO grouping:** pin G envs to one drawn word with `reset(word=...)` (or the
  `make_local_group(n, word=...)` helper) so rewards can be normalized within the group.
- **Pin a *real* word.** A pinned target that isn't in the vocabulary can never be legally
  guessed → guaranteed loss. Always draw pins from a pool (`bank.sample(...)`).
- **Abandoned games** (no win, `rounds < max_rounds`) leave `target == None`; treat
  "unfinished" as its own reward bucket — don't assume the answer is always available.
- **HTTP also works for training** (a localhost `/guess` is sub-ms and generation dominates
  wall-clock), but in-process is free, so prefer `LocalWordleClient` for rollouts.

---

## Extending / future development

- **Reward module** — implement `training/rewards/wordle.py::compute_reward(...)`; the env
  is ready (trajectory in `GameState.rounds`, answer in `GameState.target`).
- **Agent policy** — `agents/` (has `anthropic`) can wrap a model that consumes
  `render_observation(state)` and returns a 5-letter guess; drive it with any `WordleClient`.
- **Inference loop** — `inference/` (has `httpx`) can host the eval harness pointing
  `HTTPWordleClient` at a running server.
- **New games** — mirror this layout: a pure `game.py` core, a thin `server.py`, a uniform
  `client.py`, and a `render.py`. Keep reward out of the env.
- **Variants** — word length and round count are parameters (`WORD_LENGTH`,
  `max_rounds`); a different-length variant mainly needs its own word list + split.
- **Durability** — state is in-memory and lost on server restart (fine for rollouts; just
  `reset` again). Swap in SQLite/Redis only if long runs must survive restarts.

## File map

```
games/wordle/
├── game.py            # pure core: WordleGame, WordBank, compute_feedback, models
├── server.py          # async FastAPI: /reset /guess /state
├── client.py          # WordleClient Protocol + Local/HTTP transports
├── render.py          # dependency-free text board (human + LLM observation)
├── play.py            # terminal UI (rich), `python -m games.wordle.play`
├── split.py           # regenerate the deterministic split
├── words.txt          # full allowed-guess list (12,972)
├── train_words.txt    # committed split artifact (10,384)
├── val_words.txt      # committed split artifact (2,588)
└── tests/             # test_game / test_server / test_client / test_render / test_play
```
