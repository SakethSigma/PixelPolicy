# Agents — how to implement one

This is the reference for the `agents/` layer: what an agent *is* in PixelPolicy, how the
pieces fit, and how to **write an agent for a new game**. Driving an agent from a training
loop is a separate concern — see **[training_integration.md](training_integration.md)**.

> Hard rule (from the root README): **a new game must not require changes to agent code,
> and a new agent must not require changes to any game.** Everything game-specific is
> isolated into one small per-game subpackage; the rest of `agents/` is written once.

---

## Mental model: three seams

An agent playing a game is three jobs that talk to each other. We keep them as separate
seams so each can be swapped without touching the others:

| Seam | What it does | Game-aware? | Stateful? | Where |
|------|--------------|-------------|-----------|-------|
| **Env** | Game logic: `reset / guess / state → GameState` | yes (it *is* the game) | yes — source of truth | `games/<game>/` — already exists |
| **Backend** | `messages → text` (a model behind an API) | no | no | `agents/backend.py` |
| **Agent (adapter)** | Glue: build prompt from state; parse text → action | yes | no — pure functions | `agents/<game>/agent.py` |
| **Runner** | Drive the loop start→end; produce a `Trajectory` | no | only the Trajectory it returns | `agents/rollout.py` |

The only game-aware code you write is the **Agent adapter**. The Backend and Runner are
generic and never change per game or per model.

```
                 ┌──────────────── agents/ ────────────────┐
                 │                                          │
 GameState ──►   Agent.build_messages(state) ──► messages   │
 (from env)      │                                  │       │
                 │                                  ▼        │
                 │                            Backend.chat ──┼──► HTTP ──► vLLM server (OpenAI API)
                 │                                  │        │            (inference/server.py)
                 │                                  ▼        │
 env.guess(g) ◄──── Agent.parse_action(text) ◄── completion │
                 │                                          │
                 │   run_episode(...) drives this loop and  │
                 │   records a Trajectory; an Observer       │
                 │   renders it (demo) or not (headless).    │
                 └──────────────────────────────────────────┘
```

---

## Layout: generic core vs. per-game adapter

```
agents/
├── pyproject.toml
├── base.py          # GENERIC: GameAgent protocol; LLMBackend + Completion; Turn / Trajectory
├── backend.py       # GENERIC: OpenAICompatBackend (impl of LLMBackend) — inference only
├── rollout.py       # GENERIC: run_episode + run_eval + win_rate; Observers (Terminal/Null)
├── run.py           # GENERIC: .env config + CLI wiring
└── wordle/          # GAME-SPECIFIC: everything that knows Wordle lives here
    ├── __init__.py
    └── agent.py     # WordleAgent: system_prompt, build_messages, parse_action
```

**Rule of thumb:** if a file would change when you add a second game, it belongs in the
game subpackage (`agents/<game>/`); if it wouldn't, it stays flat. Adding a game =
add `agents/<game>/` and nothing else moves.

---

## State ownership (read this before writing anything)

Getting this wrong is how feedback drifts and rollouts become unreproducible. The rules:

- **The env owns game truth.** Target word, rounds, feedback, win/loss — only the env
  computes these. The agent *reads* `GameState`; it never decides whether a guess was
  correct or the game is won.
- **The agent is stateless.** `build_messages` and `parse_action` are pure functions of
  their inputs. The episode runs **multi-turn by default**, but the conversation is rebuilt
  each call from `(state, history)` that the rollout threads in — the agent stores nothing,
  so it stays restartable and reproducible (see [conversation framing](#conversation-framing)).
- **The rollout owns the `Trajectory`** — the per-episode record. That's the only thing
  that accumulates across turns.
- **We do not own tokens or logprobs.** Those belong to whoever runs the generation
  engine. At inference that's the backend's business; in training it's the RL library's
  (see the integration doc). Our `Completion` and `Trajectory` are **text + game data
  only**.

---

## Writing an agent (the only game-specific work)

Implement the `GameAgent` protocol. Two pure methods and a system prompt:

```python
# agents/base.py  (generic) — the contract every game adapter implements
class GameAgent(Protocol):
    system_prompt: str
    # history = the prior Turns of THIS episode, threaded in by the rollout (not stored here).
    def build_messages(self, state: GameState, history: list[Turn] = ()) -> list[dict]: ...
    def parse_action(self, text: str) -> str: ...
```

```python
# agents/wordle/agent.py  (game-specific)
from games.wordle.render import render_round       # the per-round feedback line — the same atom a human sees

class WordleAgent:
    system_prompt = (
        "You are playing Wordle (5 letters, 6 guesses).\n"
        "Feedback per letter: ✓ right spot, - wrong spot, x not in word.\n"
        "Think through the clues, then end your reply with '<guess>word</guess>' (a real 5-letter word)."
    )

    def build_messages(self, state, history=()):
        # Multi-turn by default: replay the conversation so the model sees its own prior
        # reasoning and gets incremental feedback. Pure — `history` is passed in, never stored.
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": "Make your first guess."},
        ]
        for turn in history:
            msgs.append({"role": "assistant", "content": turn.response})              # the model's prior reply
            msgs.append({"role": "user",      "content": render_round(turn.state.rounds[-1])})  # just the latest feedback
        return msgs
        # (The real WordleAgent strips each reply's <think>…</think> before replay — see
        #  agents/wordle/README.md — but the shape is exactly this.)

    def parse_action(self, text):
        # See "Parsing contract" below.
        ...
```

The message list is **prefix-stable**: each turn appends one `(assistant, user)` pair, so
turn *N*'s prompt is turn *N-1*'s plus the model's last reply and its feedback. That's the
exact shape an agentic RL trainer wants (each assistant turn is one action span for
token-level credit assignment — see the integration doc).

**Requirements for any agent:**
1. **Pure & stateless.** No I/O, no network, no hidden mutable state — `build_messages` is a
   pure function of `(state, history)`, and the rollout owns `history`. This is what lets the
   trainer import and call these exact functions (integration doc, Mode B2).
2. **Reuse the game's renderers** (`render_round` per turn, or `render_observation` for the
   full board) — so the feedback text the model trains on and is evaluated on is byte-identical
   to what a human sees, with no second copy to drift.
3. **Keep model/transport choices out.** The agent doesn't know if it's behind vLLM, OpenAI,
   or an RL library's policy — that's the Backend / injected `generate`.

### Conversation framing

**Multi-turn is the default.** A game is an interaction, so `build_messages(state, history)`
replays the episode as a conversation — `system`, the opening user prompt, then for each
completed turn the model's `assistant` reply followed by a `user` message carrying *only*
that guess's feedback. The model therefore sees its own chain of thought across turns and
gets incremental feedback, which is both the natural format for chat models and the right
shape for agentic RL.

Crucially, this keeps the agent **pure and stateless**: `history` (the prior `Turn`s) is owned
by the rollout and passed in; the agent stores nothing. To rebuild the prompt at any point you
need `state` (game truth, from the env) plus `history` (the model's replies, from the
trajectory) — both of which the rollout holds, so episodes stay reproducible and restartable.

> _Self-contained variant:_ if a game's observation fully captures state and you don't care
> about preserving the model's reasoning, an agent may instead return a single
> `render_observation(state)` user message and ignore `history`. That's a deliberate
> simplification, not the default. The single-turn `CharCountAgent`
> (`agents/charcount/agent.py`) is a worked example: one prompt, one `<answer>` reply,
> `history` unused.

---

## Parsing contract

The system prompt asks the model to end with `<guess>word</guess>`. With thinking enabled the
model reasons first (the chat template opens `<think>` in the prompt, so the reply carries the
reasoning, a closing `</think>`, then the guess). `parse_action` is **strict**: it returns the
word inside the **last** `<guess>…</guess>` tag, and nothing else.

If there is no `<guess>` tag, it returns `""` — the env then counts a consumed round with an
`inadequate length` error. We deliberately do **not** dig a candidate out of the reasoning text
or silently retry: "invalid costs a round" is a game rule (`WordleGame.guess`), so a model that
ignores the format pays the same price a human would.

---

## What you get for free (the generic core)

You write the adapter; these are already provided and game-agnostic:

### Backend — `messages → text`

```python
# agents/base.py
class Completion(BaseModel):
    text: str                       # the assistant message content
    finish_reason: str | None = None
    raw: dict | None = None         # full provider response, for debugging
    # NOTE: no token_ids / logprobs — those belong to the generation engine, not us.

class LLMBackend(Protocol):
    def generate(self, prompts: list[list[dict]], **sampling) -> list[Completion]: ...
```

```python
# agents/backend.py
class OpenAICompatBackend:          # vLLM and OpenAI both speak this; only base_url differs
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"): ...
    def generate(self, prompts, *, temperature=0.7, max_tokens=512, **kw) -> list[Completion]: ...
```

The backend is **batch-first** (`prompts: list` → `list[Completion]`); a single call is just
batch-of-1. It is **inference-only** — training supplies its own `generate` (integration doc).

### Runner — drive an episode, record a `Trajectory`

```python
# agents/base.py
Turn       = { messages: list[dict], response: str, action: str, state: GameState }  # state = post-action snapshot
Trajectory = { turns: list[Turn], final: GameState }     # final has status, rounds, target

# agents/rollout.py
def run_episode(agent, env, generate, observer=None) -> Trajectory: ...   # one episode (demo/eval)
def run_eval(pairs, generate, *, concurrency=8) -> list[Trajectory]: ...   # many (agent, env) pairs, threaded
def win_rate(trajectories) -> float: ...                                  # fraction with final.status == "won"
```

`generate` is injected (the backend at inference, the policy in training). Each step the
loop calls `agent.build_messages(state, history=trajectory.turns)`, so the growing
conversation is threaded in from the trajectory — the agent never holds it. `run_eval` just
runs many `run_episode`s on a thread pool (one thread per in-flight game) and lets the
inference server batch the overlapping requests; the loop is the same, only the injected
`generate` and the `Observer` change.

### Observers — visible demo vs. silent batch

```python
# agents/rollout.py
class Observer(Protocol):
    def on_start(self, state) -> None: ...               # before the first move
    def on_step(self, turn: Turn, completion: Completion) -> None: ...   # after each move
    def on_end(self, final) -> None: ...                 # at game end

class NullObserver:        # headless — every hook is a no-op
    ...

class TerminalObserver:    # demo — prints reasoning + action, then a colored board
    def __init__(self, render_fn, *, pace=0.0, step=False, console=None): ...   # render_fn is game-supplied
```

`TerminalObserver` stays generic by taking an injected `render_fn`; `run.py` wires in
Wordle's `games/wordle/play.py::render_board`, so the agent's game looks exactly like a
human's.

---

## Running it

```bash
# Demo: watch one game on a colored board (reasoning + parsed guess shown per move)
uv run --package agents python -m agents.run --demo --word crane
#   --pace <sec>  delay between moves     --step  wait for Enter per move

# Headless: many games in pure code, prints a win-rate summary (no rich)
uv run --package agents python -m agents.run --episodes 20
```

Both go through the same `run_*` loop with `OpenAICompatBackend` as `generate`; the only
difference is `TerminalObserver` vs `NullObserver`.

---

## Configuration (`.env`)

The agent needs to know which API to hit. `run.py` loads this via `python-dotenv`; CLI flags
override:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # local vLLM server, or an OpenAI URL
INFERENCE_MODEL=Qwen/Qwen3.5-0.8B           # the model the server is serving
OPENAI_API_KEY=EMPTY                        # any non-empty value for local vLLM
```

---

## The inference server (`inference/`)

`inference/server.py` is a **thin launcher** over vLLM's built-in OpenAI-compatible server
(no hand-rolled FastAPI). It reads the model id (default `Qwen/Qwen3.5-0.8B`) and host/port
and starts vLLM, exposing `/v1/chat/completions`. Agents connect by pointing
`OPENAI_BASE_URL` at it. Any HF chat/VLM model vLLM supports can be served without agent
changes — the agent only knows "an OpenAI-compatible URL". It also injects a few
overridable launch defaults (context length, GPU-memory fraction, attention backend) so the
default launch fits a ~12 GB card — see [`inference/README.md`](../inference/README.md).

---

## Checklist: adding a new game's agent

1. Create `agents/<game>/agent.py` implementing `GameAgent`: `system_prompt`,
   `build_messages(state, history=())`, `parse_action(text)`. Keep them pure.
2. Reuse the game's `render_observation` for the prompt; do not write a second renderer.
3. Wire the game's board renderer into `run.py` for the demo observer.
4. (For training) add `training/rewards/<game>.py::compute_reward(trajectory, target)` —
   see **[training_integration.md](training_integration.md)**.

Nothing in the generic core (`base.py`, `backend.py`, `rollout.py`) should need to change.

---

## Dependencies (how this design is wired)

- **`agents/pyproject.toml`** — `openai` (the OpenAI-compatible backend), `pydantic`,
  `python-dotenv`, and a workspace dependency on `game-wordle` (for `render_round`,
  `WordleClient`, `render_board`); the demo's `rich` is an optional `[tui]` extra.
- **`inference/pyproject.toml`** — `vllm` (the launcher just `exec`s `vllm serve`).
- **`.env.example`** — `OPENAI_BASE_URL`, `INFERENCE_MODEL`, `OPENAI_API_KEY`.
