# Agent ↔ Training & Inference integration

How the agent pieces plug into a **training loop** (RL/GRPO/SFT) and into an **inference /
eval harness**. The companion doc **[Readme.md](Readme.md)** covers how to *write* an agent;
this doc covers how something *drives* one and turns episodes into a learning signal.

The guiding constraint: **`agents/` must not depend on any training library.** The RL
library we pick (verl / TRL / OpenRLHF — undecided) owns generation, tokenization, logprobs,
advantages, and weight updates. Our job is to expose a small, pure surface it consumes. The
dependency points one way: `training/` imports `agents/`, never the reverse.

---

## The integration surface — four pure pieces + one data type

Everything an integrator needs, and nothing more:

| Piece | Where | Game-aware? | Used by |
|-------|-------|-------------|---------|
| `agent.build_messages(state, history=()) -> messages` | `agents/<game>/agent.py` | yes | training **and** inference |
| `agent.parse_action(text) -> action` | `agents/<game>/agent.py` | yes | training **and** inference |
| `env.reset(...)` / `env.guess(action) -> GameState` | `games/<game>/client.py` | yes | training **and** inference |
| `compute_reward(trajectory, target) -> float` | `training/rewards/<game>.py` | yes | training only |
| `Trajectory` / `Turn` (text + game-state record) | `agents/base.py` | no | training **and** inference |

**What is explicitly NOT ours** (lives in the RL library / generation engine):

- the **generation engine** (its in-process vLLM/policy) and the `generate` call,
- **`token_ids` and `logprobs`** — captured by the library against the exact weights being
  updated; ours would be redundant and, mid-training, *wrong*,
- **credit assignment** — turning an episode reward into per-turn/per-token advantages
  (GRPO group-norm, PPO/GAE discounting),
- the **optimizer / weight update**.

---

## Components matrix: training vs. inference

| Component | Inference / eval | Training |
|-----------|------------------|----------|
| `build_messages`, `parse_action` (agent) | ✅ | ✅ |
| `env` (`LocalWordleClient` / HTTP) | ✅ | ✅ (in-process, `LocalWordleClient`) |
| `Trajectory` / `Turn` | ✅ (metrics, demo) | ✅ (reward input; turn boundaries for credit assignment) |
| `generate(prompts) -> [Completion]` | `OpenAICompatBackend` (HTTP → vLLM) | the **library's policy** |
| `run_episode` / `run_eval` | ✅ our driver | optional — only if the framework lets us drive |
| `Observer` (Terminal/Null) | ✅ | ✗ |
| `compute_reward` | ✗ | ✅ |
| tokenization, logprobs, advantages, optim | ✗ | the **library**, not us |

The agent's two pure functions and the env are the *only* things shared verbatim across both
worlds — that's what guarantees train/eval parity.

---

## Reward & credit assignment (the part games get wrong)

Games almost never have a true per-step reward. The normal case is a **terminal/episodic**
reward that is **credit-assigned backward across the turns** that produced it. So:

- **The rollout unit is a full episode** (`Trajectory`), not a single prediction. A rollout
  plays start→end and returns the whole trajectory; it scores nothing.
- **Reward is a function of the cumulative outcome:** `compute_reward(trajectory, target)
  -> float` reads `final.status`, the number/quality of rounds, etc. One scalar per episode.
- **Credit assignment is the library's job, not ours.** GRPO normalizes the episode scalar
  within a group → one advantage per episode → that advantage is broadcast to every turn
  (and every token of every turn). PPO discounts/GAEs it backward. We never distribute
  reward across steps ourselves.

**The one requirement this puts on `Trajectory`:** it must preserve clean **per-turn
boundaries** (which `messages`/`response` was each turn) so the library can re-tokenize each
turn and apply the episode advantage to those tokens. Ours does — a list of `Turn`s plus the
`final` state. Still no token_ids from us; the library tokenizes the turn text itself.

Reward stays flexible without touching `agents/`:
- **Episodic (default):** scalar at game end, distributed by the optimizer.
- **Per-step shaping (optional):** if you want dense rewards (e.g. info-gain per guess),
  `compute_reward` can return a per-turn vector — still just a function of the same
  `Trajectory`.

---

## Mode A — single-step integration (one prediction = one training item)

The simplest regime: treat each `(state → guess)` as an independent item, with a reward for
that single step. Bandit / SFT-style; useful for warmups, format training, or per-turn
shaped objectives. There is no multi-turn credit assignment.

```python
# dataset of game states (boards) → one prediction each → reward per item
prompts = [agent.build_messages(s) for s in states]      # pure
comps   = policy.generate(prompts)                        # library's policy (batched)
items   = []
for s, c in zip(states, comps):
    action = agent.parse_action(c.text)                   # pure
    next_state = env_for(s).guess(action)                 # one env step
    r = step_reward(s, action, next_state)                # per-step reward (training/)
    items.append((prompts_i, c.text, r))                  # → SFT/bandit/GRPO-1step update
```

**Components used:** `build_messages`, `parse_action`, `env.guess`, a per-step reward. No
`Trajectory` needed (each item is one turn). This is the case where "each prediction → each
training item" is literally true.

---

## Mode B — multi-step agentic rollout (episodic reward)

The default for game training: play the whole episode, score the trajectory, let the
optimizer assign credit across turns. Who drives the loop has two sub-cases.

### B1 — we drive the loop (inject the library's `generate`)

If the framework is happy to call out for rollouts, reuse our generic driver and just inject
its policy as `generate`:

```python
from agents.wordle.agent import WordleAgent, WordleEnv
from agents.rollout import run_eval
from games.wordle.client import make_local_group     # G clients sharing one target

clients = make_local_group(G, word=target)           # pin a word → a GRPO group (already reset)
pairs   = [(WordleAgent(), WordleEnv(c)) for c in clients]

trajs   = run_eval(pairs, generate=policy.generate)  # G full Trajectories
rewards = [compute_reward(t, t.final.target) for t in trajs]           # one scalar each
# hand (per-turn messages+response, episode reward) to the library's GRPO/PPO update;
# the library tokenizes turns, normalizes rewards, assigns advantages.
```

`run_eval` runs each `(agent, env)` episode on a thread pool (`run_episode` per game): it builds
prompts with `build_messages`, calls `generate`, parses with `parse_action`, and advances until
every episode finishes. Generation isn't hand-vectorized — each episode issues its own request and
the inference engine batches the overlapping ones (vLLM continuous batching). It returns
`list[Trajectory]` and owns nothing else.

### B2 — the framework drives the loop (agentic RL frameworks)

verl-style multi-turn/agentic trainers drive generation themselves and only reach into our
**pure pieces** at the integration points. Then `run_eval` is *not* used in
training; the framework maintains the running conversation and calls:

```
framework generates a turn (its policy, its tokens/logprobs)   # one assistant message
   → agent.parse_action(text)        # our pure fn: text → move
   → env.guess(action)               # advance the game
   → append the assistant reply + the next user message to the conversation
        next user message = render_round(new_round)   # incremental feedback (our renderer)
   ... repeat until status != in_progress ...
   → compute_reward(trajectory, target)   # our reward (training/)
```

Because the agent is **multi-turn by default** and prefix-stable, the framework has two
equivalent options each step: re-call `agent.build_messages(state, history)` to get the full
message list, or just append the increment (the generated assistant reply +
`render_round(new_round)`). Either way the conversation it trains on is identical to the one
our rollout would build — each **assistant turn is one action span** the optimizer assigns
credit to. Both sub-cases consume the **same four pure pieces**; we don't depend on which way
the framework works.

### GRPO grouping (falls out of the env)

```
make_local_group(G, word=target)   # G envs share one answer ← env supports reset(word=...)
  → roll out G full Trajectories
  → r_i = compute_reward(traj_i, target)         # scalar per episode
  → A_i = (r_i - mean(r)) / std(r)               # group-normalize        ← library
  → broadcast A_i to every Turn (every token) of episode i → token batch  ← library
```

A group shares the target so the only thing differing across episodes is the policy's play —
which is exactly what GRPO compares.

---

## Inference / eval integration

Same pieces, generation owned by us via the HTTP backend:

```python
from agents.backend import OpenAICompatBackend
from agents.rollout import run_eval, run_episode, win_rate, TerminalObserver
from agents.wordle.agent import WordleAgent, WordleEnv
from games.wordle.client import LocalWordleClient
from games.wordle.play import render_board

backend = OpenAICompatBackend(base_url=..., model=...)   # → vLLM or OpenAI

# Eval: many episodes, no rendering, compute win-rate over the val pool
def make_env():
    env = WordleEnv(LocalWordleClient(bank)); env.reset(mode="val"); return env
pairs = [(WordleAgent(), make_env()) for _ in range(N)]
trajs = run_eval(pairs, generate=backend.generate)       # threaded; envs already reset
wr    = win_rate(trajs)                                   # fraction with final.status == "won"

# Demo: one episode on a colored board
env = make_env()
run_episode(WordleAgent(), env, generate=backend.generate,
            observer=TerminalObserver(render_board, step=True))
```

`compute_reward` is optional at inference (you usually just want metrics from `Trajectory`).

---

## Checklist: integrating a new training library

You write, in `training/`, only:

1. **A `generate` adapter** exposing the library's policy as
   `generate(prompts: list[messages]) -> list[Completion]` — *only if you drive the loop
   (B1)*. If the framework drives (B2), you skip this and just call our pure pieces in its
   hooks.
2. **`compute_reward(trajectory, target) -> float`** (or a per-turn vector) in
   `training/rewards/<game>.py`.

Then let the library handle tokenization, logprobs, advantages, and the optimizer. Swapping
verl for TRL later changes only item (1); nothing in `agents/` or `games/` moves.
