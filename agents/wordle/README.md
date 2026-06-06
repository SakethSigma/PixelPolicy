# The Wordle agent — input/output, turn by turn

This document shows **exactly what the model sees and says** when it plays Wordle: how each
round's observation is constructed into chat messages, what the model is expected to reply, and
how that reply is parsed back into a guess. Every block below is the **real text** produced by
the code in [`agent.py`](agent.py) (with [`games/wordle/render.py`](../../games/wordle/render.py)),
not a paraphrase.

If you want the architecture/contract instead, see [`agents/Readme.md`](../Readme.md). This file
is purely about the **wire-level conversation**.

> **Reading the transcripts below:** `[system]` / `[user]` / `[assistant]` are message roles, and
> the indented lines under each are that message's exact `content`. Any `←` arrow is a doc
> annotation only — it is **not** sent to the model. The real assistant content is just the
> `<guess>…</guess>` line; the real user content is just the feedback lines.

---

## The loop in one picture

```
        ┌─────────────────────────────── one turn ───────────────────────────────┐
state ─►│ agent.build_messages(state, history) ─► messages ─► backend.generate ──►│─► completion.text
        │                                                                          │
guess ◄─│ env.step(guess) ◄─ agent.parse_action(completion.text) ◄────────────────│
        └──────────────────────────────────────────────────────────────────────────┘
                 the rollout (agents/rollout.py) drives this until status != "in_progress"
```

The agent is **stateless**. The growing conversation is *not* stored on it — the rollout passes
in `history` (the prior `Turn`s) every call, and `build_messages` rebuilds the whole message list
from scratch. That is what makes episodes reproducible and restartable.

---

## 1. The system prompt (constant, every turn)

Defined once as `WordleAgent.system_prompt`. Note it does **not** ask the model to emit `<think>`
tags — a Qwen3.5 *thinking* model opens `<think>` itself via its chat template, so we only ask for
the final `<guess>`:

```
You are an expert Wordle player.

I have chosen a secret 5-letter English word. You have 6 guesses to find it. After each guess, every letter is scored:
  ✓  correct letter in the correct position
  -  correct letter but in the wrong position
  x  the letter is not in the secret word at all

Rules:
- Each guess must be a real 5-letter English word.
- An invalid guess (wrong length, or not a real word) still uses up one of your 6 guesses, so never waste one.
- Use every clue: keep ✓ letters in their position, move - letters to a different position (they ARE in the word), and never reuse x letters.
- A letter can repeat; feedback is per position.

Think through the clues, then give your final answer on its own line as:
<guess>word</guess>
where word is a single lowercase 5-letter English word. Output the <guess> tag exactly once, as the very last thing you write.
```

---

## 2. Turn 1 — opening move

`build_messages(state, history=[])` with a fresh game produces just the system prompt and an
opening nudge:

```
[system]
  <the system prompt above>

[user]
  Make your first guess.
```

The model replies (reasoning in `<think>`, answer in `<guess>` — this is a real Qwen3.5-style
reply; the `completion.text` the backend returns):

```
<think>No clues yet. I will open with CRANE — common letters, two vowels.</think>
<guess>crane</guess>
```

`parse_action(text)` pulls the word from the **last** `<guess>…</guess>` and lowercases it:

```
parse_action(...) -> "crane"
```

The rollout calls `env.step("crane")`, which returns the updated `GameState`. For the secret
word **VIVID**, none of `C R A N E` are in the word, so the round scores all `x`.

---

## 3. How feedback becomes the next user message

After each guess, `WordleAgent._feedback(state)` builds the user message for the next turn from
the game's own renderer ([`render_round`](../../games/wordle/render.py)) plus a rounds-left nudge:

```
C R A N E   x x x x x
Rounds left: 5. Make your next guess.
```

- The first line is **byte-identical** to what a human sees — it comes from the same
  `render_round` used by the terminal UI, so the text the model trains/evals on can never drift
  from the human view.
- The `Rounds left: N` line is only added while the game is `in_progress` (see the terminal cases
  in §6).

---

## 4. Turn 2 — the conversation is replayed (with `<think>` stripped)

`build_messages(state, history=[turn1])` rebuilds the **entire** message list. The model's prior
reply is replayed as an `assistant` message, but its `<think>…</think>` block is **stripped** —
matching Qwen's multi-turn guidance (don't feed old chain-of-thought back in) and keeping context
small. Only the visible `<guess>` answer remains:

```
[system]
  <the system prompt above>

[user]
  Make your first guess.

[assistant]
  <guess>crane</guess>          ← turn 1's reply, <think> removed on replay

[user]
  C R A N E   x x x x x         ← turn 1's feedback
  Rounds left: 5. Make your next guess.
```

The list is **prefix-stable**: each new turn appends exactly one `(assistant, user)` pair, so
turn *N*'s prompt is turn *N-1*'s prompt plus the model's last answer and its feedback. (This is
also the exact shape an agentic RL trainer wants — each assistant span is one action.)

---

## 5. Feedback symbols & duplicate letters

Per-letter scoring uses three symbols (from `LetterFeedback` in
[`games/wordle/game.py`](../../games/wordle/game.py)):

| Symbol | Meaning |
|:------:|---------|
| `✓` | right letter, **right** position |
| `-` | right letter, **wrong** position (it IS in the word) |
| `x` | letter not in the word (or all its copies already accounted for) |

Duplicates are scored with a two-pass rule (greens first consume a target copy, then yellows).
Example for secret **VIVID**, guess `DIVED`:

```
D I V E D   x ✓ ✓ x ✓
```

The trailing `D` is `✓` (correct position), which consumes the *only* `D` in VIVID — so the
leading `D` scores `x`, not `-`. That is correct Wordle behavior, and it's the same logic for a
human and the model.

---

## 6. Invalid guesses and terminal states

**Invalid guess** — it still **costs a round**, and the feedback says *why*, with no per-letter
scoring (so a non-word can't probe letters for free). There are two reasons, validated in order
(structural first, then vocabulary):

*Wrong length* (`"cat"` — not 5 letters):

```
CAT  [invalid: inadequate length — counted as a round]
Rounds left: 5. Make your next guess.
```

*Out of vocabulary* (`"zzzzz"` — 5 letters but not a real word):

```
ZZZZZ  [invalid: out of vocabulary — counted as a round]
Rounds left: 5. Make your next guess.
```

**Win / loss** — once `status != "in_progress"`, the feedback is just the final line, with **no**
`Rounds left:` nudge:

```
V I V I D   ✓ ✓ ✓ ✓ ✓
```

At that point the rollout stops; `state.status` is `"won"` or `"lost"` and `state.target` is
revealed (uppercased).

---

## 7. A full multi-round conversation (incl. invalid rounds)

This is the **complete message list** `build_messages` produces going into the 4th turn of a game
on secret **VIVID**, after the model played: `crane` (valid), `cat` (wrong length), `zzzzz` (out of
vocabulary). Every invalid guess still consumed a round — note `Rounds left` ticking 5 → 4 → 3 — and
each prior reply is replayed with its `<think>` stripped. The whole history is rebuilt from
`(state, history)` each turn; nothing is stored on the agent.

```
[system]
  <the system prompt from §1>

[user]
  Make your first guess.

[assistant]
  <guess>crane</guess>                                    ← turn 1 reply (think stripped)

[user]
  C R A N E   x x x x x                                   ← valid guess: per-letter feedback
  Rounds left: 5. Make your next guess.

[assistant]
  <guess>cat</guess>                                      ← turn 2 reply

[user]
  CAT  [invalid: inadequate length — counted as a round]  ← wrong length: no per-letter feedback
  Rounds left: 4. Make your next guess.

[assistant]
  <guess>zzzzz</guess>                                    ← turn 3 reply

[user]
  ZZZZZ  [invalid: out of vocabulary — counted as a round]← not a real word: no per-letter feedback
  Rounds left: 3. Make your next guess.
```

Things to notice:
- **Valid vs invalid rounds look different in the transcript.** A valid round carries the
  `✓ - x` feedback; an invalid one carries an `[invalid: …]` reason and *no* letters — the model
  must learn from the reason, not probe for letters.
- **Both invalid kinds cost a round.** The game treats `cat` and `zzzzz` exactly as it would for a
  human typing them — the agent gets no free retry (`parse_action` submits whatever it parsed; the
  env decides legality). See [`games/wordle/README.md`](../../games/wordle/README.md) → *Invalid
  guesses*.
- **Prefix-stable.** Each turn only *appends* one `(assistant, user)` pair; earlier messages are
  byte-for-byte unchanged.

---

## 8. The parsing contract (`parse_action`)

**Strict** — the model must put its answer in a `<guess>` tag:

1. return the word inside the **last** `<guess>…</guess>` tag (lowercased); else
2. return `""`.

No digging a 5-letter word out of the reasoning, no raw-text fallback. The env — not the agent —
decides legality: an empty (or non-word) guess is a consumed round with an `error` (the §6 invalid
case), exactly as for a human typing a non-word. We deliberately **do not** silently retry —
"invalid costs a round" is a game rule, so a reply that ignores the format pays the same price.

```python
parse_action("<think>…</think>\n<guess>CRANE</guess>")  # -> "crane"
parse_action("I think it's plumb")                       # -> ""   (no <guess> tag → burns a round)
```

---

## 9. Watch the agent play (demo)

Two terminals: one hosts the model, the other runs the game loop.

### Terminal A — host the model (vLLM)

```bash
uv run --package inference python -m inference.server
```

Defaults to `Qwen/Qwen3.5-0.8B` and auto-adds 12 GB-safe flags (`--max-model-len 8192`,
`--gpu-memory-utilization 0.85`, and text-only mode via `--limit-mm-per-prompt`). Any flag you pass
overrides the matching default. See [`inference/README.md`](../../inference/README.md) for details.

**Wait for `Application startup complete` / `Uvicorn running on http://127.0.0.1:8000`** before
connecting — on WSL the first load is slow (~3-4 min) and mostly quiet after the startup banner.
Sanity-check from another shell:

```bash
curl -s http://127.0.0.1:8000/v1/models      # should list the served model id
```

**Tips**
- **CUDA OOM?** Lower memory with `--gpu-memory-utilization 0.80` or shorten context with
  `--max-model-len 4096`.
- **Want a stronger player?** Serve a Wordle fine-tune instead:
  `--model saketh-chervu/qwen3-06b-wordle-sft-phase1-best` (text-only Qwen3-0.6B, plays much
  better than the base model).
- The agent reads the endpoint from `.env` (`OPENAI_BASE_URL`, `INFERENCE_MODEL`, `OPENAI_API_KEY`).
  If you serve a different `--model`, set `INFERENCE_MODEL` to match (or pass `--model` to the run
  command below).

### Terminal B — run the game loop

```bash
# watch one game on a colored board — prints the model's reasoning + parsed guess per move
uv run --package agents python -m agents.run --demo --word vivid
#   --pace <sec>   delay between moves        --step   wait for Enter per move
#   --model <id>   override INFERENCE_MODEL    --base-url <url>   override OPENAI_BASE_URL

# headless: many games, prints a win-rate summary (no rich UI)
uv run --package agents python -m agents.run --episodes 20
#   --mode val|train    --concurrency <n>     --word <w>  (pin every game to one answer)
```

The demo's per-move printout (`TerminalObserver`) shows the same `completion.text` and parsed
guess described in §2–§4, then renders the board with the `✓ - x` tiles — exactly the
conversation this document walks through.
