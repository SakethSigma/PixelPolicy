# Word-skill games — code implementation

How the six word-skill tasks from [DATA_SOURCING.md](DATA_SOURCING.md) plug into the
existing PixelPolicy layers. The guiding rule is unchanged from the root README: **a new
game must not require changes to agent or training code, and a new agent must not require
changes to any game.** Each task is a full game-env package mirroring `games/wordle/`, plus
a per-task agent and one distillation registry entry.

> Status: partially built. **Game #1 (`charcount`)** and the **shared vocabulary asset
> (`games/wordvocab/`)** are implemented, along with the **programmatic generator**
> (`distillation/programmatic.py`) and a **unified SFT schema** (`distillation/schema.py`)
> shared by every game. Games 2–6 remain a spec to build from. Where the built code's names
> differ from this doc's proposals, the notes below flag the real names.

---

## The shape of a single-turn game

Wordle is multi-turn (6 rounds, each guess depends on prior feedback). These tasks are
**single-turn**: `reset()` poses one challenge, `step(answer)` scores it and ends the
episode. That's the only structural difference — everything else (pure core, client
Protocol, render, agent contract, rollout) is identical to Wordle, so the existing generic
machinery (`agents/base.py`, `backend.py`, `rollout.py`) is reused **unchanged**.

A single-turn episode is just a game that reaches a terminal `status` after one `step`:

```
reset() ─► GameState(challenge, status="in_progress")
              │  render_observation(state) ─► prompt
              │  model ─► "<answer>…</answer>"
              ▼
step(answer) ─► GameState(..., status="correct" | "incorrect", solution revealed)
```

We reuse the Wordle `status` convention so the distillation filter is game-agnostic: the
terminal "good" status plays the role of Wordle's `"won"` (see
[Rejection sampling](#rejection-sampling-is-the-same-gate)).

---

## Per-task game package — `games/<task>/`

Mirror the Wordle package table (see [wordle/README.md](wordle/README.md)). Tasks:
`charcount`, `validity`, `anagram`, `endstart`, `rhyme`, `crossword`.

| File | Responsibility |
|------|----------------|
| `game.py` | **Pure core.** A single-turn `*Game` (`reset` → challenge, `step(answer)` → scored terminal `GameState`) that **owns the ground truth**, plus a `*Bank` that loads the shared word list/split once and builds challenges. No FastAPI, no reward. |
| `render.py` | Dependency-free text — turns a `GameState` challenge into the exact observation a human and the model read. |
| `client.py` | The shared `*Client` Protocol + `Local*Client` (in-process) and `HTTP*Client`. Same two-transport pattern as Wordle. |
| `server.py` | Thin async FastAPI wrapper: `POST /reset`, `POST /step`, `GET /state/{id}`. |
| `tests/` | Core scoring, render parity, client parity — mirror `games/wordle/tests/`. |

**The core owns ground truth** (so scoring never drifts and the rejection filter stays
game-agnostic):

| Task | `reset()` challenge | `step(answer)` checks |
|------|---------------------|------------------------|
| charcount | a word | parsed counts == computed counts |
| validity | a word (real or pseudo) | valid/invalid == WordNet membership (meaning compared loosely / non-empty) |
| anagram | two words | yes/no == `sorted(w1)==sorted(w2)` |
| endstart | word1 + 5 candidates | choice == the candidate starting with `word1[-1]` |
| rhyme | a word (+options for MCQ) | answer ∈ `pronouncing.rhymes(word)` |
| crossword | definition + length + masked pattern | answer == target **and** matches revealed letters |

WordNet/`pronouncing` lookups live in the `*Bank` (challenge construction + scoring), not in
the agent — keeping the agent pure and transport-agnostic.

---

## Per-task agent — `agents/<task>/agent.py`

Implement the existing `GameAgent` protocol (`agents/base.py`); reuse `base.py`,
`backend.py`, `rollout.py` unchanged (see [agents/Readme.md](../agents/Readme.md)).

- `system_prompt` — task instructions + the required output tags (`<answer>`, and
  `<meaning>` for validity, `<think>` for the reasoning games). For thinking models, don't
  ask for `<think>` (the chat template opens it); just specify the final answer tag, exactly
  as the Wordle agent does.
- `build_messages(state, history=())` — single user turn from `render_observation(state)`;
  `history` is unused for one-shot tasks (the [self-contained variant](../agents/Readme.md#conversation-framing)
  the agent doc already sanctions).
- `parse_action(text)` — strict extraction of the last `<answer>…</answer>` (and
  `<meaning>` where relevant); return `""` on absence so the env scores it incorrect — same
  "malformed costs you the round" contract as Wordle's `parse_action`.

---

## Distillation wiring — one registry entry per task

`distillation/batch_play.py`, `programmatic.py`, `schema.py`, and `push.py` stay
game-agnostic; a new task is added **only** in
[`distillation/registry.py`](../distillation/registry.py) as a `GameSpec` (plus its number in
`GAME_NUMBERS`). As built, `charcount`'s spec carries `make_agent`, `make_bank` (loads the
word list/split once, shared across episodes), `reset_env`, `sample_target`, and a
`good_status` field (here `"correct"`) that tells the rejection filter which terminal status
counts as solved. This is the same one-place-to-add-a-game contract the distillation README
describes.

### Two producers, one SFT shape

Both emit the **unified SFT schema** defined in
[`distillation/schema.py`](../distillation/schema.py) (`sft_row`) — `game_name`, `game_no`,
`round`, `valid`, `target`, `system`, `messages`, `completion`, `completion_no_think`,
`has_think`, `episode` — so the combine + `push.py` step is unchanged. (The legacy Wordle
rows, whose `game` column was the episode index, are upgraded on load by
`schema.normalize_legacy`.)

**A. Programmatic generator (games 1, 2, 4, 5) — no Claude.**
Step the env, read the gold answer the core computed, and format it into the completion (a
trivial "synthetic teacher"). No API cost, fully reproducible. This is
[`distillation/programmatic.py`](../distillation/programmatic.py) (already built for
`charcount`): it loops the bank, self-checks each label, and writes unified-schema SFT JSONL
without touching the generic pipeline. Its charcount default emits 14,000 rows (≥4,000
Wordle-vocab words + 10,000 WordNet words spanning lengths 3–20).

**B. Batch distillation (games 3, 6) — Claude + rejection.**
These want reasoning, and being single-turn they fit the **Anthropic Batch API** directly —
each sample is one independent request, so there's no Wordle-style lockstep
(`distillation/batch_play.md` calls this "a future single-step dataset mode"; that mode is
exactly these games). Reuse:
- `AnthropicBackend` (`agents/backend.py`) — one-shot or `batch_generate(prompts, …)`,
  returns `Completion`s in input order with full `raw`/`usage`.
- `run_eval` (`agents/rollout.py`) for the live path, or a thin batch driver modeled on
  `distillation/batch_play.py` for the batch path (single round, since one turn).
- the **filter + explode** step — keep solved episodes, then one row per move (here, one
  move per episode), emitted in the unified schema by `batch_play.py`'s SFT writer.
- Cost/safety from `distillation/cost_probe.py`: the `PRICING` table and
  `with_options(timeout=…, max_retries=0)` (a retried in-flight request is **billed twice**
  — the double-billing guard noted in `batch_play.md`).

### Rejection sampling is the same gate

The Wordle filter keeps episodes with `final.status == "won"`. Here the core sets a terminal
`status` (e.g. `"correct"`) when `step(answer)` matches its ground truth, so the **same
filter** keeps only correct traces — programmatic samples pass by construction, and Claude
traces that reasoned to a wrong answer are dropped. As built, each game declares its terminal
"good" status in its `GameSpec.good_status` (`"won"` for Wordle, `"correct"` for charcount),
and the unified schema records the outcome in the row's `valid` flag — so the gate is
game-agnostic with no per-task special-casing.

---

## Dependencies

- `nltk` + the WordNet corpora (`nltk.download('wordnet')`, `nltk.download('omw-1.4')`) —
  one-time download, then offline. Used by `validity`/`crossword` banks and to build the
  multi-length vocabulary.
- `pronouncing` (`pip install pronouncing`) — CMU dict, offline after install. Used by the
  `rhyme` bank.

Add these to the relevant `pyproject.toml`s (the game packages that need them, and
`distillation` for vocab building). No new deps for the generic agent/training layers.

---

## Shared vocabulary build

Built as [`games/wordvocab/`](wordvocab/README.md). `build.py` produces the multi-length word
list described in [DATA_SOURCING.md](DATA_SOURCING.md#shared-vocabulary-asset): take WordNet
lemmas, filter to lowercase-alpha single tokens in length range **3–20**, **union with the
full Wordle vocab** (train + val, 12,972 words), and commit the result as `vocab.txt` (so
downstream packages read it with **no `nltk`** at runtime — `nltk` is only the `[build]`
extra). `split.py::assign_pool(game, word)` then splits **per game** with a game-salted rule —
`sha256(f"{game}:{word}") % 1000 < 200 → val else train` (a salted variant of
[`games/wordle/game.py`](wordle/game.py)'s `assign_pool`). This is the deliberate cross-game
design: a word that's val for one game (e.g. a Wordle val word) is train for another. Because
`assign_pool` is deterministic, banks derive their split at load time with **no per-game
artifact to commit**; regenerating `vocab.txt` is a run-on-purpose step. Wordle keeps its
existing unsalted committed split.

---

## Build order (suggested)

1. **Vocabulary asset** — the multi-length list + salted split. ✅ **Built**
   (`games/wordvocab/`).
2. **Programmatic games** (1, 2, 4, 5) — package + agent + programmatic generator. Cheap,
   no API, validates the single-turn game shape end to end. **Game #1 `charcount` is built**
   (package, agent, and `distillation/programmatic.py`); games 2, 4, 5 remain.
3. **Reasoning games** (3, 6) — package + agent + registry entries; wire the Batch
   distillation path and confirm rejection sampling drops wrong traces. Not yet built.
4. **Combine + push** — unchanged `push.py`; one Hub dataset across all games. **Done** for
   Wordle + charcount (pushed to
   [`saketh-chervu/word-games-distillation`](https://huggingface.co/datasets/saketh-chervu/word-games-distillation),
   17,078 rows = 3,078 Wordle + 14,000 charcount).

---

## Verification

- Per-game `tests/` (core scoring, render parity, Local/HTTP client parity) — mirror
  `games/wordle/tests/`.
- **Train == inference:** for a fresh agent, `build_messages(reset_state)` equals the
  `messages` stored in the produced SFT sample (the byte-identical-to-inference guarantee the
  unified schema in [`schema.py`](../distillation/schema.py) preserves).
- **Programmatic correctness:** generated `<answer>` matches an independent recomputation of
  the label for a sample of words.
- **Rejection works:** on a small Claude batch for games 3/6, confirm wrong-answer traces
  are filtered out and only correct `<think>…</think><answer>…</answer>` samples remain.
- **Cost:** for the batch path, the per-sample cost ≈ half the live figure (Batch discount),
  using `cost_probe.py`'s pricing.
