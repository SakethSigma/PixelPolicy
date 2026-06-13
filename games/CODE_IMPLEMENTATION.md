# Word-skill games — code implementation

How the word-skill tasks from [DATA_SOURCING.md](DATA_SOURCING.md) plug into the
existing PixelPolicy layers. The original family is six tasks (#1–#6); games #7 (`charset`),
#8 (`mistakeid`), and #9 (`tower`) extend it with more single-turn skills, and games #10
(`codebreaker`), #11 (`bullscows`), and #12 (`consistency`) round it out — #10 and #11 being
**multi-turn** deduction games. The guiding rule is unchanged from the root README: **a new
game must not require changes to agent or training code, and a new agent must not require
changes to any game.** Each task is a full game-env package mirroring `games/wordle/`, plus
a per-task agent and one distillation registry entry.

> Status: **complete.** All twelve games — **#1 (`charcount`), #2 (`validity`), #3 (`anagram`),
> #4 (`endstart`), #5 (`rhyme`), #6 (`crossword`), #7 (`charset`), #8 (`mistakeid`), #9 (`tower`),
> #10 (`codebreaker`), #11 (`bullscows`), and #12 (`consistency`)** — and the **shared vocabulary +
> meanings assets (`games/wordvocab/`)** are implemented, along with the **programmatic generator**
> (`distillation/programmatic.py`, now with a **multi-turn** path), the **batch distillation path**
> (`distillation/batch_play.py`), and a **unified SFT schema** (`distillation/schema.py`) shared by
> every game. **`endstart` was the last of the original six to be built — no game remains a spec.**
> Where the built code's names differ from this doc's proposals, the notes below flag the real
> names.

---

## The shape of a single-turn game

Wordle is multi-turn (6 rounds, each guess depends on prior feedback). **Most** of these tasks
are **single-turn**: `reset()` poses one challenge, `step(answer)` scores it and ends the
episode. That's the only structural difference — everything else (pure core, client
Protocol, render, agent contract, rollout) is identical to Wordle, so the existing generic
machinery (`agents/base.py`, `backend.py`, `rollout.py`) is reused **unchanged**.

> **Two games (#10 `codebreaker`, #11 `bullscows`) are deliberately multi-turn** — they exist to
> teach the Wordle feedback-loop directly. They mirror Wordle exactly: a `guess` verb, per-round
> feedback, `max_rounds` (12 / 10), and `good_status="won"`. They reuse the same generic
> machinery unchanged; the only new piece is a programmatic *multi-turn* generator
> ([see below](#two-producers-one-sft-shape)).

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
`charcount`, `validity`, `anagram`, `endstart`, `rhyme`, `crossword`, `charset`, `mistakeid`,
`tower`, `codebreaker`, `bullscows`, `consistency`. (The multi-turn games `codebreaker` and
`bullscows` expose a `guess` verb instead of `step`, exactly like Wordle.)

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
| charset | a few words (2–4) | submitted used/unused letter sets == union of letters / its a–z complement |
| mistakeid | a Wordle board + a proposed guess | reported mistakes + yes/no flag == errors derived from the board feedback |
| tower | a proposed 3-person placement + per-person floor/room ✓/x feedback | listed placement set == every floor/room assignment consistent with the feedback (1 or 2) |
| codebreaker *(multi-turn)* | a secret 4-slot code over A–F (repeats allowed) | per-slot ✓/-/x feedback via Wordle's `compute_feedback`; `won` when all ✓ |
| bullscows *(multi-turn)* | a secret of 4 distinct digits | bulls/cows counts; `won` when bulls == 4 |
| consistency | a Wordle board + a candidate word | yes/no == is the candidate consistent with every row (reuses Wordle's `compute_feedback`) |

WordNet/`pronouncing` lookups live in the `*Bank` (challenge construction + scoring), not in
the agent — keeping the agent pure and transport-agnostic. The `mistakeid` bank instead loads its
committed `challenges.jsonl` (boards extracted from the original Wordle teacher trajectories) and
derives the true error set from the feedback alone — no target word. The `tower` bank is fully
synthetic — it generates each challenge (random names, a shown placement, and a true placement
that yields the feedback) in pure Python with **no vocab or corpus dependency at all**. The
`codebreaker` and `bullscows` banks are likewise pure synthetic logic (random symbol/digit
secrets, **no vocab dependency**), and each ships a `*Solver` that the multi-turn generator
replays as the teacher. The `consistency` bank instead reuses **Wordle's vocab and scorer**
(`game-wordle`): it builds boards by scoring random guesses against a hidden target and selects a
candidate that is consistent / inconsistent 50/50.

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
`GAME_NUMBERS`). As built, each spec carries `make_agent`, `make_env`, `sample_targets`, and a
`good_status` field (here `"correct"`) that tells the rejection filter which terminal status
counts as solved. Reasoning-distilled specs also set `require_think` (see the anagram/crossword/
mistakeid note below); the multi-turn games set `max_rounds > 1` (codebreaker 12, bullscows 10)
and `good_status="won"`, exactly like Wordle. The registry now wires all twelve games —
`charcount`, `validity`, `anagram`, `endstart`, `rhyme`, `crossword`, `charset`, `mistakeid`,
`tower`, `codebreaker`, `bullscows`, and `consistency` — so nothing is left to add. This is the
same one-place-to-add-a-game contract the distillation README describes.

### Two producers, one SFT shape

Both emit the **unified SFT schema** defined in
[`distillation/schema.py`](../distillation/schema.py) (`sft_row`) — `game_name`, `game_no`,
`round`, `valid`, `target`, `system`, `messages`, `completion`, `completion_no_think`,
`has_think`, `episode` — so the combine + `push.py` step is unchanged. (The legacy Wordle
rows, whose `game` column was the episode index, are upgraded on load by
`schema.normalize_legacy`.)

**A. Programmatic generator (games 1, 2, 4, 5, 7, 9, 10, 11, 12) — no Claude.**
Step the env, read the gold answer the core computed, and format it into the completion (a
trivial "synthetic teacher"). No API cost, fully reproducible. This is
[`distillation/programmatic.py`](../distillation/programmatic.py), built for `charcount`,
`validity`, `rhyme`, `charset`, `tower`, `endstart`, `codebreaker`, `bullscows`, and
`consistency` (selected with `--game`): it loops the bank, self-checks each label, and writes
unified-schema SFT JSONL without touching the generic pipeline.
Defaults: charcount 14,000 rows (≥4,000 Wordle-vocab words + 10,000 WordNet words, lengths 3–20);
validity 13,254 rows (6,627 valid + 6,627 invalid); rhyme 10,000 rows (5,000 MCQ + 5,000 free);
charset 12,000 rows (an even spread of 2-, 3-, and 4-word challenges, each mixing one five-letter
Wordle word with non-five-letter words); tower 5,000 rows (deduction puzzles, ~3,343
single-solution + ~1,657 two-solution); endstart 6,000 rows (MCQ, one matching candidate + 4
distractors, shuffled); consistency 10,000 rows (5,000 yes + 5,000 no, reusing Wordle's scorer).
Tower is programmatic even though it teaches *reasoning*: the consistent placement set is exactly
enumerable, so the teacher just lists it — no `<think>`. **Consistency** goes one step further: its
completion prefixes the `<answer>` with a short, true **worded rationale** of the per-clue check
(`consistency/render.py::render_reasoning`) — templated and **not** wrapped in `<think>` (so
`has_think` stays `False`), derived from the same `compute_feedback` as the label and self-checked
on every row.

**Programmatic multi-turn (games 10, 11).** `programmatic.py` also has a **multi-turn** generator:
`_play_multiturn` reuses [`agents/rollout.py::run_episode`](../agents/rollout.py) with an injected
**solver** as the `generate` callback, emitting **one SFT row per turn** — the same per-move shape
as Wordle's `batch_play`, but at **$0 and with no Claude**. `codebreaker` and `bullscows` use it
with their unbiased `*Solver` (random opening + a uniformly random code consistent with all
feedback so far — a deterministic/ordered solver would teach a biased policy). Each solver's
`move` returns a short, true **worded rationale** (`_reason`, recapping the clues) followed by the
bare `<guess>` — it is **templated, not Claude `<think>`**, so `has_think` stays `False`; it is
derived from the same feedback/candidate-set the guess is drawn from (so always true) and every
episode is self-checked (`status == "won"`). The agents' `build_messages` replay only the bare
`<guess>` of prior turns (the rationale is dropped on replay, exactly as Wordle strips `<think>`),
so the reasoning is a training target without growing the context — `_ok_budget` keeps every
prompt + completion ~400 tokens, well under 4k. A **`--max-rows`** flag caps multi-turn output
**at an episode boundary** (whole episodes are kept, never truncated mid-trajectory), so the round
distribution stays unbiased — it never keeps "only first/last turns". Built: codebreaker capped at
**10,000 rows** (≈2,726 episodes, ~3.7 turns/episode, `--episodes 5000 --max-rows 10000`);
bullscows capped at **10,000 rows** (≈1,823 episodes, ~5.5 turns/episode, `--max-rows 10000`).

**B. Batch distillation (games 3, 6, 8) — Claude + rejection.**
Being single-turn they fit the **Anthropic Batch API** directly — each sample is one independent
request, so there's no Wordle-style lockstep (`distillation/batch_play.md` calls this "a future
single-step dataset mode"; that mode is exactly these games). `anagram` (game 3), `crossword`
(game 6), and `mistakeid` (game 8) are built this way: each is a **reasoning** game distilled at
high adaptive-thinking effort (mistakeid at `max` effort), so its spec sets `require_think=True`
and the gate drops any trace that is wrong *or* lacks a `<think>` block (see the registry note
above). Reuse:
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
"good" status in its `GameSpec.good_status` (`"won"` for Wordle, `"correct"` for the single-turn
games), and the unified schema records the outcome in the row's `valid` flag — so the gate is
game-agnostic with no per-task special-casing. The reasoning-distilled games (anagram, crossword,
mistakeid) layer on `require_think`, which additionally drops any solved trace that lacks a
`<think>` block (a reasoning SFT target with no reasoning is unusable).

**Wordle's gate is format, not correctness.** Unlike the single-turn games, Wordle's SFT `valid`
flag is **not** its win/loss outcome: `distillation/push.py` re-derives it from `has_think` (does
the move carry a `<think>` block). A well-formed reasoned move is a good imitation target even
from a *lost* game, and a move with no reasoning is dropped even from a won one. This is why
Wordle keeps `require_think=False` on its spec — the win-based status no longer drives its `valid`
flag; the format check in `push.py` does.

---

## Dependencies

- `nltk` + the WordNet corpora (`nltk.download('wordnet')`, `nltk.download('omw-1.4')`) —
  one-time download, then offline. Used only at **build time** to produce `vocab.txt` and
  `meanings.jsonl` (the wordvocab `[build]` extra). As built, both `validity` and `crossword` read
  the committed `meanings.jsonl` with **no `nltk` at runtime**.
- `pronouncing` (CMU dict, bundled/offline) — a **runtime** dependency of the `game-rhyme`
  package; used by the `rhyme` bank.

Keep `nltk` as the wordvocab `[build]` extra; `pronouncing` is a normal dependency of the game
packages that need it. No new deps for the generic agent/training layers.

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

1. **Vocabulary asset** — the multi-length list + salted split + meanings asset. ✅ **Built**
   (`games/wordvocab/`).
2. **Programmatic games** (1, 2, 4, 5, 7, 9, 10, 11, 12) — package + agent + programmatic
   generator. Cheap, no API, validates the game shape end to end. **All are built**: single-turn
   `charcount`, `validity`, `endstart`, `rhyme`, `charset`, `tower`, `consistency`; multi-turn
   `codebreaker` and `bullscows` (package, agent, and `distillation/programmatic.py` `--game`
   paths, the multi-turn ones driven by the `_play_multiturn` generator + `--max-rows`).
3. **Reasoning-distilled games** (3, 6, 8) — package + agent + registry entries; wire the Batch
   distillation path and confirm rejection sampling drops wrong traces. **Games #3 `anagram`,
   #6 `crossword`, and #8 `mistakeid` are built** (all reasoning, `require_think`; mistakeid at
   `max` effort).
4. **Combine + push** — unchanged `push.py`; one Hub dataset across all games. **Done** for all
   thirteen games (pushed to
   [`saketh-chervu/word-games-distillation`](https://huggingface.co/datasets/saketh-chervu/word-games-distillation),
   96,162 rows = 3,078 Wordle + 14,000 charcount + 13,254 validity + 1,000 anagram + 10,000
   rhyme + 1,500 crossword + 12,000 charset + 330 mistakeid + 5,000 tower + 6,000 endstart +
   10,000 codebreaker + 10,000 bullscows + 10,000 consistency; 95,520 valid, where Wordle's
   `valid` is format compliance, `has_think`. Multi-turn share ~24%).

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
