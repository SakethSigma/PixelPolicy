# Distillation — Blog Notes (Wordle teacher data from Claude)

> Working notes to refine into a blog post later. Captures *what* we did and *why*, the
> numbers, and the lessons — not the code.

## The goal

Teach a small open model (Qwen3.5-0.8B) to play Wordle by **imitating a strong teacher**
(Claude Sonnet 4.6). The small base model plays poorly; instead of hand-crafting a reward,
we have Claude play many games, keep the good ones, and turn each move into a supervised
training example. This is rejection-sampling distillation: the dataset *is* the teacher's
behavior.

**Reuse over rebuild.** A teacher is just another LLM backend, so the *same* game loop that
runs inference also records the teacher's trajectories. We didn't rebuild the game or agent —
we drove the existing pieces with a Claude backend.

## Cost probing across reasoning effort (the core experiment)

Claude Sonnet 4.6 uses *adaptive thinking* — it decides when and how much to reason — and an
`effort` knob (low / medium / high) tunes the depth. Reasoning is the expensive part, so
before generating at scale we measured what effort actually costs.

We built a **cost probe**: play N games, capture the *real* token usage per move, and price
it from list rates. Run on the same 5 words at two effort levels:

| effort | output (thinking) tokens | cost / game | →500 games | solved |
|--------|--------------------------|-------------|-----------|--------|
| high   | 44,229                   | $0.138      | ~$69      | 3/5    |
| low    | 10,655                   | $0.038      | ~$19      | 4/5    |

**Takeaways:**
- **Output (thinking) tokens dominate cost** — ~95% of it. Effort is *the* lever.
- **Low effort was ~4× cheaper at comparable accuracy** here. (Small sample — treat the
  accuracy as "roughly equal," not "low is better.")
- Prompt caching didn't help: our system prompt (~250 tokens) is below the model's cacheable
  minimum, so it never cached. (Worth revisiting only if prompts grow.)
- Cheaper has a catch: at low effort the teacher *skips* reasoning on easy moves (e.g. the
  opening guess), so some samples have no `<think>` trace. Reasoning coverage trades against cost.

**Decision — scale effort, don't pay flat.** Rather than buy high effort everywhere, we mixed:
a large **low-effort** bulk (cheap, mostly-reasoned) plus a smaller **high-effort** slice
(richer reasoning). Final mix: 400 low + 200 high = 600 games.

## Scaling with the Batch API (lockstep)

The Anthropic **Batch API is ~50% cheaper** (asynchronous). Stacking batch + low effort
roughly halves the already-cheap low cost.

The wrinkle: a batch is a bag of *independent* requests, but Wordle is multi-turn (guess N
needs feedback N-1), so a whole game can't be one batch. Solution — **lockstep batching across
games**: batch every active game's round-N guess together, apply each game's feedback locally,
then batch round N+1 for whoever's still playing. N games = at most 6 batches instead of 6×N
live calls.

## Resilience (a lesson we paid for)

Early on, a stalled network connection during a long run triggered the SDK's silent
auto-retries and quietly **double-billed ~$1**. Fixes:
- **Batches are durable server-side** (run to completion, results kept ~29 days) — a dropped
  connection only kills the local poller, not the batch.
- Polling now **retries through network blips** instead of crashing; the create call does *not*
  auto-retry (avoids duplicate batches).
- We **checkpoint after every round** and persist the in-flight batch id, so `--resume` replays
  the saved guesses to restore state and re-attaches to a running batch — no re-submit, no
  double cost.

## What we store (decide filters at training time, not now)

Two formats per run:
- **Raw** — full Claude responses + per-move token usage (provenance, cost, inspection).
- **Processed SFT** — one sample per move: the exact inference prompt, plus the completion in
  **two forms** (`completion` with `<think>…</think>`, and `completion_no_think` stripped to
  just the guess) and a `has_think` flag.

Keeping all variants + flags means the training target (reason vs answer-only) and any quality
filter (effort, reasoned-only) are **training-time choices**, not baked into generation.

## Prompt tweaks

Small additions to the (shared) system prompt to improve teacher data quality/diversity:
explore-vs-exploit guidance, and "vary your opening word." Since teacher and student share the
prompt, the student inherits the same guidance.

## The generation run

600 distinct random training words (seeded, reproducible), played via batch:

| run  | games | win rate     | cost   |
|------|-------|--------------|--------|
| low  | 400   | 252/400 (63%)| $10.74 |
| high | 200   | 92/200 (46%) | $16.33 |
| **total** | 600 | —          | **$27.07** |

Caveats worth keeping honest in the post:
- Total ran **higher than the ~$19 estimate** — the random train pool skewed toward obscure
  words, so games went more rounds (more tokens) than the 5-word probe implied. *Probes under
  small/easy samples can under-predict.*
- The low-beats-high win rate is **not** an effort comparison — the two runs used disjoint word
  sets, and the high set drew harder words.

## Dataset + release

- Combined **3,078 SFT samples**, 90/10 train/test split.
- ~**85%** of moves carry a `<think>` reasoning trace.
- Pushed to the HuggingFace Hub (public), alongside the raw trajectories and a usage-focused
  dataset card. The Hub is the source of truth — nothing depends on local files to *use* the data.

## Reusable for other games

The pipeline is game-agnostic: the batch driver only talks to a generic agent/env interface,
and each game is one entry in a small registry. Adding the next word game = one registry entry,
same generate → capture → combine → push flow.

## Lessons worth keeping
- Measure cost on *real* usage before scaling; the thinking tokens are the bill.
- `effort` is the dominant cost/quality dial; low is often "good enough" and ~4× cheaper.
- Batch API + low effort compound the savings — if you can tolerate async.
- Treat external batches as durable; make the *client* resumable, not the batch.
- Store raw + multiple completion forms + flags; filter at train time.
- Small-sample cost probes can under-predict; budget headroom.

---

# Part II — From one game to a word-skill curriculum

## Why build more games at all

Wordle teacher data is **expensive**: the run above was **$27 for 600 games**, and the bill is
almost entirely Claude's *thinking* tokens. Scaling Wordle-by-Claude to cover everything the
model is weak at would be costly and slow.

The insight: most of what makes a *good* Wordle player isn't "Wordle" — it's a handful of
**atomic word skills** the model can be taught far more cheaply, and on *related* tasks. So we
built a family of small single-turn games, each isolating one sub-skill, and generate their data
**programmatically** wherever the label is cheap and exact (zero API cost, fully reproducible),
reserving Claude only for the games that genuinely need reasoning.

**The compositional bet (why this helps the later RL phase).** The plan is: SFT on this
curriculum first, then reinforcement-learn (GRPO) on Wordle itself. Wordle play is a
*composition* of smaller skills — map a word to its letters, know which letters are present vs
absent, count length, judge whether a guess is a real word, recall what a word means, rearrange
letters, fit a partial pattern, and above all *read the feedback and not throw away what it told
you*. If the model already **owns these skills as separable, reusable pieces** after SFT, RL
doesn't have to discover them from a sparse win/lose reward — it can **mix and reuse** them while
exploring. SFT installs the parts; RL learns to compose them.

**Vocabulary exposure (you can't play Wordle if you don't know the words).** There's a second,
quieter payoff. A model can't play Wordle well if it doesn't *know English words* — their
spelling, letters, meaning, and sound. These games are how we get broad word exposure in cheaply.
They all draw from one **shared multi-length vocabulary** (the full Wordle vocab unioned with
~73k WordNet words, lengths 3–20), and each game derives its own train/val split with a
**game-salted hash**, so a word held out for one game is trained in another. The deliberate
consequence: nearly every word shows up in *some* game's training data — the model meets the
whole vocabulary (including Wordle's own held-out answers) and learns to spell it, count it,
define it, and sound it out, **without leaking the Wordle-playing skill on eval words**. validity
in particular is sourced straight from the Wordle vocabulary (train+val both), so the model is
taught what every Wordle word *means* and that it's *real* — exactly the background knowledge a
Wordle player takes for granted.

**Raising the multi-turn share (the codebreaker/bullscows/consistency wave).** The single-turn
curriculum teaches the atomic skills, but Wordle's *defining* difficulty is multi-turn: read
several rounds of feedback and adjust without throwing away what you learned. With Wordle as the
only multi-turn game, that loop was just **~5%** of the SFT data. So we added two **multi-turn
deduction games** — **codebreaker** (Mastermind: per-position ✓/-/x feedback over a 4-slot A–F
code) and **bullscows** (4 distinct digits, *count* feedback: bulls + cows) — plus **consistency**
(single-turn: is a candidate word still possible given a Wordle board?). codebreaker and bullscows
deliberately **decouple the feedback-deduction skill from vocabulary** (one uses per-position
tiles, the other uses counts), while consistency and endstart add the clue-filtering and
first/last-letter attention pieces. All three are **programmatic ($0)** — codebreaker/bullscows are
driven by an **unbiased solver** (random opening + a uniformly random code consistent with all
feedback so far; a deterministic/ordered solver would teach a biased policy) replayed through the
same game loop, one SFT row per turn. Together they lift the **multi-turn share of the SFT data
from ~5% to ~24%**, so the student sees far more of the read-feedback-and-refine loop before RL.

**Every game also teaches structured output.** Each one demands a fixed, parseable format —
`<guess>…</guess>`, `<answer>…</answer>`, `<meaning>…</meaning>`, numbered `solution N:` blocks,
`position N, letter X, grey|yellow`, etc. So beyond its skill, every sample reinforces the
model's ability to **emit reliable structured text**. That matters twice over: the RL harness
parses the model's action from these tags, and a malformed reply (or, for the reasoning games, a
missing `<think>` block) is treated as failure — so format discipline is itself part of the
reward surface we're preparing the model for.

## The skills, and which game teaches each

| # | Game | Atomic skill it installs | Producer |
|---|------|--------------------------|----------|
| 0 | wordle | (the RL *target* — composition of all the below) | Claude (reasoning) |
| 1 | charcount | word → its characters: length + vowel/consonant split | programmatic |
| 2 | validity | is this a *real* word? + recall its meaning | programmatic |
| 3 | anagram | letter-multiset reasoning (same letters, rearranged) | Claude (reasoning) |
| 4 | endstart | first/last-character attention (MCQ) | programmatic |
| 5 | rhyme | phonetic / sound mapping | programmatic |
| 6 | crossword | meaning + **partial-pattern → word** (some letters known, fill the rest) | Claude (reasoning) |
| 7 | charset | across several words, **which letters are present vs absent** (a-z) | programmatic |
| 8 | mistakeid | **read ✓/-/x feedback** and spot when a guess repeats a grey/yellow mistake | Claude (reasoning) |
| 9 | tower | **deduce from feedback** under constraints (a non-word feedback puzzle) | programmatic |
| 10 | codebreaker | **parse MULTIPLE turns of per-position feedback and adjust** (Mastermind) | programmatic (multi-turn) |
| 11 | bullscows | **parse MULTIPLE turns of count feedback and adjust** (bulls/cows) | programmatic (multi-turn) |
| 12 | consistency | **is a candidate still possible?** filter words against the clues so far | programmatic |

Read top-to-bottom these are exactly the moves a Wordle player makes: *spell the word*
(charcount), *only guess real words* (validity), *the answer is a rearrangement / shares letters*
(anagram), *match a last letter to a first* (endstart), *I have some green letters, what fits?*
(crossword), *track the used/eliminated letters* (charset), *don't reuse a grey or re-place a
yellow* (mistakeid), *use every clue consistently* (tower), *is this word still in the running?*
(consistency), and — across several turns — *read the feedback and refine the next guess*
(codebreaker, bullscows). crossword/charset/mistakeid/tower/consistency/codebreaker/bullscows are
the most directly Wordle-shaped.

## Two producers, one schema

- **Programmatic (no Claude)** — games 1, 2, 4, 5, 7, 9, 12, and the **multi-turn** 10 & 11. The
  env computes the exact gold answer; a tiny "synthetic teacher" formats it into the completion and
  self-checks it by feeding it back through `step`. **$0**, deterministic, and as much data as we
  want (bounded only by the game's distinct space — e.g. tower's logic space is just 1,920, so we
  vary surface form with random names). For the **multi-turn** games (codebreaker, bullscows) the
  "teacher" is an unbiased solver replayed via the same `run_episode` loop, emitting one row per
  turn; a `--max-rows` cap drops *whole episodes* at a boundary so the per-round distribution stays
  unbiased. Programmatic doesn't have to mean a bare label/guess: the **deduction games**
  (codebreaker, bullscows, consistency) now prefix their `<guess>`/`<answer>` with a short,
  **programmatically-generated "programmed reasoning"** — a true, templated worded rationale
  recapping the clues (codebreaker/bullscows) or the per-clue check (consistency). It's derived
  from the *same* feedback computation as the label and self-checked on every row, so it can never
  teach false reasoning, and it's **not** Claude chain-of-thought (not wrapped in `<think>`, so
  `has_think` stays `False`). For the multi-turn pair the rationale is dropped on replay
  (`build_messages` keeps only the bare `<guess>` of prior turns, exactly as Wordle strips
  `<think>`), so it's a training target without growing the context — prompt + completion stays
  ~400 tokens.
- **Claude-distilled + rejection** — games 3, 6, 8, where *reasoning* is the point. Claude
  produces `<think>…</think><answer>…</answer>`; the env scores it programmatically and we keep
  only traces that are **correct and actually reasoned** (have a `<think>` block). Costs stayed
  small because these are single-turn (no Wordle-style lockstep) and we pick effort per game:
  anagram ≈ **$1.78** (1k @ high), crossword ≈ **$4.96** (1.5k @ high), mistakeid ≈ **$2.78**
  (330 @ max). Note `claude-sonnet-4-6`'s effort levels are **low/medium/high/max** — there is no
  `xhigh`.

The two producers emit the **same unified SFT row schema**, so they combine into one dataset and
the student can't tell which teacher wrote a row. mistakeid and tower are nice examples of
*reusing what we already had*: mistakeid's challenges are mined from the **original Wordle
trajectories** (the real moves where the agent reused a grey or repeated a yellow), and tower is
pure synthetic logic — neither needed new Wordle play.

## A note on the `valid` flag (quality gate ≠ winning)

We initially marked Wordle rows valid iff the *game was won*. That throws away good, well-formed
moves from games that happened to lose. We changed the Wordle gate to **format compliance**:
a row is valid iff its completion has a `<think>` block, regardless of win/loss (this lifted
Wordle's valid rows 1,542 → 2,602). The single-turn games keep a *correctness* gate
(programmatic ones pass by construction; reasoning ones must match ground truth **and** have
reasoned). In every case the gate lives in the row's `valid` column, so it's a training-time
filter, not something baked into generation.

## Where it stands

One combined Hub dataset, **13 games, 96,162 rows** (≈95.5k valid), 90/10 split, **~24%
multi-turn**. Wordle is 3,078 of those; the other ~93k are the cheap-to-make skill curriculum that
cost a few dollars of Claude (three reasoning games) plus **$0** for everything programmatic —
including the two new multi-turn deduction games (codebreaker, bullscows) and the
endstart/consistency clue games. Adding the next game is still *one registry entry + one small
package*, the same generate → capture → combine → push flow.
