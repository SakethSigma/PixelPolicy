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

---

# Part III — Training the student, and evaluating the checkpoints

## Three recipes, one harness

With the dataset built, the question is *how* to feed it to the 0.8B student. We don't want to guess
— we want to **compare**, so the trainer (HuggingFace TRL `SFTTrainer`) runs the exact same way for
three recipes that differ only in *which rows, in what order*:

1. **wordle-only** — train on just the 2,602 valid Wordle moves. The baseline: how far does
   single-game imitation get you, and what does it cost on everything else?
2. **full, shuffled** — all ~95.5k valid rows, standard random order. The "just throw the curriculum
   at it" control.
3. **full, curriculum** — all rows, but introduced **easy→hard** (see below).

Every recipe trains **4 epochs** and pushes **each epoch's checkpoint to the Hub as its own
revision** (`epoch-1..4`). That's deliberate: the training box is ephemeral (rented GPU, no git), so
the Hub is the *only* output channel, and keeping every epoch lets us watch the model **over training
time**, not just at the end. We train on the completion only (the prompt is masked) so the loss is
about *what the model should say*, never the prompt we already fed it — byte-identical to inference.

## The curriculum bet (and the forgetting worry)

The interesting recipe is #3. The naïve version — sort everything strictly easy→hard and feed it
once — is exactly the version the literature warns against. Across 4 epochs a *fixed* easy→hard order
re-floods the model with trivial single-turn data at the **start of every epoch**, right after it
last saw the hard reasoning data. Small models are the worst case for this: multi-task interference
and reasoning fragility both peak below ~1B params, and only ~5.5% of our data carries real
chain-of-thought, so the `<think>` skill is easy to drown.

So our curriculum is **competence-*widening*, not blocked**: harder games become *eligible*
progressively, but once introduced a game stays in the shuffled mix; the four reasoning games are
kept present **throughout** (a "reasoning floor"); and a small **replay** slice of easy+reasoning data
is spliced into the tail. The hypothesis we're actually testing: *strict* easy→hard is the arm most
likely to erode reasoning, plain shuffle is a strong baseline, and the widening+replay variant is the
one with a real chance of beating shuffle without forgetting. (Full design, hypotheses, and citations
in `training/CURRICULUM_NOTES.md`.)

## The tooling tax (lessons we paid for, again)

Most of the friction wasn't the ML — it was the plumbing. Worth a paragraph because everyone hits it:

- **Large vocab → cross-entropy OOM.** Qwen3.5 has a ~248k-token vocabulary, so the loss logits are
  `batch × seq × 248k` in fp32 — ~15 GB *just for the loss* at batch 16, which OOM'd a 48 GB card
  before training even started. The fix is **chunked cross-entropy** (`loss_type="chunked_nll"`):
  identical math, computed in chunks so the full logits never materialize. (TRL is making it the
  default — heed the deprecation warning.) Lesson: with a big-vocab model, the *loss*, not the
  weights, is your memory wall.
- **torch vs. the driver.** The default PyPI torch now targets CUDA 13, which won't initialize on the
  CUDA-12.8 drivers cloud GPU hosts ship ("driver too old"). Pinning torch via an index in
  `pyproject` quietly broke the bundled CUDA libs (`libcudnn.so.9` missing). What actually worked:
  install the matching `cu128` wheel manually and stop the package manager from re-resolving it.
  Lesson: pin the *wheel*, verify `torch.cuda.is_available()` before anything else, and don't trust a
  green install.
- **Cross-machine tracking.** Runs land on different rented boxes, so metrics go to **wandb** (cloud,
  one project) rather than local files — you compare `wordle` / `full` / `curriculum` side by side
  regardless of which machine produced them.

## A peek at *where* it learns

A cheap aside that pays off in intuition: a training callback logs the **per-layer gradient and
update norms** (and, per block, attention vs. MLP) to wandb every N steps. Pulled back and drawn as a
**layer × step heatmap**, it shows *where* in the 24-layer stack the optimizer is actually moving
weight, and how that front shifts across epochs and recipes. Early read (noisy, one run): the lower
layers barely move while the **middle/upper-middle** blocks do most of the adapting — consistent with
"lower layers are general/done from pretraining, task-specific work happens higher." We treat this as
intuition-building, not a result. (See `training/LEARNING_DYNAMICS_NOTES.md`.)

## Evaluating the checkpoints — the metric that matters

Training loss is **not** the scorecard. What we care about is whether the model can *play*. So each
checkpoint is evaluated **behaviorally**: host it locally (vLLM) and have it actually play a fixed,
seeded held-out test set of **all 13 games** — 300 instances each, the *same* instances for every
checkpoint, frozen sampling (`temp 0.6`, thinking on). A game counts as solved when the env's own
ground truth says so (`won` for the deduction games, `correct` for the single-turn ones), and we
report accuracy / win-rate with a **Wilson 95% CI** so a 52%-vs-55% wiggle isn't over-read. The whole
thing reuses the existing game registry — one generic loop drives all 13 games — and the base
(untrained) model is evaluated too, as the reference line.

The questions this is built to answer (results to drop in once the run completes):
- Does the wordle-only model's **win rate climb** across its 4 epochs — and does it **overfit** (peak
  then dip) by epoch 4?
- **Transfer / interference:** a wordle-only SFT never saw the other 12 games — does playing Wordle
  *help* or *hurt* the related word skills vs. the base model? (The base→best **delta-per-game** plot
  is the headline here.)
- Eventually, the three-way comparison: does **full** beat **wordle-only** on transfer, and does the
  **curriculum** protect reasoning (the `<think>` games) better than plain shuffle — i.e. do the
  forgetting hypotheses hold?

*(Numbers + figures — the game×checkpoint heatmap, per-game curves, the Wordle win-rate-by-epoch, and
the transfer delta — go here after the eval run.)*

## Lessons worth keeping (Part III)
- Keep **every epoch's checkpoint** (push to the Hub); "the model over training" is far more
  informative than the final weights, and it costs nothing.
- **Compare recipes, don't guess** — wordle-only / full / curriculum share one trainer so the only
  variable is the data.
- For a small model on a big-vocab base, the **loss** is the memory wall → chunked cross-entropy.
- **Behavioral eval, fixed seeded test set, base as reference, CIs on everything** — loss curves lie
  about whether the model can actually play.
- Strict easy→hard curriculum is a *cautionary* arm, not the default — protect the fragile reasoning
  skill (keep it interspersed + a little replay).

---

# Part III — Watching *where* the model learns (grad norm vs update norm)

> Working notes for explaining the learning-dynamics plots to a general reader. Goal: make the two
> numbers we log per layer (`gradnorm`, `updnorm`) understandable without jargon, then use them to
> ask *where in the network* fine-tuning actually changes things.

## The model is just a big pile of numbers, organized into floors

A language model is ~800 million numbers called **weights**. Training does one thing, over and over:
nudge each number a little so the next prediction is better. Those weights are stacked into **24
layers** — think of them as 24 floors of a building. The input text enters at floor 0 (the bottom),
gets processed floor by floor, and the answer comes out at floor 23 (the top). Each floor holds
roughly the same number of weights.

## One training step is two moves: the *push*, then the *actual move*

**Move 1 — the push.** For every weight, the math computes one number: "to make the model better,
change this weight by about *this much*, in *this direction*." That per-weight number is the
**gradient**. It's a wish list — pure "what would help," with no decision yet about how far to go.

**Move 2 — the actual move.** An optimizer (Adam) reads the wish list and decides how far to
*really* nudge each weight, multiplying everything by a tiny **learning rate** (e.g. 0.00002). So the
real moves are far smaller than the wishes, and Adam reshapes them too. Then the weights take their
new values.

So per weight there are two things worth watching: **the push it got** and **how far it actually
moved** (new value minus old value).

## "Norm" just means "the overall size of a bunch of numbers"

A floor has millions of weights, so millions of push-numbers. We want **one** summary number per
floor, not millions. That's the **norm**, and the recipe is mechanical:

> square every number, add them all up, take the square root.

Tiny example — pretend a floor had only 3 weights whose pushes were `0.02, 0.01, -0.03`:

```
norm = sqrt(0.02² + 0.01² + 0.03²) = sqrt(0.0014) = 0.037
```

That `0.037` is the floor's **gradnorm**: "the overall size of the push on this floor." Big = shoved
hard; small = barely touched.

## The two numbers we log

- **`gradnorm` (the push / the *wish*)** — square-sum-sqrt over all of a floor's gradients. *How
  hard the loss wanted to change this floor this step.* Crucially, it has **no learning rate in it** —
  it's measured before the optimizer scales anything, so it's the pure learning *pressure*. (One
  honest footnote: it's read just after gradient *clipping*, a safety cap on the total push, so it's
  "the push that actually drives the step," minus only that cap.)
- **`updnorm` (the actual move)** — square-sum-sqrt over how far each of a floor's weights *actually*
  moved (new − old). *How much this floor really changed this step.* This one **does** include the
  learning rate and everything Adam did.

They differ because of Move 2. Continuing the example: pushes `0.02, 0.01, -0.03` give gradnorm
`0.037`, but after Adam the weights only moved by `-0.001, -0.0008, 0.0015` → updnorm `0.002`. **Push
0.037, actual move 0.002** — the move is tiny because the learning rate is tiny. **`updnorm` is the
honest "did this part of the model change" signal; `gradnorm` is "how badly did it want to."**

## The one trap: these numbers are not divided by anything

The recipe adds **one term per weight**, so a part with *more* weights gets a bigger number
automatically — even if each individual weight moved the same amount. Practical rules:

- **Floor-to-floor is a *mostly* fair comparison.** The 24 floors have ~the same weight count, so a
  higher number usually means that floor moved more — but with one caveat we hit below: every 4th
  floor is a *different kind* of floor, so it's only an apples-to-apples comparison within a type.
- **The word-lookup table (`embed`) is not comparable to a floor.** It has ~150 million numbers vs a
  floor's few million, so its norm looks huge mostly because there's more to add up — *not* because
  each weight there learns more. Only compare equal-sized things.

## What the two views tell you

- **Across floors** (lay floors 0→23 left to right): *which depth of the network is moving, and which
  is sitting still.* A peak at floor 13 literally means floor 13's weights changed more than the
  others'.
- **Across time** (step 50, 100, 150…): *is the movement dying down as the model learns, and do
  different floors move early vs. late?* A floor whose `updnorm` shrinks is settling; one that stays
  high near the end is still being rewritten.

**The payoff question:** the push and the actual move often peak at *different* floors. Raw push tends
to pile up at the edges (the first floor near the input, the last floor near the answer), while the
actual movement concentrates in the upper-middle floors. That gap — wanted-to-move vs. did-move — is
the interesting story, and it's the whole reason we log both instead of just one.

## What we actually saw (first runs)

We log these two numbers every **50 optimizer steps** (not once per epoch) for all 24 blocks, plus the
embedding and the final norm. So far we have two of three planned runs: **`wordle`** (one game,
finished, small — only ~106 steps, so just 2 logged points) and **`full`** (all 13 games, still
training, hundreds of steps). `curriculum` isn't logged yet. The wordle curve is a *hint* from 2
points; `full` is where to trust a shape.

### Finding 1 — fine-tuning barely touches the bottom of the network
Reading the *actual move* (`updnorm`) across floors: the **input-side blocks (0–7) move least**,
movement **rises through the middle, and the upper-middle blocks (~8–18) move most**, easing slightly
at the very top. Plain reading: the lower floors already hold generic spelling/word machinery from
pretraining that transfers to the word games unchanged, so the optimizer leaves them alone; the
task-specific work (read ✓/–/x feedback, narrow candidates, emit the format) is composed in the
upper-middle stack, so that's where the weights actually move. This shows in both runs — it's the
headline.

### Finding 2 — where the model *wants* to change ≠ where it *does* change
The push (`gradnorm`) and the move (`updnorm`) peak at **different** floors. The push piles up at the
two **edges** — block 0 (against the input) and especially block 23 (against the output/loss, where
the error signal is most direct). The movement piles up in the **upper-middle**. The sharpest case:
the **last block has the biggest push but nearly the smallest move** — it wants to change the most,
and Adam moves it the least. Why: Adam divides each weight's step by a running estimate of that
weight's own gradient size, so a big push gets normalized back down — a large gradient does *not* buy
a proportionally large step. (A safety cap, gradient clipping, also pins the whole-model push to a
fixed size each step, so per-block push is really a *share* of a fixed budget.) Lesson: for "where did
the model actually change," read `updnorm`, not `gradnorm`.

### Finding 3 — the moves collapse at the very end because of the schedule, not because Adam "gives up"
In the `wordle` heatmap the final logged column goes dark — every block's move shrinks at once. Cause:
the **cosine learning-rate schedule** ramps the learning rate to ~zero by the end, and since every
move is multiplied by the learning rate, all moves shrink together. We can rule out the tempting "the
gradients went noisy and cancelled out" story: the **push was still at full strength** there (still
hitting the clipping cap), so the gradients hadn't weakened — only the step size applied to them had.
(This just says moves vanish in the final handful of steps *by design*; expected, not a discovery.)

### Finding 4 — the repeating 4-layer "bands" are the architecture, not learning
The most eye-catching pattern is a ripple with period 4: every 4th block (indices **3, 7, 11, 15, 19,
23**) moves a bit *less* than its neighbors, slicing the plot into bands. It's **real** (identical
indices in both runs) but it's **not about the task or about depth** — it's the base model. **Qwen3.5-
0.8B is a hybrid-attention model**: 3 of every 4 blocks use cheap **linear attention** (an
RNN/state-space-style mixer with fixed memory), and every 4th block uses the original **full (softmax)
attention** (all-pairs, exact recall, expensive). The config's `full_attention_interval: 4` puts full
attention at exactly 3, 7, 11, … — the dip indices. (It's done to stay cheap at long context: linear
attention is light, the periodic full-attention layers restore exact long-range recall.)

The tell that it's an **optimizer effect, not a learning one**: the ripple is in `updnorm` (the move)
but **absent in `gradnorm`** (the push) — if anything the full-attention blocks *push harder*. They
want to move more, but Adam moves them less, because their different internals give them different
gradient statistics and Adam's per-weight normalization reacts to that. Two different machines,
measured with one ruler. **Consequence:** "this block moved more than that one" is only fair *within a
block type*; in the figures we mark the every-4th (full-attention) layers so the bands aren't misread
as a depth trend. The slow envelope (Finding 1) is the trustworthy depth signal; the 4-layer ripple
riding on it is the architecture seam.

**And that optimizer behavior is a feature, not a bug.** Note *where* the bands come from: the raw
push (`gradnorm`) is smooth — no 4-layer pattern — so the periodicity is created entirely by Adam
turning push into move. The reason it does this is the whole point of Adam: it divides each weight's
step by a running estimate of that weight's own gradient size, which **equalizes progress across parts
of the network that naturally have very different gradient scales.** Without it (plain SGD), the
loudest-gradient layers — like the full-attention blocks, or the last block against the loss — would
dominate every step and starve everything else; *that* would be skewed learning. So Adam taking
smaller steps where gradients are larger/noisier isn't damage, it's calibrated caution, and it matters
*more* for a hybrid model whose two block types have genuinely different gradient geometry. The honest
limit: a block moving less is **not** evidence it's under-trained — move size says nothing about
whether a move helped (only loss/eval can), and the gap here is only ~10%, not a starvation.

### What would sharpen this next
- A run that logs **attention and MLP movement separately** (not done yet): every block's MLP is
  identical and only the attention differs, so the prediction is the ripple should **vanish in the MLP
  view and sharpen in the attention view** — a clean confirmation the bands are the attention seam.
- The **`curriculum`** run, for the order/forgetting comparison it was built for.
- These norms are *absolute* move size, not move *relative to* a block's existing weights — a relative
  view would be a useful addition.
