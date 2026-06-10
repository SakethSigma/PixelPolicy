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
