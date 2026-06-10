# Lockstep Batch Wordle Pipeline + Dual-Format Data Capture

> Design doc for `distillation/batch_play.py`. Kept in-repo (alongside [PLAN.md](PLAN.md)) so the
> design survives across sessions. Status: **planned, not yet implemented.**

## Context

We're building a Claude→student distillation dataset for Wordle. Live generation
(`run_eval` / `distillation/cost_probe.py`) works but is expensive. Measured on Sonnet 4.6, 5 games:

| effort | output tokens | total | per game | →500 games | solved |
|---|---|---|---|---|---|
| high | 44,229 | $0.69 | $0.138 | ~$69 | 3/5 |
| low  | 10,655 | $0.19 | $0.038 | ~$19 | 4/5 |

(Same 5 words `GLEYS,METHS,ANOMY,BULLS,MONER`. Cost is dominated by output/thinking tokens; effort=low is
~¼ the cost at comparable accuracy — but N=5 is too small to trust the accuracy delta.)

The next lever is the **Anthropic Batch API (~50% off)**, which stacks with low effort to roughly halve
the already-cheap low-effort cost (~$0.019/game).

**The catch:** a batch is a bag of *independent one-shot* requests, but Wordle is multi-turn — guess N
needs feedback N-1, so a whole game can't be one batch. The fix is **lockstep batching across games**:
batch all N games' round-1 prompts → apply each env's feedback locally → batch all still-active games'
round-2 prompts → … up to 6 rounds. N games take ≤6 batches instead of 6×N live calls.

Goal: write that pipeline, verify it works + the cost on a single Sonnet game, and store data in **two
formats** — (1) raw Claude JSON (full response + usage) and (2) processed exploded-per-move SFT samples
carrying `system`, `messages` (prompt), and the completion in **both** full-`<think>` and think-stripped
forms.

## Files

- **New:** `distillation/batch_play.py` — the **game-agnostic** lockstep batch driver + dual-format
  writer + cost. Knows nothing about Wordle; drives whatever `--game` resolves to via the registry.
- **New:** `distillation/registry.py` — `GameSpec` + `GAMES`. The **only** place a new game is added.
  A `GameSpec` is `(make_agent, make_env(target), sample_targets(n, mode, rng), max_rounds)`; `GAMES`
  maps a name to a zero-arg factory that loads shared resources once and returns the spec. Wordle is
  the first entry; a new word game = one more entry, no driver changes. Targets are sampled with a
  caller-supplied seeded `random.Random` (reproducible, distinct).
- **Edit:** `agents/wordle/agent.py` — add a 1–2 line explore/exploit note to `WordleAgent.system_prompt`
  (the **main/shared** agent prompt, so teacher now and student later both get it, and it flows into the
  stored `system` field). Insert one bullet in the Rules list, e.g.:
  *"- With several guesses left but many words still consistent with the clues, it can pay to explore
  untried letters rather than commit to one likely answer — eliminating possibilities faster improves your
  odds of winning."* Keep the existing bullet style/voice.
- **Reused unchanged:**
  - `AnthropicBackend.batch_generate(prompts, *, poll_interval, **sampling)` — `agents/backend.py`;
    submits one batch, polls to `ended`, returns `Completion`s **in input order**; `completion.raw` =
    full Claude `model_dump()` incl. `usage`.
  - `WordleAgent.build_messages(state, history)` + `parse_action(text)` — `agents/wordle/agent.py`.
    `build_messages` already **strips `<think>` from prior-turn history** (replays only `<guess>`), so
    stored prompts are think-clean automatically.
  - `Turn(messages, response, action, state)` — `agents/base.py`.
  - `WordBank`, `LocalWordleClient` (`games/wordle/`), `WordleEnv` (`agents/wordle/agent.py`).
  - Cost/safety patterns from `distillation/cost_probe.py`: `PRICING` table, and
    `with_options(timeout=90, max_retries=0)` — a retried in-flight request is **billed twice** (this
    caused an earlier ~$1 wedge when a stalled connection auto-retried).

## Design — `distillation/batch_play.py`

CLI (argparse, mirrors `cost_probe.py`): `--episodes` (default 1), `--model` (default `claude-sonnet-4-6`),
`--effort` (default `low`), `--mode` (train/val), `--words` (pin targets, comma-sep), `--poll` (poll secs,
default 15), `--out-raw` (`distillation/data/batch_raw.json`), `--out-sft` (`distillation/data/batch_sft.jsonl`).

**Setup:** one shared `WordleAgent()`; for each game build `WordleEnv(LocalWordleClient(bank))` and
`env.reset(mode, word=...)`. Track per game: `{env, history: list[Turn], turns: [...], target, status}`.
Apply `backend._client = backend._client.with_options(timeout=90.0, max_retries=0)`.

**Lockstep loop** (`for r in range(6)`):
1. `active = [g for g in games if g.status == "in_progress"]`; `break` if empty.
2. `prompts = [agent.build_messages(g.env.state(), g.history) for g in active]`.
3. `completions = backend.batch_generate(prompts, poll_interval=args.poll)` — **one batch for the whole round**.
4. `for g, prompt, comp in zip(active, prompts, completions)`: `action = parse_action(comp.text)`;
   `state = g.env.step(action)`; append `Turn(prompt, comp.text, action, state)` to history; record the
   per-turn row (below); `g.status = state.status`.
5. `print(f"round {r+1}: batched {len(active)} games -> {wins}/{losses}/{still active}")` (flush via `-u`).

**Two outputs:**
- **Raw** → `batch_raw.json`: `{model, effort, batch: true, system, cost, per_game_usd,
  projected_500_games_usd, games: [{episode, target, status, turns: [{round, input: prompt[1:],
  output: comp.text, action, usage, raw: comp.raw}]}]}`. `input` = messages after the system turn;
  `raw` = full Claude dump.
- **Processed SFT** → `batch_sft.jsonl`, one line per move:
  `{game, round, target, system, messages: <full build_messages list, byte-identical inference prompt>,
  completion: comp.text, completion_no_think: strip_think(comp.text)}`.
  - `strip_think(t)` = `re.sub(r"<think>.*?</think>\s*", "", t, flags=re.DOTALL)` → turns
    `"<think>…</think>\n<guess>crane</guess>"` into `"<guess>crane</guess>"`; no-op if no think block.

**Cost:** sum `usage` across all turns (`input_tokens`, `output_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens`); price with `PRICING[model]` × **0.5 (batch discount)**. Print/store batch
total, per-game, and ×500 projection; also print the live-equivalent (un-halved) for reference.

## Resilience (network loss / crash)

A submitted batch is **durable on Anthropic's side** — it runs to completion server-side (≤24h) and
results are kept ~29 days. A dropped connection only kills the local poller, not the batch. So:

- **Blip tolerance:** `batch_generate`'s poll + results reads are idempotent GETs wrapped in a backoff
  retry (`_resilient`) — a transient drop pauses polling and continues when the net returns; it does **not**
  crash. The `create` POST is *not* retried (a retried create could spawn a duplicate batch → double cost).
- **Checkpoint every round:** `batch_play` writes `--checkpoint` (default `distillation/data/batch_state.json`)
  after each round, and — via `on_created(batch_id)` — the instant a batch is created, *before* the
  failure-prone poll. Checkpoint holds run config, next round index, the in-flight `{batch_id, active}`, and
  per-game `turns`.
- **`--resume <checkpoint>`:** rebuilds games by resetting each env to its target and **replaying the saved
  guesses** (deterministic, local, free) to restore env state + think-stripped history; if an in-flight
  batch id is present, re-attaches via `resume_batch_id` (fetches its results, **no re-submit → no double
  cost**) and continues from the next round.
- **Orphan recovery:** a batch that was in-flight during a crash *before* we persisted its id is still
  listable via `client.messages.batches.list()` for 29 days — its results (already billed) can be fetched
  manually if needed.

## Notes / gotchas
- **Single-game batch is slow, not cheap-fast.** Batching 1 game gives no cross-game parallelism — it's
  ≤6 sequential batches (one per round), each paying batch queue latency (usually minutes, SLA 24h). It
  proves correctness; the throughput/cost win only appears at large N. Also try `--episodes 3`.
- **Caching is moot here:** system prompt (~250 tokens, even with the new bullet) is below Sonnet 4.6's
  2048-token cache minimum, so `cache_*` stays 0 in batch and live alike.
- **Failed batch request** → `Completion(text="", finish_reason="errored")`; `parse_action("")=""` → env
  counts a consumed round with an error. Handle gracefully (shows as `action: ""`).

## Verification
1. Single Sonnet game through the batch path (writes both files), backgrounded + a process-exit waiter
   (batch rounds take minutes):
   ```bash
   uv run --package distillation python -u -m distillation.batch_play --episodes 1 --effort low
   ```
2. **Correctness:** `batch_raw.json` has 1 game, populated `turns` (status won/lost, ≤6 rounds), real
   `usage` + full `raw` per turn.
3. **Processed format:** `batch_sft.jsonl` has one line per round; `messages` is the exact inference prompt
   (history think-stripped), `completion` keeps `<think>…</think>`, `completion_no_think` has **no**
   `<think>` (assert `"<think>" not in completion_no_think`).
4. **Cost:** batch per-game ≈ half the live low-effort figure (~$0.038 → ~$0.019); printed batch total is
   0.5× the un-halved live-equivalent it also prints.
5. (Optional) `--episodes 3 --words GLEYS,METHS,ANOMY` → round-1 line reads "batched 3 games".
