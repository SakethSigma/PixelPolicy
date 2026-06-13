# Word-skill games — progress log

Live tracker. Built games 2,3,5 first, then 6 (crossword), 7 (charset), 8 (mistakeid),
then 4 (endstart) + 10/11 (codebreaker/bullscows, multi-turn) + 12 (consistency).
Plan: `~/.claude/plans/ok-can-uimplement-game-eventual-thimble.md`.

## 🔄 Update 6 — endstart(4) + multi-turn feedback games + consistency(12)
- **endstart (4, programmatic)**: 6,000 rows. Last of the original six. 15 tests.
- **codebreaker (10, MULTI-TURN programmatic)**: Mastermind 4×A–F, ✓/-/x dup-correct feedback.
  Unbiased random-consistent solver as teacher (random opening, uniform consistent pick). 16 tests.
- **bullscows (11, MULTI-TURN programmatic)**: 4 distinct digits, count feedback (bulls/cows).
  Unbiased solver. 16 tests.
- **consistency (12, programmatic)**: Wordle board + candidate → yes/no still-possible, reusing
  `compute_feedback`. Balanced 5k/5k, <4k tokens. 17 tests.
- **New infra**: programmatic multi-turn generator (`_play_multiturn` reuses `run_episode` + solver
  `generate`, one row/turn). `--max-rows` cap drops WHOLE episodes (round distribution unbiased).
- codebreaker/bullscows capped at **10,000 rows each** (whole episodes; per user). Round spreads
  natural (cb 1→6, bc 1→8).
- Re-pushed (overwrite): **96,162 rows** (train 86,545 / test 9,617), 95,520 valid, 13 games,
  **multi-turn share ~5% → ~24%**. All 348 tests green ($0 — all programmatic).
- Docs for the 4 new games: docs-maintainer (background).

## 🔄 Update 4 — games 7 (charset) + 8 (mistakeid) shipped
- **charset (7, programmatic)**: 12,000 rows, all valid. 20 tests. (built earlier this turn)
- **mistakeid (8, Claude reasoning, require_think)**: identify repeated grey/yellow mistakes in a
  proposed Wordle guess. Self-contained via committed `games/mistakeid/challenges.jsonl` (built
  from the original wordle trajectories: 165 mistake + 1498 clean boards). 22 tests.
  - Note: `xhigh` is NOT a valid sonnet-4-6 effort (low/medium/high/max) — the user's "ultra"/xhigh
    pick errored ($0); substituted **max**. 330 episodes (165+165 balanced) → **317 valid**
    (157 mistake + 160 clean, 13 wrong rejected). ~$2.78.
- Re-pushed (overwrite): **55,162 rows** (train 49,645 / test 5,517), **53,460 valid**. 8 games:
  wordle 3,078 · charcount 14,000 · validity 13,254 · anagram 1,000 (932) · rhyme 10,000 ·
  crossword 1,500 (1,415) · charset 12,000 · mistakeid 330 (317).
- Total Claude spend across the session ≈ $9.7 (within the $10 cap). All 163 tests green.
- Docs for charset + mistakeid: docs-maintainer launched (background).

## 🧭 RESTART GUIDE (read these if context was lost)
Adding a word-skill game = mirror an existing one. Reference docs/code:
- `games/CODE_IMPLEMENTATION.md` + `games/DATA_SOURCING.md` — the design + game numbering.
- Reference impls to copy: `games/charcount/` (programmatic single-turn), `games/crossword/`
  (Claude-distilled reasoning + committed asset), `games/mistakeid/` (reasoning + committed
  challenges.jsonl built from data).
- Per-game wiring (the ONLY places a game is added):
  - `games/<g>/` package: game.py/render.py/client.py/server.py/play.py/__init__.py/pyproject.toml/tests/
  - `agents/<g>/agent.py` (+`__init__.py`): `<G>Agent` + `<G>Env`.
  - `distillation/registry.py`: `_<g>_spec()` + `GAMES` + `GAME_NUMBERS` (wordle0 charcount1
    validity2 anagram3 [endstart4 unbuilt] rhyme5 crossword6 charset7 mistakeid8 **tower9**).
    `GameSpec(make_agent, make_env, sample_targets, max_rounds=1, good_status="correct",
    require_think=<True for reasoning games>)`.
  - `distillation/programmatic.py`: `--game` branch (programmatic games only) + `_DEFAULT_N/_OUT`.
  - `distillation/batch_play.py`: game-agnostic Claude runner (`--game <g> --effort high|max`);
    valid = solved AND (not require_think or has `<think>`).
  - `distillation/push.py`: `DEFAULT_INPUTS` + `_STEM_GAME` (`<g>_sft` → (`<g>`, no)).
  - `agents/pyproject.toml` + `distillation/pyproject.toml`: add `game-<g>` dep + `[tool.uv.sources]`.
- Commands: `uv sync`; tests `uv run --package game-<g> pytest games/<g>/tests/ -q`;
  programmatic `uv run --package distillation python -m distillation.programmatic --game <g>`;
  Claude batch `... -m distillation.batch_play --game <g> --episodes N --model claude-sonnet-4-6 --effort high`;
  push `... -m distillation.push --test-size 0.1 --overwrite` (HF_TOKEN+HF_HUB_REPO_ID in .env).
- Sonnet-4-6 effort levels: **low/medium/high/max** (NO xhigh). Dataset: saketh-chervu/word-games-distillation.

## ▶ IN PROGRESS — game 9 "tower" + wordle valid fix (todos #1-#3 in task list)
- **tower (9, programmatic)**: 3 floors × 2 rooms, 3 people one-per-floor. Shown placement + per-person
  feedback (floor ✓/x, room ✓/x). Model lists ALL placements consistent with feedback.
  - MATH CONFIRMED by enumeration: solutions ∈ {1,2}; **1536 challenges have 1 sol, 384 have 2**
    (2-sol ⟺ all floors wrong = the 2 derangements). Rooms always uniquely deducible.
    **Distinct challenge space = 1920** (so 5000 distinct impossible — cap 1920; pending user: 1920
    unique vs ~5000 via random names).
- **wordle valid fix**: change gate from won→has_think in push.py. Verified: wordle valid 1542→2602
  (476 no-think rows invalid; lost-but-thought moves now kept). batch_low think=1547/2023, batch_high all think.

## Status legend
⬜ not started · 🟡 in progress · ✅ done · ❌ blocked/error

## Checklist

### Shared
- ⬜ `games/wordvocab/build_meanings.py` + `load_meanings()` + committed `meanings.jsonl`
- ⬜ `uv sync` after new packages/deps

### Game 2 — validity
- ⬜ `games/validity/` package (game/render/client/server/play/pyproject/tests)
- ⬜ `agents/validity/agent.py` (+ `__init__`)
- ⬜ registry + GAME_NUMBERS entry (validity=2)
- ⬜ programmatic generator (`--game validity`, 14k 50/50)

### Game 3 — anagram
- ⬜ `games/anagram/` package
- ⬜ `agents/anagram/agent.py` (no "sort" in prompt)
- ⬜ registry + GAME_NUMBERS entry (anagram=3)
- ⬜ batch_play run (claude-sonnet-4-6 low, 2000 ep, 40/60)

### Game 5 — rhyme
- ⬜ `games/rhyme/` package
- ⬜ `agents/rhyme/agent.py` (MCQ + free)
- ⬜ registry + GAME_NUMBERS entry (rhyme=5)
- ⬜ programmatic generator (`--game rhyme`, 10k = 5k MCQ + 5k free)

### Wiring / deps / docs
- ⬜ `agents/pyproject.toml` + `distillation/pyproject.toml` deps & sources
- ⬜ `distillation/push.py` DEFAULT_INPUTS + _STEM_GAME
- ⬜ docs (per-game READMEs; flip DATA_SOURCING/CODE_IMPLEMENTATION to built; root + distillation README counts)

### Generate + push
- ⬜ tests pass per package
- ⬜ programmatic data generated (validity, rhyme)
- ⬜ anagram batch data generated (paid)
- ⬜ push --dry-run reviewed
- ⬜ push to Hub

## Status update (mid-build)
All three packages + agents + registry + generators + wiring written and `uv sync` clean.
- Tests: rhyme 24 ✅, anagram 21 ✅, validity 21 ✅ (+ charcount 29, agents 7 still green).
- Meanings asset built: 79,564 defs (92% of vocab). Wordle∩meanings = 6,627.
- **validity**: 13,254 rows (6,627 valid + 6,627 invalid). Balanced (user choice). 500/500 train==inference + correct.
- **rhyme**: 10,000 rows (5k MCQ + 5k free), full-vocab seeds. Fixed hyphen parse bug + alpha-only options.
- **anagram**: 2000 episodes via Batch API (claude-sonnet-4-6, low, $0.64). 1977 correct / 23 wrong.
  - At low effort only 28/2000 produced <think>. Per user: keep the no-think direct answers,
    rewrote the system prompt to ask for a DIRECT answer, and mark the 28 think-traces invalid
    (GameSpec.reject_think). Re-derived in place → **1949 valid**, 51 invalid. train==inference clean.
- Sourcing confirmed for user: anagram + rhyme draw from full 85,909-word vocab (general words,
  not Wordle-only); only validity is Wordle-restricted (intended).
- Dry-run push: 42,332 rows (40,745 valid) — wordle 3078, charcount 14000, validity 13254,
  anagram 2000, rhyme 10000. Existing dataset already carries valid=False rows, so consistent.

## ✅ COMPLETE
All three games built, tested, generated, pushed, and documented.
- Tests: charcount 29 · validity 21 · anagram 21 · rhyme 24 · agents 7 — all green.
- Dataset pushed: https://huggingface.co/datasets/saketh-chervu/word-games-distillation
  — **42,332 rows** (train 38,098 / test 4,234), **40,745 valid**. Per game: wordle 3,078,
  charcount 14,000, validity 13,254, anagram 2,000 (1,949 valid), rhyme 10,000.
- Game-3 spend: ~$0.64 (Batch API, claude-sonnet-4-6, low effort).
- Docs updated (docs-maintainer): 3 new game READMEs + wordvocab/DATA_SOURCING/CODE_IMPLEMENTATION/
  root README/distillation README. `new_game.md` (this file) intentionally left as the tracker.

## 🔄 Update 2 — anagram redone at high effort + game 6 (crossword) added
- **anagram** regenerated at **high adaptive-thinking effort** (claude-sonnet-4-6): reverted the
  prompt to "think it through", flag back to `require_think` (valid = correct AND has-think).
  1000 episodes → all correct, **932 valid** (68 no-think dropped). ~$1.78. Original 2000-row
  low-effort anagram discarded/overwritten.
- **crossword (game 6)** built (Claude-distilled, reasoning, `require_think`): clue = WordNet def
  + length + deterministic masked pattern (~half hidden); exact-match-to-seed-word rejection.
  Seeds 750 Wordle + 750 general (varied length). 1500 episodes → 1415 correct, **1415 valid**.
  ~$4.96. Tests 19 ✅.
- Budget: user cap $10 → chose anagram 1000 + crossword 1500; actual spend ≈ **$6.8** (+ ~$0.08 probes).
- Re-pushed (overwrite): **42,832 rows** (train 38,548 / test 4,284), **41,143 valid**. Per game:
  wordle 3,078, charcount 14,000, validity 13,254, anagram 1,000 (932), rhyme 10,000, crossword 1,500 (1,415).
- Final tests: charcount 29 · validity 21 · anagram 21 · rhyme 24 · crossword 19 · agents 7 — all green.
- Docs update for crossword + anagram-change launched (docs-maintainer, background).

## 🔄 Update 3 — game 7 (charset) built + game-8 mistake analysis
- **charset (game 7, programmatic)**: "given a few words, list used + unused letters of a-z".
  Each challenge = 1 Wordle word + 1-3 non-5-letter words. Built (20 tests ✅), generated
  **12,000 rows** (all valid). game_no 7. Wired into registry/push/programmatic/deps. NOT pushed
  yet (bundling with game 8).
- **Game 8 (identify-the-error) prep**: counted repeated-mistake moves in the ORIGINAL wordle
  trajectories (feedback computed from each episode's target):
  - LOW (no-reasoning, 400 ep, 1488 follow-up guesses): grey-reuse **72**, yellow-repeat **107**,
    either **174** (5 both).
  - HIGH (reasoning, 200 ep, 610 guesses): grey-reuse 5, yellow-repeat 12, either 17.
  - Defs: grey-reuse = guess uses a known-absent letter; yellow-repeat = letter re-placed in a
    slot already shown yellow. Awaiting user's game-8 prompt/design + budget.
- Dry-run (charset incl.): 54,832 rows (53,143 valid). All 141 tests green.

## Log
- (start) plan approved; created tracker.
- Wrote all code: wordvocab `build_meanings.py`; `games/{rhyme,validity,anagram}/` full packages
  (game/render/client/server/play/__init__/pyproject/tests); `agents/{rhyme,validity,anagram}/agent.py`
  (+__init__); registry specs + GAME_NUMBERS (validity=2, anagram=3, rhyme=5); generalized
  `programmatic.py` (`--game charcount|validity|rhyme`); `push.py` inputs+stems (skips missing
  files); deps in agents/ + distillation/ pyproject. Next: uv sync, build meanings, run tests.
