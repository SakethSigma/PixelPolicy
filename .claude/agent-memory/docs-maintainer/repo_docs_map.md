---
name: repo-docs-map
description: PixelPolicy in-scope Markdown docs, their purpose, plan-* files to skip, and shared style conventions
metadata:
  type: reference
---

PixelPolicy is a uv-workspace research framework (games / agents / training / inference / distillation).

## In-scope docs and their purpose
- `README.md` (root) — overview, repo-structure tree, setup, quickstart per game, distillation example, "Adding a New Game".
- `games/DATA_SOURCING.md` — DESIGN doc: word/deduction games. Original family #1-6, extended by #7 charset, #8 mistakeid, #9 tower, #10 codebreaker, #11 bullscows, #12 consistency. Status now COMPLETE — ALL 12 built (endstart #4 was the last of the original six). #10/#11 are MULTI-TURN. Documents two production modes (now incl programmatic multi-turn), shared vocab asset, Wordle's valid=format gate. Summary table + per-game sections kept in sync.
- `games/CODE_IMPLEMENTATION.md` — DESIGN doc: how the games plug into repo layers. Pairs with DATA_SOURCING.md. Status COMPLETE (all 12 built). Has a multi-turn note (codebreaker/bullscows mirror Wordle's guess verb), the programmatic multi-turn generator (_play_multiturn + --max-rows). Build-order step 4 carries the Hub row count (96,162) — keep in sync with dataset memory. Has "Wordle's gate is format, not correctness" note.
- `games/tower/README.md` — tower deduction env (game #9, programmatic, no Claude, no require_think). 3 floors x 2 rooms, 3 people one-per-floor; deduce all placements (1 or 2) from Wordle-style ✓/x feedback. Pure Python, NO vocab dep. game-tower; 5,000 SFT rows.
- `games/endstart/README.md` — endstart env (game #4, programmatic, single-turn MCQ). NEW 2026-06-13. word + 5 candidates, pick one starting with word's last letter. Needs vocab. game-endstart; 6,000 rows; 15 tests. charcount house style.
- `games/codebreaker/README.md` — codebreaker/Mastermind env (game #10, programmatic, MULTI-TURN). NEW 2026-06-13. 4 slots A-F, per-pos ✓/-/x (Wordle dup rule), max_rounds 12, won. Unbiased CodebreakerSolver teacher. NO vocab dep. game-codebreaker; 10,000 rows; 16 tests. Wordle (multi-turn) house style.
- `games/bullscows/README.md` — bulls&cows env (game #11, programmatic, MULTI-TURN). NEW 2026-06-13. 4 distinct digits, COUNT feedback (bulls+cows), max_rounds 10, won. Unbiased solver. NO vocab dep. game-bullscows; 10,000 rows; 16 tests. Wordle house style.
- `games/consistency/README.md` — consistency env (game #12, programmatic, single-turn yes/no). NEW 2026-06-13. Wordle board + candidate → still possible? Reuses Wordle compute_feedback (dep game-wordle). Balanced 50/50, <4k tokens. Complement to mistakeid. game-consistency; 10,000 rows; 17 tests. charcount house style.
- `games/charset/README.md` — charset env (game #7, programmatic, no Claude). Used/unused letters of a-z across 2-4 words (1 five-letter Wordle word + non-five-letter). Salted `charset` split; needs vocab.txt.
- `games/mistakeid/README.md` — mistakeid env (game #8, Claude-distilled HIGH-EFFORT REASONING at `max`, `require_think`). Wordle board + proposed guess → identify repeated grey/yellow mistakes. Self-contained: reads committed challenges.jsonl (NO wordvocab/distillation runtime dep, NO target word).
- `games/wordle/README.md` — Wordle env (multi-turn reference game). Very detailed.
- `games/charcount/README.md` — charcount env (single-turn, word-skill game #1). Authored by user; verified accurate.
- `games/validity/README.md` — validity+meaning env (game #2, programmatic). Draws from Wordle vocab (not full vocab); oracle = committed meanings.jsonl.
- `games/anagram/README.md` — anagram env (game #3, Claude-distilled HIGH-EFFORT REASONING `<think>`+yes/no, `require_think`). Full multi-length vocab. (Was REDONE from direct/low-effort.)
- `games/rhyme/README.md` — rhyme env (game #5, programmatic, MCQ+free). Full multi-length vocab; pronouncing/CMU dict runtime dep.
- `games/crossword/README.md` — crossword env (game #6, Claude-distilled HIGH-EFFORT REASONING, `require_think`). Clue=WordNet def+length+~half-masked pattern (deterministic per seed word). Needs vocab.txt + meanings.jsonl, no nltk runtime. Seeds half Wordle-vocab + half general.
- `games/wordvocab/README.md` — shared vocab support package (not a game, no server). Now also documents meanings.jsonl + build_meanings.py.
- `distillation/README.md` — distillation pipeline, unified SFT schema table, two producers, Files table.
- `agents/Readme.md` (note: capital R, lowercase rest) — generic "how to write an agent" guide, Wordle as worked example.
- `agents/wordle/README.md`, `agents/training_integration.md`, `agents/wordle/running_inference.md`, `inference/README.md`, `distillation/batch_play.md`, `distillation/blog_notes.md` — supporting docs.

## SKIP (plan-* prefix and PLAN.md) — never edit
- `PLAN.md`, `distillation/PLAN.md` are uppercase PLAN.md (NOT `plan-*`). The agent's exclusion rule targets the `plan-*` prefix; these are arguably out of the strict `plan-*` glob. Treat PLAN.md files as design/owner docs — leave untouched unless explicitly asked, and note suggested changes in summary instead.

## Style conventions
- Tone is dense, technical, opinionated; uses `> ` blockquotes for caveats/footguns and a "Status:" blockquote near the top of design docs.
- File-map sections use fenced ``` trees with inline `# comments`; module Files use Markdown tables with `[file](file)` links.
- Code fences tagged `bash`, `python`, `jsonc`. Em-dashes and ✓/✅ markers used freely.
- Package names: `game-wordle`, `game-charcount`, `game-wordvocab`; run via `uv run --package <pkg> ...`.
- Per-game env layout mirrored everywhere: `game.py` (pure core) / `server.py` / `client.py` / `render.py` / `play.py` / `tests/`.

## Known gap (design-vs-built)
- DATA_SOURCING "Coverage guarantee" bullet describes verifying union of all games' train sets + forcing stragglers. The built `wordvocab.assign_pool` is purely deterministic with NO committed per-game artifact or straggler pass — still a multi-game aspiration. Left as design intent. (Validity is a deliberate exception: it draws from Wordle vocab, not the full vocab, and exposes all valid words rather than only its train pool.)
- new_game.md is a live progress tracker — do NOT edit (per-user instruction). PLAN.md / distillation/PLAN.md also off-limits.
