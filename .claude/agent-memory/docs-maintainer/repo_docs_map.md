---
name: repo-docs-map
description: PixelPolicy in-scope Markdown docs, their purpose, plan-* files to skip, and shared style conventions
metadata:
  type: reference
---

PixelPolicy is a uv-workspace research framework (games / agents / training / inference / distillation).

## In-scope docs and their purpose
- `README.md` (root) — overview, repo-structure tree, setup, quickstart per game, distillation example, "Adding a New Game".
- `games/DATA_SOURCING.md` — DESIGN doc: six single-turn word-skill games (numbered 1–6), ground-truth sources, two production modes, shared vocab asset. charcount = game #1.
- `games/CODE_IMPLEMENTATION.md` — DESIGN doc: how the word-skill games plug into repo layers. Pairs with DATA_SOURCING.md.
- `games/wordle/README.md` — Wordle env (multi-turn reference game). Very detailed.
- `games/charcount/README.md` — charcount env (single-turn, word-skill game #1). Authored by user; verified accurate.
- `games/wordvocab/README.md` — shared vocab support package (not a game, no server).
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
- DATA_SOURCING "Coverage guarantee" bullet describes verifying union of all games' train sets + forcing stragglers. The built `wordvocab.assign_pool` is purely deterministic with NO committed per-game artifact or straggler pass — that guarantee is a multi-game aspiration, only charcount exists. Left as design intent.
