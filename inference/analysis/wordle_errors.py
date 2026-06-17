# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.7"]
# ///
"""Wordle ERROR-TYPE metrics across checkpoints — computed from raw predictions, no re-inference.

Four error classes per guess (turn), tracked as a % of all guesses, across epochs and model
variants (e.g. wordle-only vs full-v2):

  1. no_guess_tag   — model didn't emit a parseable `<guess>…</guess>` (parse_action → "").
                      (we also report no_think = response lacks a closing </think>.)
  2. bad_length     — a guess was emitted but isn't 5 letters.
  3. out_of_vocab   — 5 letters but not an allowed word (matches WordBank.is_valid: train ∪ val).
  4. clue_violation — a VALID guess that ignores accumulated feedback: reuses a known-absent (gray)
                      letter [grey_reuse], abandons a known green position, or drops a known-present
                      (green/yellow) letter. Round-1 and invalid guesses can't violate → counted 0.

Reads `<results_dir>/raw/<label>/wordle.jsonl` (records from inference/evaluate.py::_episode_record:
fields `target`, `n_turns`, `turns=[{round, response, action}]`). Vocab + feedback logic mirror
`games/wordle/game.py` (kept inline so this runs standalone with `--no-project`, like viz_eval.py).

Usage:
  uv run --no-project inference/analysis/wordle_errors.py \
    --results eval_results_v2 eval_results_fullv2 --out wordle_error_metrics
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
_GUESS_TAG = re.compile(r"<guess>\s*([A-Za-z]+)\s*</guess>", re.IGNORECASE | re.DOTALL)


def _load_vocab() -> set[str]:
    """Allowed-guess set = train ∪ val, matching games/wordle/game.py::WordBank.is_valid."""
    vocab: set[str] = set()
    for fn in ("train_words.txt", "val_words.txt"):
        p = _REPO / "games" / "wordle" / fn
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w:
                vocab.add(w)
    return vocab


def _feedback(guess: str, target: str) -> list[str]:
    """Per-letter Wordle feedback with duplicate handling — copy of games/wordle/game.py.
    Returns 'G' (green/correct), 'Y' (yellow/wrong-pos), 'X' (gray/absent)."""
    guess, target = guess.upper(), target.upper()
    fb: list[str | None] = [None] * len(guess)
    remaining = Counter(target)
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            fb[i] = "G"
            remaining[g] -= 1
    for i, g in enumerate(guess):
        if fb[i] is not None:
            continue
        if remaining[g] > 0:
            fb[i] = "Y"
            remaining[g] -= 1
        else:
            fb[i] = "X"
    return fb  # type: ignore[return-value]


def _parse_guess(turn: dict) -> str:
    """The guess the env saw: prefer the stored parsed `action`, else re-parse the response."""
    a = (turn.get("action") or "").strip().lower()
    if a:
        return a
    m = _GUESS_TAG.findall(turn.get("response") or "")
    return m[-1].lower() if m else ""


def _label_metrics(jsonl: Path, vocab: set[str]) -> dict:
    """Classify every guess in one checkpoint's wordle.jsonl."""
    turns = no_guess = no_think = bad_len = oov = clue_viol = grey_reuse = 0
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        target = (rec.get("target") or "").upper()
        # accumulated constraints from prior VALID guesses this episode
        greens: dict[int, str] = {}
        present: set[str] = set()
        seen_absent: set[str] = set()
        for t in rec.get("turns", []):
            turns += 1
            resp = t.get("response") or ""
            if "</think>" not in resp:
                no_think += 1
            g = _parse_guess(t)
            if not g:
                no_guess += 1
                continue
            if len(g) != 5 or not g.isalpha():
                bad_len += 1
                continue
            if g not in vocab:
                oov += 1
                continue
            # valid 5-letter, in-vocab guess → check it against accumulated clues
            absent = seen_absent - present
            gu = g.upper()
            v_grey = any(ch in absent for ch in gu)
            v_green = any(gu[p] != ch for p, ch in greens.items())
            v_present = any(ch not in gu for ch in present)
            if greens or present or absent:           # only meaningful once clues exist
                if v_grey or v_green or v_present:
                    clue_viol += 1
                if v_grey:
                    grey_reuse += 1
            # update constraints with this guess's true feedback
            fb = _feedback(g, target)
            for i, (ch, f) in enumerate(zip(gu, fb)):
                if f == "G":
                    greens[i] = ch; present.add(ch)
                elif f == "Y":
                    present.add(ch)
                elif f == "X":
                    seen_absent.add(ch)
    pct = lambda n: round(100 * n / turns, 1) if turns else 0.0
    return {"turns": turns, "no_guess_tag": pct(no_guess), "no_think": pct(no_think),
            "bad_length": pct(bad_len), "out_of_vocab": pct(oov),
            "clue_violation": pct(clue_viol), "grey_reuse": pct(grey_reuse)}


def _family_epoch(label: str) -> tuple[str, int]:
    """'full-v2-e3' -> ('full-v2', 3); non-conforming -> (label, 0)."""
    m = re.match(r"^(.*)-e(\d+)$", label)
    return (m.group(1), int(m.group(2))) if m else (label, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wordle error-type metrics from raw, across checkpoints.")
    ap.add_argument("--results", nargs="+", required=True,
                    help="one or more eval dirs (each with raw/<label>/wordle.jsonl).")
    ap.add_argument("--out", default="wordle_error_metrics", help="output basename (.md/.png/.svg).")
    args = ap.parse_args()

    vocab = _load_vocab()
    rows: dict[str, dict] = {}                          # label -> metrics (first dir wins on dup labels)
    for d in args.results:
        for jsonl in sorted(Path(d).glob("raw/*/wordle.jsonl")):
            label = jsonl.parent.name
            if label not in rows and jsonl.stat().st_size > 0:
                rows[label] = _label_metrics(jsonl, vocab)

    if not rows:
        raise SystemExit("no raw/*/wordle.jsonl found under: " + ", ".join(args.results))

    order = sorted(rows, key=_family_epoch)
    cols = ["no_guess_tag", "no_think", "bad_length", "out_of_vocab", "clue_violation", "grey_reuse"]
    # --- markdown table ---
    lines = ["# Wordle error-type metrics (% of all guesses)", "",
             "| checkpoint | turns | " + " | ".join(cols) + " |",
             "|---|---|" + "|".join(["---"] * len(cols)) + "|"]
    for lbl in order:
        m = rows[lbl]
        lines.append(f"| {lbl} | {m['turns']} | " + " | ".join(f"{m[c]}" for c in cols) + " |")
    table = "\n".join(lines) + "\n"
    Path(f"{args.out}.md").write_text(table, encoding="utf-8")
    print(table)

    # --- plot: one subplot per error type, a line per model family across epochs ---
    families = sorted({_family_epoch(l)[0] for l in order})
    plot_cols = ["no_guess_tag", "bad_length", "out_of_vocab", "clue_violation"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, plot_cols):
        for fam in families:
            pts = sorted((_family_epoch(l)[1], rows[l][col]) for l in order if _family_epoch(l)[0] == fam)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=fam)
        ax.set_title(col); ax.set_xlabel("epoch"); ax.set_ylabel("% of guesses")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Wordle error types across epochs", fontsize=14)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=130, bbox_inches="tight")
    print(f"[wordle-errors] wrote {args.out}.md / .png / .svg")


if __name__ == "__main__":
    main()
