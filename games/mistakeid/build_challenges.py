"""Build the committed Mistake-identification challenge asset — ``challenges.jsonl``.

Extracts real boards + proposed guesses from the original Wordle teacher trajectories
(``distillation/data/wordle_sonnet_{low,high}_trajectories.jsonl``). For every episode and every
round ``r >= 2``, the board is rounds ``1..r-1`` (each guess scored against the episode's target)
and the proposed guess is round ``r``'s guess; the label is whether that guess repeated a
grey/yellow mistake. The result is committed so the game reads it with no dependency on the
distillation data — exactly how ``vocab.txt`` / ``meanings.jsonl`` are committed.

    uv run --package game-mistakeid python -m games.mistakeid.build_challenges

Regenerating is a deliberate, run-on-purpose step (it changes which challenges exist).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from games.mistakeid.game import encode_target, score_feedback, true_errors

_DIR = Path(__file__).parent
_OUT = _DIR / "challenges.jsonl"

# The original Wordle teacher rollouts (raw trajectories) are the source of real boards.
_TRAJ = [
    Path("distillation/data/wordle_sonnet_low_trajectories.jsonl"),
    Path("distillation/data/wordle_sonnet_high_trajectories.jsonl"),
]


def _valid(w: str) -> bool:
    return isinstance(w, str) and len(w) == 5 and w.isalpha()


def build(out_path: Path = _OUT, traj_paths: list[Path] = _TRAJ) -> list[dict]:
    """Extract (board, attempt) challenges + mistake labels; write the committed asset."""
    seen: set[str] = set()
    rows: list[dict] = []
    for path in traj_paths:
        if not path.exists():
            raise FileNotFoundError(f"Wordle trajectory file missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            target = ep["target"].lower()
            guesses = [(t.get("action") or "").lower().strip() for t in ep["turns"]]
            for r in range(1, len(guesses)):              # r = index of the proposed guess (0-based)
                attempt = guesses[r]
                prior = guesses[:r]
                if not _valid(attempt) or not all(_valid(g) for g in prior) or not prior:
                    continue
                rounds = [(g, score_feedback(g, target)) for g in prior]
                target_str = encode_target(rounds, attempt)
                if target_str in seen:
                    continue
                seen.add(target_str)
                errs = true_errors(rounds, attempt)
                rows.append({"target": target_str, "mistake": bool(errs), "n_errors": len(errs)})

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_mistake = sum(1 for r in rows if r["mistake"])
    kinds = Counter(r["n_errors"] for r in rows if r["mistake"])
    print(f"wrote {len(rows)} challenges -> {out_path}")
    print(f"  mistake boards: {n_mistake}   clean boards: {len(rows) - n_mistake}")
    print("  errors-per-mistake spread:", {k: kinds[k] for k in sorted(kinds)})
    return rows


if __name__ == "__main__":
    build()
