"""Pure scoring for the evaluation harness — no IO, no network.

Turns a list of played `Trajectory`s into per-game and aggregate metrics. A game is "solved" when
its terminal status equals the game's `good_status` (`"won"` for the multi-turn deduction games,
`"correct"` for the single-turn games — see `distillation/registry.py`). Accuracy here IS win-rate
for the multi-turn games.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

# The three multi-turn deduction games (the rest are single-turn).
MULTI_TURN: frozenset[str] = frozenset({"wordle", "codebreaker", "bullscows"})
# The games whose completions carry real chain-of-thought (<think>) — mirrors
# training/sft/format.py::REASONING_GAMES.
REASONING: frozenset[str] = frozenset({"wordle", "anagram", "crossword", "mistakeid"})


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a proportion k/n (good at small n / extreme p)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_trajectory(traj: Any, good_status: str) -> dict:
    """One played episode → {solved, rounds, n_turns, n_action_ok, n_think}."""
    turns = traj.turns
    n_turns = len(turns)
    return {
        "solved": getattr(traj.final, "status", None) == good_status,
        "rounds": n_turns,                                            # turns played this episode
        "n_turns": n_turns,
        "n_action_ok": sum(1 for t in turns if t.action),            # parsed a non-empty action
        "n_think": sum(1 for t in turns if "<think>" in (t.response or "")),
    }


def game_metrics(trajs: list, good_status: str) -> dict:
    """Aggregate one game's trajectories into its metrics."""
    n = len(trajs)
    per = [score_trajectory(t, good_status) for t in trajs]
    solved = sum(1 for p in per if p["solved"])
    accuracy = solved / n if n else 0.0
    ci_lo, ci_hi = wilson_interval(solved, n)

    won_rounds = [p["rounds"] for p in per if p["solved"]]
    total_turns = sum(p["n_turns"] for p in per) or 1
    return {
        "n": n,
        "solved": solved,
        "accuracy": accuracy,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        # multi-turn only (single-turn games are always 1 round): kept for all, meaningful for wordle etc.
        "avg_rounds_to_win": (sum(won_rounds) / len(won_rounds)) if won_rounds else None,
        "solved_by_round": dict(sorted(Counter(won_rounds).items())),
        # format discipline
        "action_parse_rate": sum(p["n_action_ok"] for p in per) / total_turns,
        "think_rate": sum(p["n_think"] for p in per) / total_turns,
    }


def aggregate(per_game: dict[str, dict]) -> dict:
    """Cross-game roll-ups for one checkpoint (macro-averages over games)."""
    def macro(names) -> float | None:
        accs = [per_game[g]["accuracy"] for g in names if g in per_game]
        return (sum(accs) / len(accs)) if accs else None

    games = list(per_game)
    return {
        "macro_accuracy": macro(games),
        "single_turn_acc": macro([g for g in games if g not in MULTI_TURN]),
        "multi_turn_acc": macro([g for g in games if g in MULTI_TURN]),
        "reasoning_acc": macro([g for g in games if g in REASONING]),
    }
