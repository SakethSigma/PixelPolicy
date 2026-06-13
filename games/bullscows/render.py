"""Plain-text rendering of a Bulls & Cows ``GameState`` (dependency-free)."""

from __future__ import annotations

from games.bullscows.game import GameState, RoundResult


def render_round(r: RoundResult) -> str:
    """One scored row, e.g. ``1234  ->  bulls: 1, cows: 2`` (or an invalid note)."""
    if r.error is not None:
        return f"{r.guess}  ->  [invalid: {r.error} — counted as a round]"
    return f"{r.guess}  ->  bulls: {r.bulls}, cows: {r.cows}"


def render_observation(state: GameState) -> str:
    """The opening challenge (rules + feedback meaning)."""
    return (
        f"Guess the secret {state.n_digits}-digit number. All {state.n_digits} digits are "
        "different (digits 0-9).\n"
        "After each guess you are told: bulls = digits that are correct and in the right place; "
        "cows = digits that are in the number but in the wrong place.\n"
        "Make your first guess."
    )
