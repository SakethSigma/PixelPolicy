"""Plain-text rendering of a Mistake-identification ``GameState``.

Dependency-free on purpose: the board is shown exactly as a Wordle player sees it (the same
``✓ - x`` tiles), then the proposed guess to review. The per-symbol legend lives in the agent's
system prompt, so the observation is just the board + the proposed guess.
"""

from __future__ import annotations

from games.mistakeid.game import GameState

_SYM = {"g": "✓", "y": "-", "x": "x"}


def _row(guess: str, feedback: str) -> str:
    letters = " ".join(c.upper() for c in guess)
    syms = " ".join(_SYM.get(f, "?") for f in feedback)
    return f"{letters}   {syms}"


def render_observation(state: GameState) -> str:
    """The board a human and the model read, plus the proposed next guess."""
    board = "\n".join(_row(g, f) for g, f in state.rounds)
    proposed = " ".join(c.upper() for c in state.attempt)
    return f"Guesses so far:\n{board}\n\nProposed next guess: {proposed}"
