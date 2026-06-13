"""Plain-text rendering of a Codebreaker ``GameState``.

Dependency-free: the same per-position tile layout Wordle uses, so the feedback the model reads
is the familiar ✓/-/x.
"""

from __future__ import annotations

from games.codebreaker.game import GameState, RoundResult


def render_round(r: RoundResult) -> str:
    """One scored row, e.g. ``A C E F   ✓ x - x`` (or an invalid note)."""
    symbols = " ".join(r.guess)
    if r.feedback:
        return f"{symbols}   {' '.join(r.feedback)}"
    return f"{symbols}   [invalid: {r.error} — counted as a round]"


def render_observation(state: GameState) -> str:
    """The opening challenge a human and the model read (the rules + symbol set)."""
    return (
        f"Crack the secret code: {state.code_length} slots, each one of {len(state.symbols)} "
        f"symbols ({' '.join(state.symbols)}); symbols can repeat.\n"
        "After each guess every slot is scored: ✓ right symbol right slot, - right symbol wrong "
        "slot, x symbol not in the code.\n"
        "Make your first guess."
    )
