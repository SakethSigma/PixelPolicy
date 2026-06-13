"""Plain-text rendering of a Character-set ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown. :func:`render_answer` is the canonical ``<answer>`` block —
the synthetic teacher writes the SFT completion with it, and :func:`games.charset.game.parse_answer`
reads it back.
"""

from __future__ import annotations

from games.charset.game import GameState, analyze


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read: the list of words to scan."""
    return f"Words: {', '.join(state.words)}"


def render_answer(used: list[str], unused: list[str]) -> str:
    """The canonical ``<answer>`` block body for the used/unused letter sets.

    Letters are space-separated and UPPERCASE. Example for words {cat, dog}::

        used (5): A C D G O T
        unused (21): B E F H I J K L M N P Q R S U V W X Y Z
    """
    u = " ".join(c.upper() for c in used) if used else "-"
    n = " ".join(c.upper() for c in unused) if unused else "-"
    return f"used ({len(used)}): {u}\nunused ({len(unused)}): {n}"


def render_solution(words: list[str]) -> str:
    """Convenience: the canonical answer block for ``words`` (used by play.py)."""
    used, unused = analyze(words)
    return render_answer(used, unused)
