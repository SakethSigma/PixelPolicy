"""Plain-text rendering of an Anagrams ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown, so the two can never drift.
"""

from __future__ import annotations

from games.anagram.game import GameState


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read."""
    return f"Are '{state.word1}' and '{state.word2}' anagrams of each other?"


def render_answer(yes: bool) -> str:
    """The canonical ``<answer>`` body — ``yes`` or ``no``."""
    return "yes" if yes else "no"
