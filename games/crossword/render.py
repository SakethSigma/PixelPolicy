"""Plain-text rendering of a Crossword-fill ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown, so the two can never drift. The pattern is spaced out
(``c _ a _ e``) so revealed and hidden positions are easy to read.
"""

from __future__ import annotations

from games.crossword.game import GameState


def render_observation(state: GameState) -> str:
    """The clue a human and the model read: definition, length, and the masked pattern."""
    pattern = " ".join(state.pattern)
    return (
        f'Definition: "{state.definition}"\n'
        f"Length: {state.length}\n"
        f"Pattern: {pattern}"
    )


def render_answer(word: str) -> str:
    """The canonical ``<answer>`` body — just the solved word, lowercase."""
    return word.strip().lower()
