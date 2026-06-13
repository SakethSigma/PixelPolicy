"""Plain-text rendering of an Ends-with → starts-with ``GameState``.

Dependency-free on purpose: this is the text a human reads and the observation a model is shown.
"""

from __future__ import annotations

from games.endstart.game import GameState


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read."""
    opts = ", ".join(state.options)
    return (
        f'word1 = "{state.word1}". Which of these words starts with the last letter of '
        f'"{state.word1}"?\nOptions: {opts}'
    )


def render_answer(word: str) -> str:
    """The canonical ``<answer>`` body — the chosen word, lowercase."""
    return word.strip().lower()
