"""Plain-text rendering of a Rhymes ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown, so the two can never drift. :func:`render_answer` is the
canonical ``<answer>`` block the synthetic teacher writes for the SFT completion.
"""

from __future__ import annotations

from games.rhyme.game import GameState


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read."""
    if state.variant == "mcq":
        opts = ", ".join(state.options or [])
        return f'Which of these words rhymes with "{state.word}"?\nOptions: {opts}'
    return f'Name a word that rhymes with "{state.word}".'


def render_answer(word: str) -> str:
    """The canonical ``<answer>`` block body — just the chosen/answer word, lowercase."""
    return word.strip().lower()
