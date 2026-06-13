"""Plain-text rendering of a Validity ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown. :func:`render_answer` is the canonical completion the
synthetic teacher writes — and exactly what :func:`games.validity.game.parse_answer` reads back.
"""

from __future__ import annotations

from typing import Optional

from games.validity.game import GameState


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read: just the word to judge."""
    return f"Word: {state.word}"


def render_answer(valid: bool, meaning: Optional[str] = None) -> str:
    """The canonical answer completion: a verdict, plus a ``<meaning>`` block when valid.

    Examples::

        <answer>valid</answer>
        <meaning>a fire that has been kindled or is burning</meaning>

        <answer>invalid</answer>
    """
    if valid:
        return f"<answer>valid</answer>\n<meaning>{(meaning or '').strip()}</meaning>"
    return "<answer>invalid</answer>"
