"""Tests for the dependency-free Mistake-identification rendering."""

from __future__ import annotations

from games.mistakeid.game import MistakeGame
from games.mistakeid.render import render_observation


def test_observation_shows_board_and_proposed():
    g = MistakeGame([("crane", "xxxxx"), ("slate", "xxgxg")], "track", "id")
    text = render_observation(g.state())
    assert "Guesses so far:" in text
    assert "C R A N E   x x x x x" in text
    assert "S L A T E   x x ✓ x ✓" in text
    assert "Proposed next guess: T R A C K" in text
