"""Tests for the dependency-free Crossword rendering."""

from __future__ import annotations

from games.crossword.game import CrosswordGame
from games.crossword.render import render_observation


def test_observation_shows_clue_and_masked_pattern():
    g = CrosswordGame("crane", "a large wading bird", "id")
    text = render_observation(g.state())
    assert 'Definition: "a large wading bird"' in text
    assert "Length: 5" in text
    assert "Pattern: " in text
    assert "_" in text                 # at least one hidden position is shown
    assert "crane" not in text         # the full word is never shown in the clue
