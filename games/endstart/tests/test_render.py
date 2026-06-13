"""Tests for the dependency-free Ends-with → starts-with rendering."""

from __future__ import annotations

from games.endstart.game import GameState
from games.endstart.render import render_observation


def test_observation_lists_options():
    state = GameState(game_id="x", word1="mango", options=["river", "oasis", "tundra", "cliff", "marsh"])
    text = render_observation(state)
    assert 'word1 = "mango"' in text
    for o in state.options:
        assert o in text
