"""Tests for the dependency-free Rhymes rendering."""

from __future__ import annotations

from games.rhyme.game import RhymeGame
from games.rhyme.render import render_answer, render_observation


def test_free_observation():
    state = RhymeGame("bright", "id", variant="free").state()
    assert render_observation(state) == 'Name a word that rhymes with "bright".'


def test_mcq_observation_lists_options():
    opts = ["flight", "table", "garden", "purple", "ocean"]
    state = RhymeGame("bright", "id", variant="mcq", options=opts).state()
    text = render_observation(state)
    assert 'rhymes with "bright"' in text
    for o in opts:
        assert o in text


def test_render_answer_lowercases():
    assert render_answer("  FLIGHT ") == "flight"
