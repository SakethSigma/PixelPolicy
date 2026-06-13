"""Tests for the dependency-free Anagrams rendering."""

from __future__ import annotations

from games.anagram.game import GameState
from games.anagram.render import render_answer, render_observation


def test_observation():
    state = GameState(game_id="x", word1="listen", word2="silent")
    assert render_observation(state) == "Are 'listen' and 'silent' anagrams of each other?"


def test_render_answer():
    assert render_answer(True) == "yes"
    assert render_answer(False) == "no"
