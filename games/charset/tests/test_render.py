"""Tests for the dependency-free Character-set rendering."""

from __future__ import annotations

from games.charset.game import GameState, analyze, parse_answer
from games.charset.render import render_answer, render_observation


def test_observation():
    state = GameState(game_id="x", words=["cat", "planet"])
    assert render_observation(state) == "Words: cat, planet"


def test_render_answer_shape():
    used, unused = analyze(["cat", "dog"])
    text = render_answer(used, unused)
    assert text.startswith("used (6): A C D G O T")
    assert "unused (20): " in text


def test_empty_unused_sentinel():
    # A pangram-ish set leaves no unused letters -> "-" sentinel parses back to empty.
    used = list("abcdefghijklmnopqrstuvwxyz")
    text = render_answer(used, [])
    p_used, p_unused = parse_answer(text)
    assert p_unused == set()
