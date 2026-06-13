"""Tests for the dependency-free Validity rendering / parse round-trip."""

from __future__ import annotations

from games.validity.game import GameState, parse_answer
from games.validity.render import render_answer, render_observation


def test_observation():
    state = GameState(game_id="x", word="planet")
    assert render_observation(state) == "Word: planet"


def test_valid_round_trip():
    text = render_answer(True, "a definition")
    assert parse_answer(text) == (True, "a definition")


def test_invalid_round_trip():
    text = render_answer(False)
    assert text == "<answer>invalid</answer>"
    assert parse_answer(text) == (False, "")
