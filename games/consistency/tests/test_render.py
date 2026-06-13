"""Tests for the dependency-free Candidate-consistency rendering."""

from __future__ import annotations

from games.consistency.game import ConsistencyGame, feedback_str, is_consistent
from games.consistency.render import render_observation, render_reasoning


def test_observation_shows_board_and_candidate():
    rows = [("CRANE", feedback_str("crane", "plant"))]
    g = ConsistencyGame(rows, "plant", "id")
    text = render_observation(g.state())
    assert "C R A N E" in text
    assert 'Is the word "PLANT" still possible' in text


def test_reasoning_consistent():
    rows = [("CRANE", feedback_str("crane", "plant"))]
    assert is_consistent("plant", rows)
    text = render_reasoning(rows, "plant")
    assert "still possible" in text and "ruled out" not in text


def test_reasoning_inconsistent_pinpoints():
    rows = [("CRANE", feedback_str("crane", "plant"))]
    assert not is_consistent("doggy", rows)
    text = render_reasoning(rows, "doggy")
    assert "ruled out" in text and "position 3 must be A" in text
