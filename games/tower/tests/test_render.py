"""Tests for the dependency-free Tower rendering."""

from __future__ import annotations

from games.tower.game import TowerGame
from games.tower.render import render_observation, render_solutions

NAMES = ["Alice", "Bob", "Carol"]


def test_observation_shows_guess_and_feedback():
    g = TowerGame(NAMES, [2, 1, 3], [0, 1, 0], [False, True, False], [True, False, False], "id")
    text = render_observation(g.state())
    assert "Alice — guess: floor 2, Left  ->  floor x, room ✓" in text
    assert "Bob — guess: floor 1, Right  ->  floor ✓, room x" in text
    assert "no two people share a floor" in text


def test_render_solutions_numbered():
    g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [False, False, False], [True, True, True], "id")
    text = render_solutions(g.solution_placements())
    assert "solution 1:" in text and "solution 2:" in text
    assert text.count("Alice: floor") == 2     # Alice appears in both solutions
