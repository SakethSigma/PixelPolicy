"""Tests for the dependency-free Bulls & Cows rendering."""

from __future__ import annotations

from games.bullscows.game import BullsCowsGame
from games.bullscows.render import render_observation, render_round


def test_render_round():
    g = BullsCowsGame("1243", "t")
    g.guess("1234")
    assert render_round(g.state().rounds[-1]) == "1234  ->  bulls: 2, cows: 2"


def test_render_observation():
    g = BullsCowsGame("1234", "t")
    text = render_observation(g.state())
    assert "4-digit" in text and "bulls" in text and "cows" in text
