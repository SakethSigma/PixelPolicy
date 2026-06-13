"""Tests for the dependency-free Codebreaker rendering."""

from __future__ import annotations

from games.codebreaker.game import CodebreakerGame
from games.codebreaker.render import render_observation, render_round


def test_render_round_tiles():
    g = CodebreakerGame("ACEF", "t")
    g.guess("AAEB")
    assert render_round(g.state().rounds[-1]) == "A A E B   ✓ x ✓ x"


def test_render_observation_lists_rules():
    g = CodebreakerGame("ACEF", "t")
    text = render_observation(g.state())
    assert "4 slots" in text and "A B C D E F" in text and "✓" in text
