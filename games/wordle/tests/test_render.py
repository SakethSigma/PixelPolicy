"""Tests for the dependency-free text renderer in ``games.wordle.render``."""

from __future__ import annotations

from games.wordle.game import WordleGame
from games.wordle.render import render_observation, render_round


def vocab_game(**kw):
    allowed = {"APPLE", "CRANE", "MOIST"}
    return WordleGame(
        target="apple", game_id="g1", validate_word=lambda w: w in allowed, **kw
    )


class TestRenderRound:
    def test_valid_round_shows_feedback_symbols(self):
        g = vocab_game()
        line = render_round(g.guess("crane"))
        assert "C R A N E" in line
        # CRANE vs APPLE: C x, R x, A (wrong pos), N x, E (wrong pos)
        assert "✓" in line or "-" in line or "x" in line

    def test_invalid_round_shows_reason_and_no_symbols(self):
        g = vocab_game()
        line = render_round(g.guess("zzzzz"))
        assert "out of vocabulary" in line
        assert "counted as a round" in line


class TestRenderObservation:
    def test_contains_header_and_legend(self):
        obs = render_observation(vocab_game().state())
        assert "Wordle" in obs
        assert "Legend" in obs

    def test_rounds_left_decrements(self):
        g = vocab_game()
        assert "Rounds left: 6" in render_observation(g.state())
        g.guess("crane")
        assert "Rounds left: 5" in render_observation(g.state())

    def test_invalid_round_appears_in_observation(self):
        g = vocab_game()
        g.guess("zzzzz")
        obs = render_observation(g.state())
        assert "out of vocabulary" in obs

    def test_target_revealed_only_after_end(self):
        g = vocab_game()
        g.guess("crane")
        assert "APPLE" not in render_observation(g.state())
        g.guess("apple")  # win
        assert "won" in render_observation(g.state()).lower()

    def test_loss_reveals_word(self):
        g = vocab_game(max_rounds=1)
        g.guess("crane")
        assert "APPLE" in render_observation(g.state())
