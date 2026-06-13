"""Tests for the pure Crossword-fill logic in ``games.crossword.game``."""

from __future__ import annotations

import pytest

from games.crossword.game import (
    CrosswordGame,
    GameOverError,
    make_pattern,
    matches_pattern,
    parse_answer,
)


class TestPattern:
    def test_deterministic(self):
        assert make_pattern("crane") == make_pattern("crane")

    def test_has_revealed_and_hidden(self):
        p = make_pattern("crane")
        assert len(p) == 5
        assert "_" in p                       # at least one hidden
        assert any(c != "_" for c in p)       # at least one revealed

    def test_matches_pattern(self):
        p = make_pattern("crane")
        assert matches_pattern("crane", p)
        assert matches_pattern("plane", "p_a_e")       # plane fits the revealed letters
        assert not matches_pattern("crane", "x____")   # first letter conflicts


class TestParseAnswer:
    def test_tag_body(self):
        assert parse_answer("<answer>crane</answer>") == "crane"

    def test_no_tag_last_word(self):
        assert parse_answer("the answer is crane") == "crane"

    def test_empty(self):
        assert parse_answer("12345") is None


class TestCrosswordGame:
    def test_correct_exact_match(self):
        g = CrosswordGame("crane", "a large long-necked wading bird", "id1")
        assert g.state().status == "in_progress"
        assert g.state().solution is None
        state = g.step("<answer>crane</answer>")
        assert state.status == "correct"
        assert state.solution.word == "crane"

    def test_wrong_word_incorrect(self):
        g = CrosswordGame("crane", "a bird", "id2")
        assert g.step("<answer>plane</answer>").status == "incorrect"

    def test_unparseable_incorrect(self):
        g = CrosswordGame("crane", "a bird", "id3")
        assert g.step("hmm 123").status == "incorrect"

    def test_state_hides_word_in_progress(self):
        # The in-progress observation must not leak the target word: GameState has no `word`
        # field, and the solution (which reveals it) is None until the episode ends.
        g = CrosswordGame("crane", "a bird", "id4")
        dumped = g.state().model_dump()
        assert "word" not in dumped
        assert dumped["solution"] is None

    def test_step_after_end_raises(self):
        g = CrosswordGame("crane", "a bird", "id5")
        g.step("<answer>crane</answer>")
        with pytest.raises(GameOverError):
            g.step("crane")
