"""Tests for the pure Candidate-consistency logic."""

from __future__ import annotations

import pytest

from games.consistency.game import (
    ConsistencyGame,
    GameOverError,
    decode_target,
    encode_target,
    feedback_str,
    is_consistent,
    parse_answer,
)


def _rows(target: str, *guesses: str) -> list[tuple[str, str]]:
    return [(g.upper(), feedback_str(g, target)) for g in guesses]


class TestConsistency:
    def test_target_is_always_consistent(self):
        rows = _rows("plant", "crane", "slate")
        assert is_consistent("plant", rows)

    def test_obvious_inconsistency(self):
        # crane vs plant marks 'a' present; a candidate with no 'a' is inconsistent.
        rows = _rows("plant", "crane")
        assert not is_consistent("doggy", rows)

    def test_encode_decode_round_trip(self):
        rows = _rows("plant", "crane")
        assert decode_target(encode_target(rows, "plant")) == ([(g, fb) for g, fb in rows], "PLANT")


class TestParseAnswer:
    def test_yes_no(self):
        assert parse_answer("<answer>yes</answer>") is True
        assert parse_answer("<answer>no</answer>") is False

    def test_absent(self):
        assert parse_answer("unsure") is None


class TestConsistencyGame:
    def test_consistent_yes_correct(self):
        rows = _rows("plant", "crane")
        g = ConsistencyGame(rows, "plant", "id1")
        st = g.step("<answer>yes</answer>")
        assert st.status == "correct" and st.solution is True

    def test_consistent_no_incorrect(self):
        rows = _rows("plant", "crane")
        g = ConsistencyGame(rows, "plant", "id2")
        assert g.step("<answer>no</answer>").status == "incorrect"

    def test_inconsistent_no_correct(self):
        rows = _rows("plant", "crane")
        g = ConsistencyGame(rows, "doggy", "id3")
        assert g.step("<answer>no</answer>").status == "correct"

    def test_unparseable_incorrect(self):
        g = ConsistencyGame(_rows("plant", "crane"), "plant", "id4")
        assert g.step("hmm").status == "incorrect"

    def test_step_after_end_raises(self):
        g = ConsistencyGame(_rows("plant", "crane"), "plant", "id5")
        g.step("<answer>yes</answer>")
        with pytest.raises(GameOverError):
            g.step("<answer>no</answer>")
