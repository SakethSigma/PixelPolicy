"""Tests for the pure Anagrams logic in ``games.anagram.game``."""

from __future__ import annotations

import random

import pytest

from games.anagram.game import (
    AnagramGame,
    GameOverError,
    are_anagrams,
    decode_pair,
    encode_pair,
    parse_answer,
    signature,
)


class TestGroundTruth:
    def test_anagram_pair(self):
        assert are_anagrams("listen", "silent")
        assert signature("listen") == signature("silent")

    def test_non_anagram(self):
        assert not are_anagrams("listen", "listed")
        assert not are_anagrams("cat", "cats")  # different length can't be anagrams

    def test_case_and_whitespace(self):
        assert are_anagrams(" LISTEN ", "Silent")

    def test_encode_decode_roundtrip(self):
        assert decode_pair(encode_pair("listen", "silent")) == ("listen", "silent")


class TestParseAnswer:
    def test_yes_no(self):
        assert parse_answer("<answer>yes</answer>") is True
        assert parse_answer("<answer>no</answer>") is False

    def test_takes_last_verdict(self):
        assert parse_answer("maybe no... actually <answer>yes</answer>") is True

    def test_absent_is_none(self):
        assert parse_answer("I am unsure") is None


class TestAnagramGame:
    def test_correct_yes(self):
        g = AnagramGame("listen", "silent", "id1")
        state = g.step("<answer>yes</answer>")
        assert state.status == "correct" and state.solution.are_anagrams is True

    def test_correct_no(self):
        g = AnagramGame("listen", "listed", "id2")
        assert g.step("<answer>no</answer>").status == "correct"

    def test_wrong_verdict_incorrect(self):
        g = AnagramGame("listen", "silent", "id3")
        assert g.step("<answer>no</answer>").status == "incorrect"

    def test_unparseable_incorrect(self):
        g = AnagramGame("listen", "silent", "id4")
        assert g.step("hmm").status == "incorrect"

    def test_step_after_end_raises(self):
        g = AnagramGame("listen", "silent", "id5")
        g.step("<answer>yes</answer>")
        with pytest.raises(GameOverError):
            g.step("no")
