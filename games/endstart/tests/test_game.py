"""Tests for the pure Ends-with → starts-with logic."""

from __future__ import annotations

import random

import pytest

from games.endstart.game import (
    EndstartGame,
    GameOverError,
    correct_option,
    decode_target,
    encode_target,
    parse_answer,
)


class TestCorrectOption:
    def test_matches_last_letter(self):
        assert correct_option("mango", ["river", "oasis", "tundra", "cliff", "marsh"]) == "oasis"

    def test_none_when_no_match(self):
        assert correct_option("mango", ["river", "tundra", "cliff", "marsh"]) is None

    def test_encode_decode(self):
        assert decode_target(encode_target("mango", ["river", "oasis"])) == ("mango", ["river", "oasis"])


class TestParseAnswer:
    def test_tag_body(self):
        assert parse_answer("<answer>oasis</answer>") == "oasis"

    def test_last_word(self):
        assert parse_answer("I pick oasis") == "oasis"

    def test_empty(self):
        assert parse_answer("123") is None


class TestEndstartGame:
    OPTS = ["river", "oasis", "tundra", "cliff", "marsh"]

    def test_correct(self):
        g = EndstartGame("mango", self.OPTS, "id1")
        state = g.step("<answer>oasis</answer>")
        assert state.status == "correct" and state.solution == "oasis"

    def test_incorrect(self):
        g = EndstartGame("mango", self.OPTS, "id2")
        assert g.step("<answer>river</answer>").status == "incorrect"

    def test_unparseable(self):
        g = EndstartGame("mango", self.OPTS, "id3")
        assert g.step("123").status == "incorrect"

    def test_step_after_end_raises(self):
        g = EndstartGame("mango", self.OPTS, "id4")
        g.step("<answer>oasis</answer>")
        with pytest.raises(GameOverError):
            g.step("oasis")
