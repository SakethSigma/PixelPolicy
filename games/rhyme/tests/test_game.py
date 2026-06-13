"""Tests for the pure Rhymes logic in ``games.rhyme.game`` (needs the bundled CMU dict)."""

from __future__ import annotations

import random

import pytest

from games.rhyme.game import (
    GameOverError,
    RhymeGame,
    is_rhyme,
    parse_answer,
    rhymes,
)


class TestRhymeOracle:
    def test_known_rhyme(self):
        assert is_rhyme("cat", "hat")
        assert "hat" in rhymes("cat")

    def test_non_rhyme(self):
        assert not is_rhyme("cat", "dog")

    def test_case_insensitive(self):
        assert is_rhyme("CAT", "HAT")

    def test_unknown_word_has_no_rhymes(self):
        assert rhymes("zzzqx") == frozenset()


class TestParseAnswer:
    def test_pulls_last_word(self):
        assert parse_answer("I think the answer is hat") == "hat"

    def test_empty_is_none(self):
        assert parse_answer("123 !!!") is None


class TestFreeVariant:
    def test_correct_is_terminal(self):
        g = RhymeGame("cat", "id1", variant="free")
        assert g.state().status == "in_progress"
        state = g.step("<answer>hat</answer>")
        assert state.status == "correct"
        assert state.solution is not None and state.solution.variant == "free"

    def test_incorrect(self):
        g = RhymeGame("cat", "id2", variant="free")
        assert g.step("dog").status == "incorrect"

    def test_step_after_end_raises(self):
        g = RhymeGame("cat", "id3", variant="free")
        g.step("hat")
        with pytest.raises(GameOverError):
            g.step("bat")


class TestMCQVariant:
    def test_requires_options(self):
        with pytest.raises(ValueError):
            RhymeGame("cat", "id4", variant="mcq")

    def test_correct_only_for_rhyming_option(self):
        opts = ["hat", "dog", "table", "river", "purple"]
        g = RhymeGame("cat", "id5", variant="mcq", options=opts)
        assert g.step("hat").status == "correct"

    def test_non_rhyming_option_is_incorrect(self):
        opts = ["hat", "dog", "table", "river", "purple"]
        g = RhymeGame("cat", "id6", variant="mcq", options=opts)
        assert g.step("dog").status == "incorrect"

    def test_off_menu_answer_is_incorrect(self):
        # "bat" rhymes with cat but is NOT one of the options -> not a valid MCQ pick.
        opts = ["hat", "dog", "table", "river", "purple"]
        g = RhymeGame("cat", "id7", variant="mcq", options=opts)
        # both must hold for MCQ: in options AND rhymes. "sat" rhymes but isn't an option.
        assert g.step("sat").status == "incorrect"
