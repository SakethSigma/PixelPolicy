"""Tests for the pure Validity logic (needs the committed meanings.jsonl asset).

Build the asset first: ``python -m games.wordvocab.build_meanings``.
"""

from __future__ import annotations

import random

import pytest

from games.validity.game import (
    GameOverError,
    ValidityGame,
    is_valid_word,
    parse_answer,
    perturb,
)
from games.validity.render import render_answer


class TestParseAnswer:
    def test_valid_with_meaning(self):
        parsed = parse_answer("<answer>valid</answer>\n<meaning>a small thing</meaning>")
        assert parsed == (True, "a small thing")

    def test_invalid_checked_before_valid_substring(self):
        # "invalid" contains "valid" — must not be mis-read as valid.
        assert parse_answer("<answer>invalid</answer>") == (False, "")

    def test_no_verdict_is_none(self):
        assert parse_answer("I am not sure about this word") is None

    def test_meaning_containing_invalid_does_not_flip_verdict(self):
        # WordNet glosses can contain the word "invalid" (e.g. annul: "declare invalid").
        text = "<answer>valid</answer>\n<meaning>declare invalid; cancel</meaning>"
        assert parse_answer(text) == (True, "declare invalid; cancel")


class TestPerturb:
    def test_changes_the_word(self):
        rng = random.Random(0)
        # Over several tries at least one perturbation differs from the source.
        assert any(perturb("planet", rng) != "planet" for _ in range(5))


class TestValidityGame:
    def test_valid_word_needs_correct_verdict_and_meaning(self):
        assert is_valid_word("kindle")  # a real WordNet word with a definition
        g = ValidityGame("kindle", "id1")
        state = g.step(render_answer(True, "to set fire to"))
        assert state.status == "correct"
        assert state.solution.valid is True and state.solution.meaning

    def test_valid_word_without_meaning_is_incorrect(self):
        g = ValidityGame("kindle", "id2")
        assert g.step("<answer>valid</answer>").status == "incorrect"

    def test_calling_valid_word_invalid_is_incorrect(self):
        g = ValidityGame("kindle", "id3")
        assert g.step("<answer>invalid</answer>").status == "incorrect"

    def test_pseudo_word_invalid_is_correct(self):
        assert not is_valid_word("zzzqxv")
        g = ValidityGame("zzzqxv", "id4")
        assert g.step(render_answer(False)).status == "correct"

    def test_step_after_end_raises(self):
        g = ValidityGame("kindle", "id5")
        g.step(render_answer(True, "x"))
        with pytest.raises(GameOverError):
            g.step("anything")
