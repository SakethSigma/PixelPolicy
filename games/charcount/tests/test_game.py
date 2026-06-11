"""Tests for the pure Character-counts logic in ``games.charcount.game``."""

from __future__ import annotations

import pytest

from games.charcount.game import (
    Analysis,
    CharCountGame,
    GameOverError,
    analyze,
    is_correct,
    parse_answer,
)
from games.charcount.render import render_answer


class TestAnalyze:
    def test_basic(self):
        a = analyze("planet")
        assert a.length == 6
        assert a.vowels == ["a", "e"]
        assert a.consonants == ["p", "l", "n", "t"]
        assert a.vowel_count == 2 and a.consonant_count == 4

    def test_invariant_length_equals_split(self):
        for word in ["banana", "rhythm", "queue", "a", "strengths"]:
            a = analyze(word)
            assert a.length == a.vowel_count + a.consonant_count

    def test_repeats_kept(self):
        a = analyze("banana")
        assert a.vowels == ["a", "a", "a"]
        assert a.consonants == ["b", "n", "n"]

    def test_y_is_consonant(self):
        a = analyze("rhythm")
        assert a.vowels == []
        assert "y" in a.consonants

    def test_case_insensitive(self):
        assert analyze("PLANET").vowels == ["a", "e"]


class TestParseAnswer:
    def test_canonical_block(self):
        text = render_answer(analyze("planet"))
        parsed = parse_answer(text)
        assert parsed is not None
        assert is_correct("planet", parsed)

    def test_order_and_spacing_forgiving(self):
        text = "consonants: p l n t\nlength=6\nVowels: a, e"
        parsed = parse_answer(text)
        assert parsed is not None
        assert is_correct("planet", parsed)

    def test_missing_lines_unparseable(self):
        assert parse_answer("length: 6") is None        # no vowel/consonant lines
        assert parse_answer("just some prose") is None

    def test_length_defaults_to_split_sum(self):
        parsed = parse_answer("vowels: a e\nconsonants: p l n t")
        assert parsed is not None and parsed.length == 6


class TestIsCorrect:
    def test_wrong_split_fails(self):
        wrong = Analysis(length=6, vowels=["a"], consonants=["p", "l", "n", "t", "e"])
        assert not is_correct("planet", wrong)

    def test_multiset_compare(self):
        right = Analysis(length=6, vowels=["a", "a", "a"], consonants=["n", "b", "n"])
        assert is_correct("banana", right)


class TestCharCountGame:
    def test_correct_answer_is_terminal(self):
        g = CharCountGame("planet", "id1")
        assert g.state().status == "in_progress"
        assert g.state().solution is None
        state = g.step(render_answer(analyze("planet")))
        assert state.status == "correct"
        assert state.solution == analyze("planet")     # revealed on completion
        assert state.submitted is not None

    def test_incorrect_answer_is_terminal(self):
        g = CharCountGame("planet", "id2")
        state = g.step("length: 6\nvowels: a\nconsonants: p")
        assert state.status == "incorrect"
        assert state.solution == analyze("planet")     # truth still revealed

    def test_unparseable_is_incorrect(self):
        g = CharCountGame("planet", "id3")
        assert g.step("I have no idea").status == "incorrect"

    def test_step_after_end_raises(self):
        g = CharCountGame("planet", "id4")
        g.step(render_answer(analyze("planet")))
        with pytest.raises(GameOverError):
            g.step("anything")
