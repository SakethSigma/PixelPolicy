"""Tests for the pure Character-set logic in ``games.charset.game``."""

from __future__ import annotations

import pytest

from games.charset.game import (
    ALPHABET,
    CharsetGame,
    GameOverError,
    analyze,
    decode_words,
    encode_words,
    is_correct,
    parse_answer,
)
from games.charset.render import render_answer


class TestAnalyze:
    def test_union_and_complement(self):
        used, unused = analyze(["cat", "dog"])
        assert used == ["a", "c", "d", "g", "o", "t"]
        assert set(used).isdisjoint(unused)
        assert sorted(set(used) | set(unused)) == sorted(ALPHABET)

    def test_repeats_collapse(self):
        used, _ = analyze(["banana"])
        assert used == ["a", "b", "n"]

    def test_encode_decode(self):
        assert decode_words(encode_words(["cat", "PLANET"])) == ["cat", "planet"]


class TestParseAnswer:
    def test_round_trip(self):
        used, unused = analyze(["cat", "dog"])
        parsed = parse_answer(render_answer(used, unused))
        assert parsed is not None
        assert is_correct(["cat", "dog"], *parsed)

    def test_unused_not_misread_as_used(self):
        # "unused" contains "used" — the used-line regex must not capture the unused letters.
        used, unused = analyze(["cat"])
        text = render_answer(used, unused)
        p_used, p_unused = parse_answer(text)
        assert p_used == set(used) and p_unused == set(unused)

    def test_missing_line_is_none(self):
        assert parse_answer("used: a c t") is None       # no unused line
        assert parse_answer("just prose") is None


class TestCharsetGame:
    def test_correct_is_terminal(self):
        words = ["cat", "dog"]
        g = CharsetGame(words, "id1")
        assert g.state().status == "in_progress"
        used, unused = analyze(words)
        state = g.step(render_answer(used, unused))
        assert state.status == "correct"
        assert state.solution.used == used

    def test_wrong_is_incorrect(self):
        g = CharsetGame(["cat"], "id2")
        # Claim 'z' is used (it isn't) -> wrong.
        assert g.step("used: c a t z\nunused: b d").status == "incorrect"

    def test_unparseable_incorrect(self):
        assert CharsetGame(["cat"], "id3").step("no idea").status == "incorrect"

    def test_step_after_end_raises(self):
        words = ["cat"]
        used, unused = analyze(words)
        g = CharsetGame(words, "id4")
        g.step(render_answer(used, unused))
        with pytest.raises(GameOverError):
            g.step("anything")
