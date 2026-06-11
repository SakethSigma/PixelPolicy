"""Render tests — and the round-trip that keeps the SFT teacher honest."""

from __future__ import annotations

from games.charcount.game import CharCountGame, analyze, is_correct, parse_answer
from games.charcount.render import render_answer, render_observation


def test_observation_is_just_the_word():
    g = CharCountGame("planet", "id")
    assert render_observation(g.state()) == "Word: planet"


def test_render_answer_canonical_shape():
    # Space-separated, UPPERCASE letters.
    assert render_answer(analyze("planet")) == (
        "length: 6\n"
        "vowels (2): A E\n"
        "consonants (4): P L N T"
    )


def test_render_handles_no_vowels_or_no_consonants():
    # Empty lists render as "-" (a non-letter sentinel) so they parse back as an empty list.
    assert "vowels (0): -" in render_answer(analyze("rhythm"))


def test_render_then_parse_round_trips_for_many_words():
    # The canonical block the synthetic teacher writes must always score correct — this is
    # the rejection-by-construction guarantee the generator relies on.
    for word in ["a", "banana", "rhythm", "strengths", "queueing", "aeiou", "zzz"]:
        block = render_answer(analyze(word))
        parsed = parse_answer(block)
        assert parsed is not None
        assert is_correct(word, parsed)
