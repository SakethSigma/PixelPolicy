"""Plain-text rendering of a Character-counts ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown. :func:`render_answer` is the canonical answer block — the
"synthetic teacher" writes the SFT completion with it, and a human sees the same format — so
the text the model trains on can never drift from the human view. :func:`games.charcount.game`'s
tolerant ``parse_answer`` reads this format (and minor variants) back.
"""

from __future__ import annotations

from games.charcount.game import Analysis, GameState, analyze


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read: just the word to analyze."""
    return f"Word: {state.word}"


def render_answer(analysis: Analysis) -> str:
    """The canonical ``<answer>`` block body for a computed analysis.

    Letters are space-separated and UPPERCASE. Example for ``planet``::

        length: 6
        vowels (2): A E
        consonants (4): P L N T
    """
    # Empty lists render as "-" (a non-letter sentinel) so parse_answer reads back an empty
    # list — a word like "(none)" would otherwise be mis-parsed as the letters n, o, n, e.
    vowels = " ".join(c.upper() for c in analysis.vowels) if analysis.vowels else "-"
    consonants = " ".join(c.upper() for c in analysis.consonants) if analysis.consonants else "-"
    return (
        f"length: {analysis.length}\n"
        f"vowels ({analysis.vowel_count}): {vowels}\n"
        f"consonants ({analysis.consonant_count}): {consonants}"
    )


def render_solution(word: str) -> str:
    """Convenience: the canonical answer block for ``word`` (used by play.py)."""
    return render_answer(analyze(word))
