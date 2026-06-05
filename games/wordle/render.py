"""Plain-text rendering of a Wordle ``GameState``.

Dependency-free on purpose (no ``rich``): this is the text a human reads in a bare
terminal *and* the observation an LLM policy is shown when it plays. Keeping both on
the same renderer is what makes "a human gets the same feedback as the model" literal.
``play.py`` layers colored tiles on top of this same logical layout.
"""

from __future__ import annotations

from games.wordle.game import GameState, LetterFeedback, RoundResult

_LEGEND = (
    f"Legend: {LetterFeedback.CORRECT.value} right spot   "
    f"{LetterFeedback.WRONG_POS.value} wrong spot   "
    f"{LetterFeedback.WRONG_LETTER.value} not in word"
)


def render_round(rnd: RoundResult) -> str:
    """One history line: the guess plus its feedback, or the invalid reason.

    Valid:   ``CRANE  x - - x -``
    Invalid: ``ZZZZZ  [invalid: out of vocabulary — counted as a round]``
    """
    if rnd.error is not None:
        return f"{rnd.guess}  [invalid: {rnd.error.value} — counted as a round]"
    letters = " ".join(rnd.guess)
    symbols = " ".join(f.value for f in rnd.feedback)
    return f"{letters}   {symbols}"


def render_observation(state: GameState) -> str:
    """Full board as text, suitable for an LLM prompt or a plain terminal.

    Shows the rules, a symbol legend, every round so far (invalid rounds included,
    with their reason and no feedback), and how many guesses remain. The target word
    is revealed only once the game is over — exactly as ``GameState`` exposes it.
    """
    word_length = len(state.target) if state.target else 5
    lines = [
        f"Wordle — {word_length} letters, {state.max_rounds} guesses.",
        _LEGEND,
        "",
    ]

    if state.rounds:
        lines += [render_round(r) for r in state.rounds]
    else:
        lines.append("(no guesses yet)")
    lines.append("")

    if state.status == "in_progress":
        remaining = state.max_rounds - state.current_round
        lines.append(f"Rounds left: {remaining}")
    elif state.status == "won":
        lines.append(f"You won in {state.current_round} guesses.")
    else:  # lost
        revealed = f" The word was {state.target}." if state.target else ""
        lines.append(f"You lost.{revealed}")

    return "\n".join(lines)
