"""Plain-text rendering of a Candidate-consistency ``GameState`` (dependency-free)."""

from __future__ import annotations

from games.consistency.game import GameState, feedback_str


def _row(guess: str, feedback: str) -> str:
    return f"{' '.join(guess)}   {' '.join(feedback)}"


def _violation_reason(guess: str, fb: str, cand: str) -> str:
    """A concrete, true reason ``cand`` conflicts with the clue ``(guess, fb)`` (caller has already
    confirmed they disagree). Checks the common Wordle constraints in priority order."""
    present = {guess[j] for j in range(len(guess)) if fb[j] in "✓-"}
    for i in range(len(guess)):
        if fb[i] == "✓" and cand[i] != guess[i]:
            return f"position {i + 1} must be {guess[i]}, but {cand} has {cand[i]} there"
    for i in range(len(guess)):
        if fb[i] == "x" and guess[i] not in present and guess[i] in cand:
            return f"{guess[i]} is not in the word, but {cand} contains it"
    for i in range(len(guess)):
        if fb[i] == "-" and cand[i] == guess[i]:
            return f"{guess[i]} cannot be in position {i + 1}, but {cand} puts it there"
    for i in range(len(guess)):
        if fb[i] == "-" and guess[i] not in cand:
            return f"{guess[i]} must be in the word, but {cand} does not contain it"
    return "the letter counts do not fit the clue"


def render_reasoning(rows: list[tuple[str, str]], candidate: str) -> str:
    """A true, programmatically-generated explanation of the consistency check (in words).

    Walks each clue: if the answer were the candidate, would guessing that word reproduce the
    clue's feedback? The first clue it fails pinpoints why it is ruled out; if all match, the
    candidate is still possible. Always agrees with :func:`games.consistency.game.is_consistent`.
    """
    cand = candidate.strip().upper()
    parts: list[str] = []
    for guess, fb in rows:
        cf = feedback_str(guess, cand)
        if cf == fb:
            parts.append(f"If the word were {cand}, guessing {guess} would score "
                         f"{' '.join(cf)}, which matches the clue.")
        else:
            parts.append(f"If the word were {cand}, guessing {guess} would score "
                         f"{' '.join(cf)}, but the clue shows {' '.join(fb)}: "
                         f"{_violation_reason(guess, fb, cand)}.")
            parts.append(f"So {cand} is ruled out.")
            return " ".join(parts)
    parts.append(f"Every clue is satisfied, so {cand} is still possible.")
    return " ".join(parts)


def render_observation(state: GameState) -> str:
    """The board + candidate a human and the model read."""
    board = "\n".join(_row(g, fb) for g, fb in state.rows)
    return (
        "A Wordle player has these clues so far (✓ = right spot, - = right letter wrong spot, "
        "x = not in the word):\n"
        f"{board}\n\n"
        f'Is the word "{state.candidate}" still possible given all of these clues? Answer yes or no.'
    )
