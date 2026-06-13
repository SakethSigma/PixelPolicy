"""Pure Codebreaker (Mastermind) game logic — multi-turn deduction game #10.

A **multi-turn** environment that teaches reading per-position feedback and adjusting — the core
Wordle loop, decoupled from vocabulary. A secret **code** is a length-``CODE_LENGTH`` string over a
small symbol alphabet (default 4 slots over ``A–F``; repeats allowed, to exercise duplicate
handling). Each guess is scored per position with the **same symbols as Wordle**:

    ✓  right symbol in the right slot      (green)
    -  right symbol in the wrong slot       (yellow)
    x  symbol not in the code (or all copies already accounted for)   (gray)

The model guesses, sees feedback, and guesses again until it cracks the code or runs out of
rounds. Like ``games.wordle.game`` this module is pure (no FastAPI) and is the single source of
truth; ``compute_feedback`` is the same two-pass duplicate-correct algorithm Wordle uses.

The data is generated **programmatically** by :class:`CodebreakerSolver` — an *unbiased* teacher
that, each turn, narrows to the set of codes consistent with all feedback so far and guesses a
**uniformly random** one (and opens with a uniformly random code). No fixed opening, no ordered
symbol preference — so the student learns "guess something consistent", not a positional habit.
"""

from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache
from itertools import product
from typing import Literal, Optional

from pydantic import BaseModel, Field

GAME_NAME = "codebreaker"
CODE_LENGTH = 4
SYMBOLS = "ABCDEF"
DEFAULT_MAX_ROUNDS = 12

Status = Literal["in_progress", "won", "lost"]


def compute_feedback(guess: str, secret: str) -> str:
    """Per-position ✓/-/x feedback with Wordle's two-pass duplicate handling."""
    g, s = guess.upper(), secret.upper()
    fb: list[Optional[str]] = [None] * len(g)
    remaining = Counter(s)
    for i, (a, b) in enumerate(zip(g, s)):
        if a == b:
            fb[i] = "✓"
            remaining[a] -= 1
    for i, a in enumerate(g):
        if fb[i] is not None:
            continue
        if remaining.get(a, 0) > 0:
            fb[i] = "-"
            remaining[a] -= 1
        else:
            fb[i] = "x"
    return "".join(fb)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def all_codes() -> tuple[str, ...]:
    """Every possible secret (``len(SYMBOLS) ** CODE_LENGTH`` of them)."""
    return tuple("".join(c) for c in product(SYMBOLS, repeat=CODE_LENGTH))


def is_valid_guess(guess: str) -> bool:
    g = guess.strip().upper()
    return len(g) == CODE_LENGTH and all(c in SYMBOLS for c in g)


class RoundResult(BaseModel):
    guess: str                       # uppercased
    feedback: str = ""               # ✓/-/x per position; empty when the guess was invalid
    error: Optional[str] = None      # set iff the guess was malformed


class GameState(BaseModel):
    game_id: str
    code_length: int = CODE_LENGTH
    symbols: str = SYMBOLS
    max_rounds: int = DEFAULT_MAX_ROUNDS
    current_round: int = 0
    rounds: list[RoundResult] = Field(default_factory=list)
    status: Status = "in_progress"
    secret: Optional[str] = None     # revealed only once the game ends


class GameOverError(Exception):
    """Raised when a guess is submitted to a game that has already ended."""


class CodebreakerGame:
    """A single Codebreaker episode. The secret is held privately until the game ends."""

    def __init__(self, secret: str, game_id: str, *, max_rounds: int = DEFAULT_MAX_ROUNDS):
        self.secret = secret.strip().upper()
        self.game_id = game_id
        self.max_rounds = max_rounds
        self.rounds: list[RoundResult] = []
        self.status: Status = "in_progress"

    @property
    def current_round(self) -> int:
        return len(self.rounds)

    def guess(self, code: str) -> RoundResult:
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        g = code.strip().upper()
        if not is_valid_guess(g):
            result = RoundResult(guess=g, error="invalid")
        else:
            result = RoundResult(guess=g, feedback=compute_feedback(g, self.secret))
        self.rounds.append(result)
        if result.feedback and all(c == "✓" for c in result.feedback):
            self.status = "won"
        elif len(self.rounds) >= self.max_rounds:
            self.status = "lost"
        return result

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id, max_rounds=self.max_rounds, current_round=self.current_round,
            rounds=self.rounds, status=self.status,
            secret=self.secret if self.status != "in_progress" else None,
        )


def consistent_codes(rounds: list[tuple[str, str]], pool: Optional[list[str]] = None) -> list[str]:
    """All codes consistent with every (guess, feedback) round."""
    cands = list(pool) if pool is not None else list(all_codes())
    for g, fb in rounds:
        cands = [c for c in cands if compute_feedback(g, c) == fb]
    return cands


class CodebreakerSolver:
    """Unbiased programmatic teacher: open randomly, then guess a uniformly random code that is
    consistent with all feedback so far. Maintains the candidate set incrementally per episode."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._candidates: list[str] = list(all_codes())
        self._seen = 0

    def move(self, state: GameState) -> str:
        """Return the teacher's reply: a short, true rationale + ``<guess>CODE</guess>``."""
        valid_rounds = [(r.guess, r.feedback) for r in state.rounds if r.feedback]
        while self._seen < len(valid_rounds):                     # narrow by any new feedback
            g, fb = valid_rounds[self._seen]
            self._candidates = [c for c in self._candidates if compute_feedback(g, c) == fb]
            self._seen += 1
        pool = self._candidates or list(all_codes())              # safety: never empty
        guess = self._rng.choice(pool)
        return f"{_reason(valid_rounds, guess, len(pool))} <guess>{guess}</guess>"


def _reason(rounds: list[tuple[str, str]], guess: str, n_left: int) -> str:
    """A true, templated rationale for guessing ``guess`` given the feedback so far."""
    if not rounds:
        return f"No feedback yet, so I'll open with {guess}."
    greens: dict[int, str] = {}
    present: set[str] = set()
    greyed: set[str] = set()
    for g, fb in rounds:
        for i, (c, f) in enumerate(zip(g, fb)):
            if f == "✓":
                greens[i] = c
                present.add(c)
            elif f == "-":
                present.add(c)
            elif f == "x":
                greyed.add(c)
    absent = greyed - present
    fixed = ", ".join(f"slot {i + 1}={greens[i]}" for i in sorted(greens)) or "none"
    misplaced = ", ".join(sorted(present - set(greens.values()))) or "none"
    notin = ", ".join(sorted(absent)) or "none"
    return (f"Clues so far — fixed: {fixed}; in the code but misplaced: {misplaced}; "
            f"not in the code: {notin}. {n_left} codes still fit; {guess} is one of them, so I'll try it.")


class CodebreakerBank:
    """Generates random secrets (repeats allowed, to exercise duplicate feedback)."""

    def __init__(self):
        self._rng = random.Random()

    def make_secret(self, rng: random.Random) -> str:
        return "".join(rng.choice(SYMBOLS) for _ in range(CODE_LENGTH))

    def sample(self, mode: str) -> str:
        return self.make_secret(self._rng)
