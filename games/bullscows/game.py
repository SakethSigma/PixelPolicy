"""Pure Bulls & Cows game logic — multi-turn deduction game #11.

A **multi-turn** environment that teaches reading **count-based** feedback and adjusting — a
different feedback representation from Wordle/codebreaker's per-position tiles. The secret is
``N_DIGITS`` **distinct** digits (default 4 from 0–9). Each guess (also distinct digits) is scored
by two counts:

    bulls = digits that are correct AND in the right position
    cows  = digits that are in the code but in the wrong position

The model guesses, sees the bull/cow counts, and guesses again until ``bulls == N_DIGITS`` or it
runs out of rounds. Pure module (no FastAPI), the single source of truth.

Data is generated **programmatically** by :class:`BullsCowsSolver` — an *unbiased* teacher that
narrows to the codes consistent with all feedback and guesses a **uniformly random** one (random
opening too). No fixed/ordered opening, so the student learns the deduction, not a habit.
"""

from __future__ import annotations

import random
from functools import lru_cache
from itertools import permutations
from typing import Literal, Optional

from pydantic import BaseModel, Field

GAME_NAME = "bullscows"
N_DIGITS = 4
DIGITS = "0123456789"
DEFAULT_MAX_ROUNDS = 10

Status = Literal["in_progress", "won", "lost"]


def compute_feedback(guess: str, secret: str) -> tuple[int, int]:
    """Return ``(bulls, cows)``. Assumes both are equal-length distinct-digit strings."""
    bulls = sum(g == s for g, s in zip(guess, secret))
    common = len(set(guess) & set(secret))
    return bulls, common - bulls


def is_valid_guess(guess: str) -> bool:
    g = guess.strip()
    return len(g) == N_DIGITS and g.isdigit() and len(set(g)) == N_DIGITS


@lru_cache(maxsize=1)
def all_codes() -> tuple[str, ...]:
    """Every distinct-digit code (10·9·8·7 = 5040 for N_DIGITS=4)."""
    return tuple("".join(p) for p in permutations(DIGITS, N_DIGITS))


class RoundResult(BaseModel):
    guess: str
    bulls: int = 0
    cows: int = 0
    error: Optional[str] = None      # set iff the guess was malformed (not N distinct digits)


class GameState(BaseModel):
    game_id: str
    n_digits: int = N_DIGITS
    max_rounds: int = DEFAULT_MAX_ROUNDS
    current_round: int = 0
    rounds: list[RoundResult] = Field(default_factory=list)
    status: Status = "in_progress"
    secret: Optional[str] = None     # revealed only once the game ends


class GameOverError(Exception):
    """Raised when a guess is submitted to a game that has already ended."""


class BullsCowsGame:
    """A single Bulls & Cows episode. The secret is held privately until the game ends."""

    def __init__(self, secret: str, game_id: str, *, max_rounds: int = DEFAULT_MAX_ROUNDS):
        self.secret = secret.strip()
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
        g = code.strip()
        if not is_valid_guess(g):
            result = RoundResult(guess=g, error="invalid")
        else:
            bulls, cows = compute_feedback(g, self.secret)
            result = RoundResult(guess=g, bulls=bulls, cows=cows)
        self.rounds.append(result)
        if result.error is None and result.bulls == N_DIGITS:
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


def consistent_codes(rounds: list[tuple[str, int, int]], pool: Optional[list[str]] = None) -> list[str]:
    """All codes consistent with every (guess, bulls, cows) round."""
    cands = list(pool) if pool is not None else list(all_codes())
    for g, bulls, cows in rounds:
        cands = [c for c in cands if compute_feedback(g, c) == (bulls, cows)]
    return cands


class BullsCowsSolver:
    """Unbiased programmatic teacher: random opening, then a uniformly random code consistent with
    all feedback so far. Narrows the candidate set incrementally per episode."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._candidates: list[str] = list(all_codes())
        self._seen = 0

    def move(self, state: GameState) -> str:
        """Return the teacher's reply: a short, true rationale + ``<guess>NNNN</guess>``."""
        valid_rounds = [(r.guess, r.bulls, r.cows) for r in state.rounds if r.error is None]
        while self._seen < len(valid_rounds):
            g, bulls, cows = valid_rounds[self._seen]
            self._candidates = [c for c in self._candidates if compute_feedback(g, c) == (bulls, cows)]
            self._seen += 1
        pool = self._candidates or list(all_codes())
        guess = self._rng.choice(pool)
        return f"{_reason(valid_rounds, guess, len(pool))} <guess>{guess}</guess>"


def _reason(rounds: list[tuple[str, int, int]], guess: str, n_left: int) -> str:
    """A true, templated rationale for guessing ``guess`` given the bull/cow clues so far."""
    if not rounds:
        return f"No clues yet, so I'll open with {guess}."
    recap = "; ".join(
        f"{g} → {b} bull{'s' if b != 1 else ''}, {c} cow{'s' if c != 1 else ''}"
        for g, b, c in rounds
    )
    return (f"From the clues so far ({recap}), {n_left} numbers still fit every count; "
            f"{guess} is one of them, so I'll try it.")


class BullsCowsBank:
    """Generates random distinct-digit secrets."""

    def __init__(self):
        self._rng = random.Random()

    def make_secret(self, rng: random.Random) -> str:
        return "".join(rng.sample(DIGITS, N_DIGITS))

    def sample(self, mode: str) -> str:
        return self.make_secret(self._rng)
