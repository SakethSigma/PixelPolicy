"""Pure Candidate-consistency game logic (Word-skill game #12).

A **single-turn** environment that teaches the *positive* side of feedback reasoning: given a
Wordle board (some past guesses + their ✓/-/x feedback) and a **candidate** word, decide whether
that candidate is **still possible** — consistent with every clue. (Its sibling ``mistakeid``
teaches *locating* a guess's errors; this one teaches the binary "is this word still in the running"
filter that narrows the answer set.)

Ground truth reuses Wordle's own scorer: a candidate ``c`` is consistent with a row ``(guess, fb)``
iff ``compute_feedback(guess, c) == fb`` — which captures greens/yellows/greys including the
duplicate rules exactly. Pure module (no FastAPI), single source of truth.

    "in_progress" -> board + candidate posed, no answer yet
    "correct"     -> yes/no answer matches whether the candidate is consistent   (the "good" status)
    "incorrect"   -> wrong verdict, or unparseable
"""

from __future__ import annotations

import re
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from games.wordle.game import compute_feedback

GAME_NAME = "consistency"

Status = Literal["in_progress", "correct", "incorrect"]

_SYM_TO_CODE = {"✓": "g", "-": "y", "x": "x"}
_CODE_TO_SYM = {v: k for k, v in _SYM_TO_CODE.items()}


def feedback_str(guess: str, target: str) -> str:
    """Wordle feedback as a ✓/-/x string (reusing the canonical scorer)."""
    return "".join(f.value for f in compute_feedback(guess, target))


def is_consistent(candidate: str, rows: list[tuple[str, str]]) -> bool:
    """Whether ``candidate`` is consistent with every ``(guess, ✓/-/x feedback)`` row."""
    return all(feedback_str(guess, candidate) == fb for guess, fb in rows)


def encode_target(rows: list[tuple[str, str]], candidate: str) -> str:
    """Encode as ``g1:code1|g2:code2;candidate`` where code is g/y/x (ASCII)."""
    rows_s = "|".join(f"{g.upper()}:{''.join(_SYM_TO_CODE[s] for s in fb)}" for g, fb in rows)
    return f"{rows_s};{candidate.upper()}"


def decode_target(target: str) -> tuple[list[tuple[str, str]], str]:
    """Inverse of :func:`encode_target` (codes back to ✓/-/x)."""
    rows_s, _, candidate = target.partition(";")
    rows: list[tuple[str, str]] = []
    for tok in rows_s.split("|"):
        g, _, code = tok.partition(":")
        rows.append((g.strip().upper(), "".join(_CODE_TO_SYM[c] for c in code.strip())))
    return rows, candidate.strip().upper()


def parse_answer(text: str) -> Optional[bool]:
    """Pull the yes/no verdict (last occurrence, ``<answer>`` body preferred); ``None`` if absent."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    src = m.group(1) if m else text
    tokens = re.findall(r"\b(yes|no)\b", src.lower())
    return None if not tokens else tokens[-1] == "yes"


class GameState(BaseModel):
    game_id: str
    rows: list[tuple[str, str]]          # (guess, ✓/-/x feedback)
    candidate: str
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solution: Optional[bool] = None      # whether the candidate is consistent (revealed at end)


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class ConsistencyGame:
    """A single Candidate-consistency episode. ``step`` scores once and ends the game."""

    def __init__(self, rows: list[tuple[str, str]], candidate: str, game_id: str):
        self.rows = [(g.upper(), fb) for g, fb in rows]
        self.candidate = candidate.strip().upper()
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def step(self, answer: str) -> GameState:
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        verdict = parse_answer(answer)
        truth = is_consistent(self.candidate, self.rows)
        self.status = "correct" if (verdict is not None and verdict == truth) else "incorrect"
        return self.state()

    def state(self) -> GameState:
        return GameState(
            game_id=self.game_id, rows=self.rows, candidate=self.candidate,
            status=self.status, submitted=self.submitted,
            solution=is_consistent(self.candidate, self.rows) if self.status != "in_progress" else None,
        )


class ConsistencyBank:
    """Builds balanced consistent/inconsistent challenges from the Wordle vocabulary.

    A board is a hidden target + 1–3 random (non-target) guesses scored against it; the candidate
    is then chosen to be consistent or inconsistent with that board, 50/50.
    """

    def __init__(self):
        from games.wordle.game import WordBank

        self.words: list[str] = sorted(WordBank().all)
        self._rng = __import__("random").Random()

    def make_challenge(self, rng, *, want_consistent: bool) -> tuple[list[tuple[str, str]], str]:
        for _ in range(1000):
            target = rng.choice(self.words)
            others = [w for w in self.words if w != target]
            n_rows = rng.randint(1, 3)
            guesses = rng.sample(others, n_rows)
            rows = [(g.upper(), feedback_str(g, target)) for g in guesses]
            for _ in range(200):
                cand = rng.choice(self.words)
                if is_consistent(cand, rows) == want_consistent:
                    return rows, cand.upper()
        raise RuntimeError("could not build a consistency challenge")

    def sample_targets(self, n: int, mode: str, rng) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        guard = 0
        while len(out) < n and guard < n * 50:
            guard += 1
            rows, cand = self.make_challenge(rng, want_consistent=(len(out) % 2 == 0))
            enc = encode_target(rows, cand)
            if enc not in seen:
                seen.add(enc)
                out.append(enc)
        rng.shuffle(out)
        return out

    def sample(self, mode: str) -> str:
        rows, cand = self.make_challenge(self._rng, want_consistent=self._rng.random() < 0.5)
        return encode_target(rows, cand)
