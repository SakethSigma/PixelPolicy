"""Pure Tower deduction game logic (Word-skill game #9).

A **single-turn** environment that teaches the model to reason from Wordle-style feedback. A
tower has 3 floors (1 = bottom, 3 = top), each with two rooms (Left / Right). Three people each
occupy a different room, **one per floor** (the floor assignment is a bijection). The model is
shown a *proposed* placement and, for each person, two ✓/x flags — whether the **floor** is
correct and whether the **room** is correct (same symbols as Wordle). It must deduce **every**
placement consistent with that feedback.

Like ``games.charcount.game`` this module has **no** web/FastAPI dependency and is the single
source of truth. The number of consistent placements is provably small:

  - Room feedback fixes each person's room exactly (it's one of two), so rooms never branch.
  - Floor feedback fixes the floor bijection up to derangements: if any floor is correct there is
    exactly **1** solution; if all three floors are wrong there are exactly **2** (the two
    derangements of 3). So the answer set always has size 1 or 2 — never more.

The single-turn ``status`` convention mirrors charcount/Wordle:

    "in_progress"  -> challenge posed, no answer yet
    "correct"      -> the listed placements exactly match the consistent set   (the "good" status)
    "incorrect"    -> wrong/missing/extra placement, or unparseable
"""

from __future__ import annotations

import re
import uuid
from itertools import permutations
from typing import Literal, Optional

from pydantic import BaseModel

GAME_NAME = "tower"
FLOORS = (1, 2, 3)
ROOM_NAME = {0: "Left", 1: "Right"}
_ROOM_IDX = {"left": 0, "right": 1, "l": 0, "r": 1}

Status = Literal["in_progress", "correct", "incorrect"]


def solve(shown_floors: list[int], shown_rooms: list[int],
          floor_ok: list[bool], room_ok: list[bool]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All placements consistent with the feedback: list of ``(floor_perm, room_tuple)``.

    Rooms are determined per person (matched room kept, mismatched room flipped). Floors are the
    permutations of (1,2,3) whose agreement with ``shown_floors`` matches ``floor_ok`` exactly.
    """
    rooms = tuple(shown_rooms[i] if room_ok[i] else 1 - shown_rooms[i] for i in range(3))
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for perm in permutations(FLOORS):
        if all((perm[i] == shown_floors[i]) == floor_ok[i] for i in range(3)):
            out.append((perm, rooms))
    return out


def encode_target(names: list[str], shown_floors: list[int], shown_rooms: list[int],
                  floor_ok: list[bool], room_ok: list[bool]) -> str:
    """Encode a challenge as ``names;shown;feedback`` — e.g. ``Alice,Bob,Carol;2L,1R,3L;01,10,00``."""
    names_s = ",".join(names)
    shown_s = ",".join(f"{shown_floors[i]}{'L' if shown_rooms[i] == 0 else 'R'}" for i in range(3))
    fb_s = ",".join(f"{int(floor_ok[i])}{int(room_ok[i])}" for i in range(3))
    return f"{names_s};{shown_s};{fb_s}"


def decode_target(target: str) -> tuple[list[str], list[int], list[int], list[bool], list[bool]]:
    """Inverse of :func:`encode_target`."""
    names_s, shown_s, fb_s = target.split(";")
    names = [w.strip() for w in names_s.split(",")]
    shown_floors, shown_rooms = [], []
    for tok in shown_s.split(","):
        tok = tok.strip()
        shown_floors.append(int(tok[0]))
        shown_rooms.append(0 if tok[1].upper() == "L" else 1)
    floor_ok, room_ok = [], []
    for tok in fb_s.split(","):
        tok = tok.strip()
        floor_ok.append(tok[0] == "1")
        room_ok.append(tok[1] == "1")
    return names, shown_floors, shown_rooms, floor_ok, room_ok


class PersonPlacement(BaseModel):
    name: str
    floor: int
    room: str          # "Left" / "Right"


class GameState(BaseModel):
    """Serializable snapshot. The solutions are revealed only once the episode ends."""

    game_id: str
    names: list[str]
    shown_floors: list[int]
    shown_rooms: list[int]                       # 0 = Left, 1 = Right
    floor_ok: list[bool]
    room_ok: list[bool]
    status: Status = "in_progress"
    submitted: Optional[str] = None
    solutions: Optional[list[list[PersonPlacement]]] = None


def _gold_set(names: list[str],
              sols: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> set[frozenset[tuple[str, int, int]]]:
    """Canonical comparable form: a set of solutions, each a frozenset of (name, floor, room_idx)."""
    return {
        frozenset((names[i].lower(), perm[i], rooms[i]) for i in range(3))
        for perm, rooms in sols
    }


_PERSON_LINE = re.compile(
    r"([A-Za-z][A-Za-z'\-]*)\s*:?\s*floor\s*([1-3])\s*,?\s*(left|right|\bl\b|\br\b)",
    re.IGNORECASE,
)
_SOLUTION_SPLIT = re.compile(r"solution\s*\d*\s*:?", re.IGNORECASE)


def parse_answer(text: str, names: list[str]) -> Optional[set[frozenset[tuple[str, int, int]]]]:
    """Parse a submitted answer into a set of solutions (each a frozenset of (name, floor, room)).

    Splits on ``solution`` headers (or treats the whole text as one solution if there are none),
    then reads ``Name: floor N, Left/Right`` lines. Only known ``names`` are kept. Returns ``None``
    if no valid person line is found at all.
    """
    valid = {n.lower() for n in names}
    segments = _SOLUTION_SPLIT.split(text)
    sols: set[frozenset[tuple[str, int, int]]] = set()
    any_line = False
    for seg in segments:
        block: set[tuple[str, int, int]] = set()
        for name, floor, room in _PERSON_LINE.findall(seg):
            if name.lower() in valid:
                block.add((name.lower(), int(floor), _ROOM_IDX[room.lower()]))
                any_line = True
        if block:
            sols.add(frozenset(block))
    return sols if any_line else None


class GameOverError(Exception):
    """Raised when an answer is submitted to an episode that has already ended."""


class TowerGame:
    """A single Tower-deduction episode. ``step`` scores once and ends the game."""

    def __init__(self, names: list[str], shown_floors: list[int], shown_rooms: list[int],
                 floor_ok: list[bool], room_ok: list[bool], game_id: str):
        self.names = list(names)
        self.shown_floors = list(shown_floors)
        self.shown_rooms = list(shown_rooms)
        self.floor_ok = list(floor_ok)
        self.room_ok = list(room_ok)
        self.game_id = game_id
        self.status: Status = "in_progress"
        self.submitted: Optional[str] = None

    def _solutions(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        return solve(self.shown_floors, self.shown_rooms, self.floor_ok, self.room_ok)

    def solution_placements(self) -> list[list[PersonPlacement]]:
        """Every consistent placement as lists of :class:`PersonPlacement` (available pre-step)."""
        return [
            [PersonPlacement(name=self.names[i], floor=perm[i], room=ROOM_NAME[rooms[i]])
             for i in range(3)]
            for perm, rooms in self._solutions()
        ]

    def step(self, answer: str) -> GameState:
        """Score the listed placements against the consistent set; terminal either way."""
        if self.status != "in_progress":
            raise GameOverError(f"Game {self.game_id} is already {self.status}")
        self.submitted = answer
        parsed = parse_answer(answer, self.names)
        gold = _gold_set(self.names, self._solutions())
        self.status = "correct" if (parsed is not None and parsed == gold) else "incorrect"
        return self.state()

    def state(self) -> GameState:
        sols = self.solution_placements() if self.status != "in_progress" else None
        return GameState(
            game_id=self.game_id, names=self.names,
            shown_floors=self.shown_floors, shown_rooms=self.shown_rooms,
            floor_ok=self.floor_ok, room_ok=self.room_ok,
            status=self.status, submitted=self.submitted, solutions=sols,
        )


# A pool of distinct first names — used only to vary the surface form of challenges.
NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
    "Karl", "Linda", "Mallory", "Niaj", "Olivia", "Peggy", "Quinn", "Rupert", "Sybil", "Trent",
    "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zane", "Amir", "Bianca", "Cesar", "Dora",
    "Elena", "Felix", "Gita", "Hugo", "Iris", "Jamal", "Kira", "Liam", "Maya", "Noah",
    "Omar", "Priya", "Rosa", "Sam", "Tara", "Usha", "Vera", "Will", "Xena", "Yusuf",
    "Zoe", "Ada", "Ben", "Cleo", "Dan", "Esha", "Finn", "Gwen", "Hana", "Ravi",
]


class TowerBank:
    """Generates tower challenges. Pure Python — no external assets.

    A challenge = a random *shown* placement + a random *true* placement (giving realizable
    feedback) + three distinct random names. ``sample_targets`` returns distinct encoded targets;
    with the name pool the surface space is far larger than the 1,920 distinct logic structures.
    """

    def __init__(self):
        import random

        self._rng = random.Random()

    def _placement(self, rng) -> tuple[list[int], list[int]]:
        floors = list(rng.choice(list(permutations(FLOORS))))
        rooms = [rng.randint(0, 1) for _ in range(3)]
        return floors, rooms

    def make_target(self, rng) -> str:
        names = rng.sample(NAMES, 3)
        shown_f, shown_r = self._placement(rng)
        true_f, true_r = self._placement(rng)
        floor_ok = [shown_f[i] == true_f[i] for i in range(3)]
        room_ok = [shown_r[i] == true_r[i] for i in range(3)]
        return encode_target(names, shown_f, shown_r, floor_ok, room_ok)

    def sample_targets(self, n: int, mode: str, rng) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        guard = 0
        while len(out) < n and guard < n * 100:
            guard += 1
            t = self.make_target(rng)
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def sample(self, mode: str) -> str:
        return self.make_target(self._rng)
