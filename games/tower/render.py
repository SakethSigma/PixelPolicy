"""Plain-text rendering of a Tower ``GameState``.

Dependency-free on purpose: this is the text a human reads in a bare terminal *and* the
observation an LLM policy is shown. :func:`render_solutions` is the canonical ``<answer>`` block
the synthetic teacher writes; :func:`games.tower.game.parse_answer` reads it back.
"""

from __future__ import annotations

from games.tower.game import GameState, ROOM_NAME, PersonPlacement

_SETUP = (
    "A tower has 3 floors (1 = bottom, 3 = top); each floor has two rooms, Left and Right.\n"
    "Three people each live in a different room, and no two people share a floor.\n"
)


def render_observation(state: GameState) -> str:
    """The challenge a human and the model read: the proposed placement + its per-person feedback."""
    lines = [_SETUP, "A guess was checked. 'floor ✓/x' = is their floor right; 'room ✓/x' = is "
             "their room (Left/Right) right:\n"]
    for i, name in enumerate(state.names):
        room = ROOM_NAME[state.shown_rooms[i]]
        f = "✓" if state.floor_ok[i] else "x"
        r = "✓" if state.room_ok[i] else "x"
        lines.append(f"{name} — guess: floor {state.shown_floors[i]}, {room}  ->  floor {f}, room {r}")
    lines.append("\nList every placement (each person's floor and room) consistent with this feedback.")
    return "\n".join(lines)


def render_solutions(solutions: list[list[PersonPlacement]]) -> str:
    """The canonical ``<answer>`` body: one numbered block per consistent placement."""
    blocks = []
    for n, sol in enumerate(solutions, start=1):
        rows = "\n".join(f"{p.name}: floor {p.floor}, {p.room}" for p in sol)
        blocks.append(f"solution {n}:\n{rows}")
    return "\n".join(blocks)
