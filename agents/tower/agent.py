"""The Tower agent — the only game-aware code for word-skill game #9.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`TowerEnv` that adapts the game's ``TowerClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history``.

This is a programmatic deduction game (no ``<think>`` requested): the student learns to map a
proposed placement + feedback to the full set of consistent placements. ``parse_action`` returns
the body of the ``<answer>`` tag, which :func:`games.tower.game.parse_answer` scores.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.tower.render import render_observation  # the challenge text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class TowerAgent:
    """Single-turn Tower-deduction policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You solve a placement puzzle from feedback.\n"
        "\n"
        "A tower has 3 floors (1 = bottom, 3 = top); each floor has two rooms, Left and Right. "
        "Three people each live in a different room, and no two people share a floor.\n"
        "\n"
        "You are shown a proposed placement and, for each person, two flags: whether their FLOOR "
        "is correct (✓) or not (x), and whether their ROOM (Left/Right) is correct (✓) or not (x). "
        "Use the feedback to deduce every placement that is consistent with it:\n"
        "  - a wrong room means the person is in the OTHER room of whatever floor they are on;\n"
        "  - the floors are a permutation, so if some floors are wrong there may be more than one "
        "consistent arrangement — list them all.\n"
        "\n"
        "Give your final answer inside a single <answer> tag. List each consistent placement as a "
        "numbered block, one person per line, e.g.:\n"
        "<answer>\n"
        "solution 1:\n"
        "Alice: floor 3, Right\n"
        "Bob: floor 1, Left\n"
        "Carol: floor 2, Left\n"
        "</answer>\n"
        "If a second placement is also consistent, add a 'solution 2:' block. Output the <answer> "
        "tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the challenge. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the body of the last ``<answer>…</answer>`` tag; ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip() if tagged else ""


class TowerEnv:
    """Adapt a ``TowerClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
