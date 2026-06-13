"""The Crossword-fill agent — the only game-aware code for word-skill game #6.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`CrosswordEnv` that adapts the game's ``CrosswordClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history``.

This is a *reasoning* game: distilled at high adaptive-thinking effort, the model opens
``<think>`` via its chat template and reasons from the definition + revealed letters before the
final ``<answer>``. A solved trace with no ``<think>`` block is rejected at distillation time
(``GameSpec.require_think``).
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.crossword.render import render_observation  # the clue text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class CrosswordAgent:
    """Single-turn Crossword-fill policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You solve a crossword clue.\n"
        "\n"
        "You are given a definition, the word's length, and a pattern. In the pattern, revealed "
        "letters are shown and hidden letters are written as '_'. Find the single English word "
        "that:\n"
        "  - has exactly the given length,\n"
        "  - matches every revealed letter in its position, and\n"
        "  - means what the definition says.\n"
        "\n"
        "Think it through, then give your final answer inside a single <answer> tag, e.g. "
        "<answer>crane</answer>.\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the clue. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the word inside the last ``<answer>…</answer>`` (lowercased); ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip().lower() if tagged else ""


class CrosswordEnv:
    """Adapt a ``CrosswordClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
