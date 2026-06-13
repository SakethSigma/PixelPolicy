"""The Mistake-identification agent — the only game-aware code for word-skill game #8.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`MistakeEnv` that adapts the game's ``MistakeClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history``.

This is a *reasoning* game: distilled at high adaptive-thinking effort, the model reasons over the
board's feedback before reporting mistakes. A solved trace with no ``<think>`` block is rejected
at distillation time (``GameSpec.require_think``). ``parse_action`` returns the body of the
``<answer>`` tag — the mistakes report that :func:`games.mistakeid.game.parse_report` scores.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.mistakeid.render import render_observation  # the board text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class MistakeAgent:
    """Single-turn Mistake-identification policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You are a Wordle coach reviewing a proposed guess.\n"
        "\n"
        "Each past guess is scored per letter:\n"
        "  ✓  the letter is correct and in the right spot\n"
        "  -  the letter is in the word but in the wrong spot\n"
        "  x  the letter is not in the word at all\n"
        "\n"
        "A proposed guess REPEATS A MISTAKE when it throws away what earlier feedback revealed:\n"
        "  - a 'grey' mistake: it uses a letter that earlier feedback marked x (not in the word);\n"
        "  - a 'yellow' mistake: it puts a letter in a position where that letter was already\n"
        "    marked - (yellow) — the letter is in the word but known not to go in that exact spot.\n"
        "Only these two repeated mistakes count; other suboptimal play does not.\n"
        "\n"
        "Decide whether the proposed guess repeats any such mistake. Think it through, then give\n"
        "your final answer inside a single <answer> tag.\n"
        "\n"
        "If there are no repeated mistakes:\n"
        "<answer>\n"
        "mistakes: no\n"
        "</answer>\n"
        "\n"
        "If there are, list one per line (position is 1-5, left to right):\n"
        "<answer>\n"
        "mistakes: yes\n"
        "position 4, letter R, grey\n"
        "position 1, letter A, yellow\n"
        "</answer>\n"
        "\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the board + proposed guess. ``history`` unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the body of the last ``<answer>…</answer>`` tag (the report); ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip() if tagged else ""


class MistakeEnv:
    """Adapt a ``MistakeClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
