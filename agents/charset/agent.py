"""The Character-set agent — the only game-aware code for word-skill game #7.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`CharsetEnv` that adapts the game's ``CharsetClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history`` and
emits one user turn — the self-contained variant the agent doc sanctions. This is a mechanical
task with a short answer, so no ``<think>`` is requested.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.charset.render import render_observation  # the challenge text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class CharsetAgent:
    """Single-turn Character-set policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You track which letters of the alphabet a set of words uses.\n"
        "\n"
        "You are given a few English words. Considering all of them together, report:\n"
        "  - the USED letters: every distinct letter of a-z that appears in at least one word, and\n"
        "  - the UNUSED letters: every letter of a-z that appears in none of the words.\n"
        "\n"
        "Rules:\n"
        "- Consider the 26 letters a-z; every letter is either used or unused (never both).\n"
        "- List each letter once, in alphabetical order, UPPERCASE, separated by single spaces.\n"
        "\n"
        "Give your final answer in exactly this format, inside a single <answer> tag:\n"
        "<answer>\n"
        "used (U): space-separated UPPERCASE letters\n"
        "unused (26-U): space-separated UPPERCASE letters\n"
        "</answer>\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the words. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the body of the last ``<answer>…</answer>`` tag; ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip() if tagged else ""


class CharsetEnv:
    """Adapt a ``CharsetClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
