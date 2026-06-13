"""The Anagrams agent — the only game-aware code for word-skill game #3.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`AnagramEnv` that adapts the game's ``AnagramClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history``.

This is a *reasoning* game: distilled at high adaptive-thinking effort, the model opens
``<think>`` via its chat template and reasons before the final ``<answer>``. A solved trace
with no ``<think>`` block is rejected at distillation time (``GameSpec.require_think``). The
prompt deliberately does **not** tell the model *how* to decide (e.g. by sorting letters) — it
must work out the multiset comparison itself, which is the skill we distil.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.anagram.render import render_observation  # the challenge text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class AnagramAgent:
    """Single-turn Anagrams policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You determine whether two words are anagrams of each other.\n"
        "\n"
        "Two words are anagrams if one can be formed by rearranging all the letters of the "
        "other: they contain exactly the same letters, each used the same number of times "
        "(and so they always have the same length).\n"
        "\n"
        "Think it through, then give your final answer inside a single <answer> tag as "
        "<answer>yes</answer> or <answer>no</answer>.\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the two words. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the verdict inside the last ``<answer>…</answer>`` (lowercased); ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip().lower() if tagged else ""


class AnagramEnv:
    """Adapt an ``AnagramClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
