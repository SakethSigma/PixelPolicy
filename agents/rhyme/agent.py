"""The Rhymes agent — the only game-aware code for word-skill game #5.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a thin
:class:`RhymeEnv` that adapts the game's ``RhymeClient`` (verb: ``step``) onto the generic
:class:`~agents.base.Env` protocol. Single-turn, so ``build_messages`` ignores ``history`` and
emits one user turn (the self-contained variant the agent doc sanctions). One prompt covers both
variants — the observation says whether options are listed.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.rhyme.render import render_observation  # the challenge text a human also reads

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class RhymeAgent:
    """Single-turn Rhymes policy adapter. Stateless: a pure function of state."""

    system_prompt = (
        "You are a rhyming expert. You are given an English word.\n"
        "\n"
        "- If a list of options is provided, choose the single option that rhymes with the word.\n"
        "- Otherwise, name any common English word that rhymes with the given word.\n"
        "\n"
        "Two words rhyme when they share the same ending sound (from the last stressed vowel "
        "onward), like 'bright' and 'flight'. A word does not rhyme with itself.\n"
        "\n"
        "Give your final answer inside a single <answer> tag, e.g. <answer>flight</answer>.\n"
        "Output the <answer> tag exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """One system turn + one user turn carrying the challenge. ``history`` is unused."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_observation(state)},
        ]

    def parse_action(self, text: str) -> str:
        """Return the word inside the last ``<answer>…</answer>`` (lowercased); ``""`` if absent."""
        tagged = _ANSWER_TAG.findall(text)
        return tagged[-1].strip().lower() if tagged else ""


class RhymeEnv:
    """Adapt a ``RhymeClient`` to the generic :class:`~agents.base.Env` protocol."""

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.step(action)

    def state(self):
        return self._client.state()
