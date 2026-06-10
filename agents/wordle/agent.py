"""The Wordle agent — the only game-aware code in the agents layer.

Two pure functions (``build_messages`` / ``parse_action``) plus a system prompt, and a
thin :class:`WordleEnv` that adapts the game's ``WordleClient`` (verb: ``guess``) onto
the generic :class:`~agents.base.Env` protocol (verb: ``step``). Nothing about models or
networks lives here; that is the backend / injected ``generate``.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import Turn
from games.wordle.render import render_round  # the per-round feedback line a human also sees

# Qwen-style format: reasoning in <think>…</think>, the final guess in <guess>…</guess>.
# With thinking enabled the chat template opens <think> in the *prompt*, so the model's reply
# carries its reasoning, a closing </think>, then the <guess>. The system prompt only asks for
# the final <guess>; parse_action reads strictly from that tag.
_GUESS_TAG = re.compile(r"<guess>\s*([A-Za-z]+)\s*</guess>", re.IGNORECASE | re.DOTALL)


class WordleAgent:
    """Multi-turn Wordle policy adapter. Stateless: a pure function of (state, history)."""

    system_prompt = (
        "You are an expert Wordle player.\n"
        "\n"
        "I have chosen a secret 5-letter English word. You have 6 guesses to find it. "
        "After each guess, every letter is scored:\n"
        "  ✓  correct letter in the correct position\n"
        "  -  correct letter but in the wrong position\n"
        "  x  the letter is not in the secret word at all\n"
        "\n"
        "Rules:\n"
        "- Each guess must be a real 5-letter English word.\n"
        "- An invalid guess (wrong length, or not a real word) still uses up one of your "
        "6 guesses, so never waste one.\n"
        "- Use every clue: keep ✓ letters in their position, move - letters to a different "
        "position (they ARE in the word), and never reuse x letters.\n"
        "- A letter can repeat; feedback is per position.\n"
        "- With several guesses left but many words still consistent with the clues, it can "
        "pay to explore untried letters rather than commit to one likely answer — eliminating "
        "possibilities faster improves your odds of winning.\n"
        "- Vary your opening guess: choose among several strong starting words (those rich in "
        "common letters) rather than always opening with the same one.\n"
        "\n"
        "Think through the clues, then give your final answer on its own line as:\n"
        "<guess>word</guess>\n"
        "where word is a single lowercase 5-letter English word. Output the <guess> tag "
        "exactly once, as the very last thing you write."
    )

    def build_messages(self, state: Any, history: list[Turn] = ()) -> list[dict]:
        """Replay the episode as a conversation: system → first ask → (reply, feedback)*.

        The model sees its own prior replies (reasoning) and only the *new* feedback each
        turn. ``history`` is supplied by the rollout; nothing is stored on the agent.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Make your first guess."},
        ]
        for turn in history:
            # Replay only the normalized guess, never the prior chain-of-thought. Qwen's
            # multi-turn guidance is to drop old reasoning; it also keeps context tiny and
            # bounded even when a turn's reasoning ran long without terminating (Qwen3.5-0.8B
            # is prone to thinking loops). `turn.action` is already parse_action(response).
            guess = f"<guess>{turn.action}</guess>" if turn.action else ""
            messages.append({"role": "assistant", "content": guess})
            messages.append({"role": "user", "content": self._feedback(turn.state)})
        return messages

    def parse_action(self, text: str) -> str:
        """Extract the guess strictly from the last ``<guess>…</guess>`` tag.

        No lenient fallback: the model must follow the format. If there is no ``<guess>``
        tag, we return ``""`` — the env then counts it a consumed round with an
        ``inadequate length`` error, exactly the "invalid costs a round" rule a human plays
        under. We deliberately do **not** guess a word out of the reasoning text.
        """
        tagged = _GUESS_TAG.findall(text)
        if tagged:
            return tagged[-1].lower()
        return ""

    @staticmethod
    def _feedback(state: Any) -> str:
        """The user message after a guess: that guess's feedback + how many rounds remain."""
        line = render_round(state.rounds[-1])
        if state.status == "in_progress":
            remaining = state.max_rounds - state.current_round
            return f"{line}\nRounds left: {remaining}. Make your next guess."
        return line


class WordleEnv:
    """Adapt a ``WordleClient`` to the generic :class:`~agents.base.Env` protocol.

    Maps ``step`` → ``guess`` and forwards ``reset`` / ``state``. This is the one place
    that knows Wordle's env verb, keeping ``agents/rollout.py`` game-agnostic.
    """

    def __init__(self, client):
        self._client = client

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def step(self, action: str):
        return self._client.guess(action)

    def state(self):
        return self._client.state()
