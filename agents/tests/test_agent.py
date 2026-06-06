"""Agent + rollout tests with a fake backend — no model server, no network.

These exercise the pieces a trainer also imports (build_messages / parse_action) and the
generic loop, proving they work without `openai` or a running vLLM server.
"""

from __future__ import annotations

import pytest

from agents.base import Completion
from agents.rollout import run_episode, run_eval, win_rate
from agents.wordle.agent import WordleAgent, WordleEnv
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank


class FakeBackend:
    """Returns a scripted reply per call, ignoring the prompt. Satisfies LLMBackend."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0

    def generate(self, prompts, **_):
        out = []
        for _ in prompts:
            text = self._responses[min(self._i, len(self._responses) - 1)]
            self._i += 1
            out.append(Completion(text=text))
        return out


@pytest.fixture(scope="module")
def bank() -> WordBank:
    return WordBank()


def _env(bank: WordBank, *, word: str) -> WordleEnv:
    env = WordleEnv(LocalWordleClient(bank))
    env.reset(word=word)
    return env


# ── parse_action ──────────────────────────────────────────────────────────────--


def test_parse_prefers_last_guess_tag():
    agent = WordleAgent()
    assert agent.parse_action("<think>hmm</think><guess>Crane</guess>") == "crane"
    assert agent.parse_action("<guess>about</guess> no, <guess>Slate</guess>") == "slate"


def test_parse_is_strict_no_tag_returns_empty():
    agent = WordleAgent()
    # Strict: without a <guess> tag we do NOT dig a word out of the reasoning — return "".
    assert agent.parse_action("<think>maybe crane then slate") == ""
    assert agent.parse_action("I'll guess slate") == ""


def test_parse_empty_is_invalid_round_in_env(bank: WordBank):
    agent = WordleAgent()
    env = _env(bank, word=bank.train[0])
    state = env.step(agent.parse_action("no tag here"))  # parse → "" → consumed invalid round
    assert state.current_round == 1
    assert state.rounds[-1].error is not None
    assert state.rounds[-1].feedback == []


# ── rollout ───────────────────────────────────────────────────────────────────--


def test_episode_win_in_one_turn(bank: WordBank):
    target = bank.train[0]
    traj = run_episode(WordleAgent(), _env(bank, word=target), FakeBackend([f"<guess>{target}</guess>"]).generate)
    assert traj.final.status == "won"
    assert len(traj.turns) == 1
    assert traj.turns[0].action == target


def test_episode_loss_consumes_all_rounds(bank: WordBank):
    target, wrong = bank.train[0], bank.train[1]
    assert target != wrong
    traj = run_episode(
        WordleAgent(), _env(bank, word=target), FakeBackend([f"<guess>{wrong}</guess>"] * 6).generate
    )
    assert traj.final.status == "lost"
    assert len(traj.turns) == 6


def test_history_is_threaded_into_later_prompts(bank: WordBank):
    target, wrong = bank.train[0], bank.train[1]
    traj = run_episode(
        WordleAgent(), _env(bank, word=target), FakeBackend([f"<guess>{wrong}</guess>"] * 6).generate
    )
    first = traj.turns[0].messages
    second = traj.turns[1].messages
    # Turn 1 is just system + opening ask; turn 2 carries the model's reply + feedback.
    assert [m["role"] for m in first] == ["system", "user"]
    assert any(m["role"] == "assistant" for m in second)
    assert len(second) > len(first)


def test_run_eval_concurrent_win_rate(bank: WordBank):
    target = bank.train[0]
    pairs = [(WordleAgent(), _env(bank, word=target)) for _ in range(5)]
    trajs = run_eval(pairs, FakeBackend([f"<guess>{target}</guess>"]).generate, concurrency=4)
    assert len(trajs) == 5
    assert win_rate(trajs) == 1.0
