"""Tests for the uniform client in ``games.wordle.client`` — and, crucially, that
the in-process and HTTP transports behave identically."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.wordle.client import (
    HTTPWordleClient,
    LocalWordleClient,
    WordleClient,
    make_local_group,
)
from games.wordle.game import GameState, LetterFeedback, WordBank
from games.wordle.server import app


@pytest.fixture(scope="module")
def bank():
    return WordBank()


@pytest.fixture
def local(bank):
    return LocalWordleClient(bank)


@pytest.fixture
def http():
    # Drive HTTPWordleClient through FastAPI's TestClient transport — no real socket.
    with TestClient(app) as tc:
        yield HTTPWordleClient(client=tc)


def drive(client: WordleClient, word: str, guesses: list[str]) -> GameState:
    client.reset(word=word)
    state = client.state()
    for g in guesses:
        state = client.guess(g)
    return state


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, WordleClient)
        assert isinstance(http, WordleClient)


class TestLocal:
    def test_reset_then_guess_and_state(self, local):
        local.reset(word="apple")
        assert local.state().current_round == 0
        st = local.guess("crane")
        assert st.current_round == 1
        assert local.state().current_round == 1

    def test_guess_before_reset_raises(self, bank):
        with pytest.raises(RuntimeError):
            LocalWordleClient(bank).guess("crane")

    def test_re_reset_starts_fresh_episode(self, local):
        local.reset(word="apple")
        local.guess("crane")
        local.reset(word="moist")
        assert local.state().current_round == 0

    def test_is_context_manager(self, bank):
        with LocalWordleClient(bank) as c:
            c.reset(word="apple")
            assert c.guess("apple").status == "won"


class TestHTTP:
    def test_returns_pydantic_gamestate(self, http):
        http.reset(word="apple")
        assert isinstance(http.guess("crane"), GameState)

    def test_feedback_enum_roundtrips_over_http(self, http):
        http.reset(word="apple")
        st = http.guess("crane")
        assert all(isinstance(f, LetterFeedback) for f in st.rounds[0].feedback)

    def test_is_context_manager(self):
        with TestClient(app) as tc:
            with HTTPWordleClient(client=tc) as c:
                c.reset(word="apple")
                assert c.guess("apple").status == "won"


class TestParity:
    """The whole point: identical input → identical state on both transports."""

    def test_identical_sequence(self, local, http):
        seq = ["crane", "zz", "zzzzz", "moist", "apple"]  # valid, length-bad, vocab-bad, valid, win
        a = drive(local, "apple", seq)
        b = drive(http, "apple", seq)
        # game_id is transport-assigned; everything else must match exactly.
        assert a.model_dump(exclude={"game_id"}) == b.model_dump(exclude={"game_id"})
        assert a.status == "won"

    def test_whitespace_and_case_normalized_both(self, local, http):
        a = drive(local, "apple", ["  ApPlE "])
        b = drive(http, "apple", ["  ApPlE "])
        assert a.status == b.status == "won"
        assert a.rounds[0].guess == b.rounds[0].guess == "APPLE"


class TestGrouping:
    def test_grpo_group_shares_pinned_target(self, bank):
        clients = make_local_group(4, word="crane", word_bank=bank)
        for c in clients:
            assert c.guess("crane").target == "CRANE"  # same target revealed on win
