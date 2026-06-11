"""Tests for the uniform client — and that the in-process and HTTP transports agree."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.charcount.client import (
    CharCountClient,
    HTTPCharCountClient,
    LocalCharCountClient,
)
from games.charcount.game import CharCountBank, analyze
from games.charcount.render import render_answer
from games.charcount.server import app


@pytest.fixture(scope="module")
def bank():
    return CharCountBank()


@pytest.fixture
def local(bank):
    return LocalCharCountClient(bank)


@pytest.fixture
def http():
    # Drive HTTPCharCountClient through FastAPI's TestClient transport — no real socket.
    with TestClient(app) as tc:
        yield HTTPCharCountClient(client=tc)


def drive(client: CharCountClient, word: str, answer: str):
    client.reset(word=word)
    return client.step(answer)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, CharCountClient)
        assert isinstance(http, CharCountClient)


class TestParity:
    @pytest.mark.parametrize("word", ["planet", "banana", "rhythm", "queue"])
    def test_correct_answer_parity(self, local, http, word):
        answer = render_answer(analyze(word))
        ls = drive(local, word, answer)
        hs = drive(http, word, answer)
        assert ls.status == hs.status == "correct"
        assert ls.solution == hs.solution == analyze(word)
        assert ls.word == hs.word == word

    def test_incorrect_answer_parity(self, local, http):
        bad = "length: 1\nvowels: a\nconsonants: z"
        assert drive(local, "planet", bad).status == "incorrect"
        assert drive(http, "planet", bad).status == "incorrect"


class TestBank:
    def test_split_is_disjoint_and_deterministic(self, bank):
        assert set(bank.train).isdisjoint(bank.val)
        # Re-deriving the split yields the identical pools (salted hash is byte-stable).
        again = CharCountBank()
        assert again.train == bank.train and again.val == bank.val

    def test_sample_pins_are_in_pool(self, bank):
        assert bank.sample("train") in set(bank.train)
        assert bank.sample("val") in set(bank.val)


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404
            assert tc.post("/step", json={"game_id": "nope", "answer": "x"}).status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"mode": "train", "word": "planet"}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": render_answer(analyze("planet"))})
            again = tc.post("/step", json={"game_id": gid, "answer": "x"})
            assert again.status_code == 400
