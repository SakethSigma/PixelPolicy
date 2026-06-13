"""Tests for the uniform client — and that the in-process and HTTP transports agree."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.charset.client import CharsetClient, HTTPCharsetClient, LocalCharsetClient
from games.charset.game import CharsetBank, analyze
from games.charset.render import render_answer
from games.charset.server import app


@pytest.fixture(scope="module")
def bank():
    return CharsetBank()


@pytest.fixture
def local(bank):
    return LocalCharsetClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPCharsetClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, CharsetClient)
        assert isinstance(http, CharsetClient)


class TestParity:
    def test_correct_parity(self, local, http):
        used, unused = analyze(["cat", "planet"])
        answer = render_answer(used, unused)
        local.reset(word="cat,planet")
        http.reset(word="cat,planet")
        assert local.step(answer).status == http.step(answer).status == "correct"

    def test_incorrect_parity(self, local, http):
        local.reset(word="cat,planet")
        http.reset(word="cat,planet")
        assert local.step("used: a\nunused: b").status == "incorrect"
        assert http.step("used: a\nunused: b").status == "incorrect"


class TestBank:
    def test_pools_split_and_mix(self, bank):
        assert bank.five_train and bank.nonfive_train
        import random

        words = bank.make_words("train", random.Random(0), k=3)
        assert len(words) == 3
        assert any(len(w) == 5 for w in words)        # a Wordle word
        assert any(len(w) != 5 for w in words)        # an "otherwise" word

    def test_split_deterministic(self, bank):
        again = CharsetBank()
        assert again.five_train == bank.five_train and again.nonfive_train == bank.nonfive_train


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            from games.charset.game import analyze as _an
            used, unused = _an(["cat", "planet"])
            gid = tc.post("/reset", json={"word": "cat,planet"}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": render_answer(used, unused)})
            again = tc.post("/step", json={"game_id": gid, "answer": "x"})
            assert again.status_code == 400
