"""Tests for the uniform client — and that the in-process and HTTP transports agree."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.validity.client import HTTPValidityClient, LocalValidityClient, ValidityClient
from games.validity.game import ValidityBank, is_valid_word
from games.validity.render import render_answer
from games.validity.server import app


@pytest.fixture(scope="module")
def bank():
    return ValidityBank()


@pytest.fixture
def local(bank):
    return LocalValidityClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPValidityClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, ValidityClient)
        assert isinstance(http, ValidityClient)


class TestParity:
    def test_valid_word_parity(self, local, http):
        answer = render_answer(True, "to set fire to")
        local.reset(word="kindle")
        http.reset(word="kindle")
        assert local.step(answer).status == http.step(answer).status == "correct"

    def test_invalid_word_parity(self, local, http):
        answer = render_answer(False)
        local.reset(word="zzzqxv")
        http.reset(word="zzzqxv")
        assert local.step(answer).status == http.step(answer).status == "correct"


class TestBank:
    def test_valid_words_are_real(self, bank):
        for w in bank.valid_words[:50]:
            assert is_valid_word(w)

    def test_split_is_disjoint_and_deterministic(self, bank):
        assert set(bank.train).isdisjoint(bank.val)
        again = ValidityBank()
        assert again.train == bank.train and again.val == bank.val

    def test_pseudo_words_are_invalid(self, bank):
        import random

        rng = random.Random(0)
        for _ in range(20):
            w = bank.make_pseudo_word(rng)
            assert not is_valid_word(w) and w not in bank.wordle


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": "kindle"}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": render_answer(True, "x")})
            again = tc.post("/step", json={"game_id": gid, "answer": "x"})
            assert again.status_code == 400
