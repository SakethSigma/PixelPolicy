"""Tests for the uniform client — and that the in-process and HTTP transports agree."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.rhyme.client import HTTPRhymeClient, LocalRhymeClient, RhymeClient
from games.rhyme.game import RhymeBank, is_rhyme
from games.rhyme.server import app


@pytest.fixture(scope="module")
def bank():
    return RhymeBank()


@pytest.fixture
def local(bank):
    return LocalRhymeClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPRhymeClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, RhymeClient)
        assert isinstance(http, RhymeClient)


class TestParity:
    @pytest.mark.parametrize("word,answer", [("cat", "hat"), ("bright", "flight")])
    def test_free_correct_parity(self, local, http, word, answer):
        local.reset(word=word, variant="free")
        http.reset(word=word, variant="free")
        ls = local.step(answer)
        hs = http.step(answer)
        assert ls.status == hs.status == "correct"

    def test_free_incorrect_parity(self, local, http):
        local.reset(word="cat", variant="free")
        http.reset(word="cat", variant="free")
        assert local.step("dog").status == "incorrect"
        assert http.step("dog").status == "incorrect"


class TestBank:
    def test_split_is_disjoint_and_deterministic(self, bank):
        assert set(bank.train).isdisjoint(bank.val)
        again = RhymeBank()
        assert again.train == bank.train and again.val == bank.val

    def test_mcq_options_have_exactly_one_rhyme(self, bank):
        import random

        rng = random.Random(0)
        word = bank.sample_seed("train")
        opts = bank.mcq_options(word, rng)
        assert opts is not None and len(opts) == 5
        assert sum(1 for o in opts if is_rhyme(word, o)) == 1


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404
            assert tc.post("/step", json={"game_id": "nope", "answer": "x"}).status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": "cat", "variant": "free"}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": "hat"})
            again = tc.post("/step", json={"game_id": gid, "answer": "bat"})
            assert again.status_code == 400
