"""Tests for the uniform client — and that the in-process and HTTP transports agree."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from games.anagram.client import AnagramClient, HTTPAnagramClient, LocalAnagramClient
from games.anagram.game import AnagramBank, are_anagrams, decode_pair
from games.anagram.server import app


@pytest.fixture(scope="module")
def bank():
    return AnagramBank()


@pytest.fixture
def local(bank):
    return LocalAnagramClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPAnagramClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, AnagramClient)
        assert isinstance(http, AnagramClient)


class TestParity:
    def test_pinned_pair_parity(self, local, http):
        local.reset(word="listen,silent")
        http.reset(word="listen,silent")
        assert local.step("yes").status == http.step("yes").status == "correct"


class TestBank:
    def test_split_is_disjoint_and_deterministic(self, bank):
        assert set(bank.train).isdisjoint(bank.val)
        again = AnagramBank()
        assert again.train == bank.train and again.val == bank.val

    def test_sample_targets_mix_and_distinct(self, bank):
        rng = random.Random(0)
        targets = bank.sample_targets(200, "train", rng, pos_fraction=0.4)
        assert len(targets) == 200 == len(set(targets))
        pos = sum(1 for t in targets if are_anagrams(*decode_pair(t)))
        # ~40% positive (allow slack from rounding/dedup).
        assert 60 <= pos <= 100

    def test_positives_are_anagrams_negatives_are_not(self, bank):
        rng = random.Random(1)
        w1, w2 = bank.positive_pair("train", rng)
        assert are_anagrams(w1, w2)
        n1, n2 = bank.negative_pair("train", rng, hard=True)
        assert not are_anagrams(n1, n2)


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": "listen,silent"}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": "yes"})
            again = tc.post("/step", json={"game_id": gid, "answer": "no"})
            assert again.status_code == 400
