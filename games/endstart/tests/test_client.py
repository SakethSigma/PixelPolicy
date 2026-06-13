"""Tests for the uniform client + Local/HTTP parity + unbiased challenge construction."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from games.endstart.client import EndstartClient, HTTPEndstartClient, LocalEndstartClient
from games.endstart.game import EndstartBank, correct_option
from games.endstart.server import app

_TARGET = "mango;river,oasis,tundra,cliff,marsh"


@pytest.fixture(scope="module")
def bank():
    return EndstartBank()


@pytest.fixture
def local(bank):
    return LocalEndstartClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPEndstartClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, EndstartClient)
        assert isinstance(http, EndstartClient)


class TestParity:
    def test_correct_parity(self, local, http):
        local.reset(word=_TARGET)
        http.reset(word=_TARGET)
        assert local.step("<answer>oasis</answer>").status == http.step("<answer>oasis</answer>").status == "correct"

    def test_incorrect_parity(self, local, http):
        local.reset(word=_TARGET)
        http.reset(word=_TARGET)
        assert local.step("river").status == http.step("river").status == "incorrect"


class TestBank:
    def test_challenges_have_unique_answer_and_random_position(self, bank):
        rng = random.Random(0)
        positions = set()
        for _ in range(50):
            word1, options = bank.make_challenge("train", rng)
            matches = [o for o in options if o[0] == word1[-1]]
            assert len(matches) == 1                      # exactly one correct option
            assert correct_option(word1, options) == matches[0]
            positions.add(options.index(matches[0]))
        # the answer is not pinned to one slot (unbiased position)
        assert len(positions) >= 3
