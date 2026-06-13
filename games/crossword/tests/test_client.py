"""Tests for the uniform client — and that the in-process and HTTP transports agree.

Needs the committed meanings.jsonl asset (``python -m games.wordvocab.build_meanings``).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.crossword.client import (
    CrosswordClient,
    HTTPCrosswordClient,
    LocalCrosswordClient,
)
from games.crossword.game import CrosswordBank
from games.crossword.server import app


@pytest.fixture(scope="module")
def bank():
    return CrosswordBank()


@pytest.fixture
def local(bank):
    return LocalCrosswordClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPCrosswordClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, CrosswordClient)
        assert isinstance(http, CrosswordClient)


class TestParity:
    def test_correct_parity(self, local, http):
        ls = local.reset(word="crane")
        hs = http.reset(word="crane")
        assert ls.pattern == hs.pattern and ls.length == hs.length == 5
        assert local.step("<answer>crane</answer>").status == "correct"
        assert http.step("<answer>crane</answer>").status == "correct"

    def test_incorrect_parity(self, local, http):
        local.reset(word="crane")
        http.reset(word="crane")
        assert local.step("<answer>plane</answer>").status == "incorrect"
        assert http.step("<answer>plane</answer>").status == "incorrect"


class TestBank:
    def test_pools_nonempty_and_have_definitions(self, bank):
        assert bank.wordle_words and bank.general_words
        assert bank.definition(bank.wordle_words[0])

    def test_sample_targets_mix(self, bank):
        import random

        rng = random.Random(0)
        targets = bank.sample_targets(200, "train", rng)
        assert len(targets) == 200 == len(set(targets))
        n_wordle = sum(1 for t in targets if t in bank.wordle)
        assert 90 <= n_wordle <= 110  # ~half from the Wordle vocab


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": "crane"}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": "<answer>crane</answer>"})
            again = tc.post("/step", json={"game_id": gid, "answer": "x"})
            assert again.status_code == 400
