"""Tests for the uniform client — and that the in-process and HTTP transports agree."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from games.tower.client import HTTPTowerClient, LocalTowerClient, TowerClient
from games.tower.game import TowerBank, decode_target, solve
from games.tower.render import render_solutions
from games.tower.server import app

# All floors wrong -> two solutions; the model must list both.
_TARGET = "Alice,Bob,Carol;1L,2L,3L;01,01,01"


def _correct_answer(target: str) -> str:
    from games.tower.game import TowerGame
    names, sf, sr, fok, rok = decode_target(target)
    return render_solutions(TowerGame(names, sf, sr, fok, rok, "x").solution_placements())


@pytest.fixture
def local():
    return LocalTowerClient(TowerBank())


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPTowerClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, TowerClient)
        assert isinstance(http, TowerClient)


class TestParity:
    def test_two_solution_parity(self, local, http):
        ans = _correct_answer(_TARGET)
        local.reset(word=_TARGET)
        http.reset(word=_TARGET)
        ls, hs = local.step(ans), http.step(ans)
        assert ls.status == hs.status == "correct"
        assert len(ls.solutions) == len(hs.solutions) == 2

    def test_incorrect_parity(self, local, http):
        local.reset(word=_TARGET)
        http.reset(word=_TARGET)
        bad = "solution 1:\nAlice: floor 2, Left\nBob: floor 3, Left\nCarol: floor 1, Left"
        assert local.step(bad).status == http.step(bad).status == "incorrect"


class TestBank:
    def test_sample_targets_distinct_and_valid(self):
        bank = TowerBank()
        rng = random.Random(0)
        targets = bank.sample_targets(500, "train", rng)
        assert len(targets) == 500 == len(set(targets))
        # every generated challenge is realizable (>=1 solution) and has at most 2.
        for t in targets:
            _, sf, sr, fok, rok = decode_target(t)
            n = len(solve(sf, sr, fok, rok))
            assert 1 <= n <= 2


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            ans = _correct_answer(_TARGET)
            gid = tc.post("/reset", json={"word": _TARGET}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": ans})
            again = tc.post("/step", json={"game_id": gid, "answer": ans})
            assert again.status_code == 400
