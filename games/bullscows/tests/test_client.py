"""Tests for the uniform Bulls & Cows client + Local/HTTP parity."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.bullscows.client import (
    BullsCowsClient,
    HTTPBullsCowsClient,
    LocalBullsCowsClient,
)
from games.bullscows.server import app


@pytest.fixture
def local():
    return LocalBullsCowsClient()


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPBullsCowsClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, BullsCowsClient)
        assert isinstance(http, BullsCowsClient)


class TestParity:
    def test_win_and_feedback_parity(self, local, http):
        for c in (local, http):
            c.reset(word="1234")
            mid = c.guess("1243")
            assert mid.status == "in_progress" and mid.rounds[-1].bulls == 2 and mid.rounds[-1].cows == 2
            end = c.guess("1234")
            assert end.status == "won" and end.secret == "1234"


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_guess_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": "1234"}).json()["game_id"]
            tc.post("/guess", json={"game_id": gid, "guess": "1234"})
            again = tc.post("/guess", json={"game_id": gid, "guess": "5678"})
            assert again.status_code == 400
