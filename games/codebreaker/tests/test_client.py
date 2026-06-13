"""Tests for the uniform Codebreaker client + Local/HTTP parity."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.codebreaker.client import (
    CodebreakerClient,
    HTTPCodebreakerClient,
    LocalCodebreakerClient,
)
from games.codebreaker.server import app


@pytest.fixture
def local():
    return LocalCodebreakerClient()


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPCodebreakerClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, CodebreakerClient)
        assert isinstance(http, CodebreakerClient)


class TestParity:
    def test_win_and_feedback_parity(self, local, http):
        for c in (local, http):
            c.reset(word="ACEF")
            mid = c.guess("AAAA")
            assert mid.status == "in_progress" and mid.rounds[-1].feedback
            end = c.guess("ACEF")
            assert end.status == "won" and end.secret == "ACEF"


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_guess_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": "ACEF"}).json()["game_id"]
            tc.post("/guess", json={"game_id": gid, "guess": "ACEF"})
            again = tc.post("/guess", json={"game_id": gid, "guess": "BBBB"})
            assert again.status_code == 400
