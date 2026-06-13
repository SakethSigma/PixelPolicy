"""Tests for the uniform client + Local/HTTP parity + balanced bank construction.

Needs game-wordle (compute_feedback + WordBank).
"""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from games.consistency.client import (
    ConsistencyClient,
    HTTPConsistencyClient,
    LocalConsistencyClient,
)
from games.consistency.game import ConsistencyBank, encode_target, feedback_str, is_consistent
from games.consistency.server import app

# A pinned board where the candidate (the target) is consistent → answer "yes".
_TARGET = encode_target([("CRANE", feedback_str("crane", "plant"))], "plant")


@pytest.fixture(scope="module")
def bank():
    return ConsistencyBank()


@pytest.fixture
def local(bank):
    return LocalConsistencyClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPConsistencyClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, ConsistencyClient)
        assert isinstance(http, ConsistencyClient)


class TestParity:
    def test_yes_parity(self, local, http):
        local.reset(word=_TARGET)
        http.reset(word=_TARGET)
        assert local.step("<answer>yes</answer>").status == http.step("<answer>yes</answer>").status == "correct"


class TestBank:
    def test_make_challenge_label_matches_request(self, bank):
        rng = random.Random(0)
        for want in (True, False):
            rows, cand = bank.make_challenge(rng, want_consistent=want)
            assert is_consistent(cand, rows) == want

    def test_sample_targets_balanced(self, bank):
        from games.consistency.game import decode_target
        rng = random.Random(1)
        targets = bank.sample_targets(100, "train", rng)
        assert len(targets) == 100 == len(set(targets))
        pos = sum(1 for t in targets if is_consistent(*reversed(decode_target(t))))
        assert 40 <= pos <= 60        # ~50/50


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": _TARGET}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": "<answer>yes</answer>"})
            again = tc.post("/step", json={"game_id": gid, "answer": "<answer>no</answer>"})
            assert again.status_code == 400
