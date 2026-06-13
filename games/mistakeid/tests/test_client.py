"""Tests for the uniform client — and that the in-process and HTTP transports agree.

Needs the committed challenges.jsonl asset (``python -m games.mistakeid.build_challenges``).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.mistakeid.client import HTTPMistakeClient, LocalMistakeClient, MistakeClient
from games.mistakeid.game import MistakeBank
from games.mistakeid.server import app

_MISTAKE_TARGET = "crane:xxxxx;track"   # r,a,c reused after all-grey -> 3 grey mistakes
_MISTAKE_REPORT = ("mistakes: yes\nposition 2, letter R, grey\n"
                   "position 3, letter A, grey\nposition 4, letter C, grey")
_CLEAN_TARGET = "crane:xxxxx;moist"     # none of m,o,i,s,t are absent -> clean


@pytest.fixture(scope="module")
def bank():
    return MistakeBank()


@pytest.fixture
def local(bank):
    return LocalMistakeClient(bank)


@pytest.fixture
def http():
    with TestClient(app) as tc:
        yield HTTPMistakeClient(client=tc)


class TestProtocol:
    def test_both_satisfy_protocol(self, local, http):
        assert isinstance(local, MistakeClient)
        assert isinstance(http, MistakeClient)


class TestParity:
    def test_mistake_parity(self, local, http):
        local.reset(word=_MISTAKE_TARGET)
        http.reset(word=_MISTAKE_TARGET)
        assert local.step(_MISTAKE_REPORT).status == http.step(_MISTAKE_REPORT).status == "correct"

    def test_clean_parity(self, local, http):
        local.reset(word=_CLEAN_TARGET)
        http.reset(word=_CLEAN_TARGET)
        assert local.step("mistakes: no").status == http.step("mistakes: no").status == "correct"


class TestBank:
    def test_has_both_classes(self, bank):
        assert bank.mistakes and bank.cleans

    def test_sample_targets_balanced(self, bank):
        import random
        from games.mistakeid.game import decode_target, true_errors

        rng = random.Random(0)
        n = min(2 * len(bank.mistakes), 200)
        targets = bank.sample_targets(n, "train", rng)
        with_err = sum(1 for t in targets if true_errors(*decode_target(t)))
        # ~50/50 by construction.
        assert abs(with_err - len(targets) / 2) <= 1


class TestServerErrors:
    def test_unknown_game_id_404(self):
        with TestClient(app) as tc:
            assert tc.get("/state/nope").status_code == 404

    def test_step_after_end_400(self):
        with TestClient(app) as tc:
            gid = tc.post("/reset", json={"word": _CLEAN_TARGET}).json()["game_id"]
            tc.post("/step", json={"game_id": gid, "answer": "mistakes: no"})
            again = tc.post("/step", json={"game_id": gid, "answer": "mistakes: no"})
            assert again.status_code == 400
