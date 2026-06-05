"""Tests for the Wordle HTTP API in ``games.wordle.server``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from games.wordle.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def new_game(client, **body) -> str:
    """Reset a game and return its game_id."""
    resp = client.post("/reset", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["game_id"]


class TestReset:
    def test_default_mode_returns_empty_game(self, client):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "in_progress"
        assert data["current_round"] == 0
        assert data["rounds"] == []
        assert data["target"] is None
        assert data["max_rounds"] == 6
        assert data["game_id"]

    def test_distinct_game_ids(self, client):
        assert new_game(client) != new_game(client)

    def test_val_mode_allowed(self, client):
        assert client.post("/reset", json={"mode": "val"}).status_code == 200

    def test_invalid_mode_rejected(self, client):
        # Literal["train", "val"] → 422 from pydantic validation.
        assert client.post("/reset", json={"mode": "bogus"}).status_code == 422

    def test_pinned_word_accepted(self, client):
        assert client.post("/reset", json={"word": "apple"}).status_code == 200

    def test_pinned_word_wrong_length_rejected(self, client):
        assert client.post("/reset", json={"word": "toolong"}).status_code == 400

    def test_pinned_word_non_alpha_rejected(self, client):
        assert client.post("/reset", json={"word": "12345"}).status_code == 400


class TestGuess:
    def test_feedback_shape_and_target_hidden(self, client):
        gid = new_game(client, word="apple")
        resp = client.post("/guess", json={"game_id": gid, "guess": "puppy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_round"] == 1
        assert data["rounds"][0]["guess"] == "PUPPY"
        assert data["rounds"][0]["feedback"] == ["-", "x", "✓", "x", "x"]
        assert data["status"] == "in_progress"
        assert data["target"] is None

    def test_history_accumulates(self, client):
        gid = new_game(client, word="apple")
        client.post("/guess", json={"game_id": gid, "guess": "crane"})
        data = client.post("/guess", json={"game_id": gid, "guess": "moist"}).json()
        assert [r["guess"] for r in data["rounds"]] == ["CRANE", "MOIST"]

    def test_winning_guess_reveals_target(self, client):
        gid = new_game(client, word="apple")
        data = client.post("/guess", json={"game_id": gid, "guess": "apple"}).json()
        assert data["status"] == "won"
        assert data["target"] == "APPLE"

    def test_case_insensitive_guess(self, client):
        gid = new_game(client, word="apple")
        data = client.post("/guess", json={"game_id": gid, "guess": "APPLE"}).json()
        assert data["status"] == "won"

    def test_loss_after_six_rounds_reveals_target(self, client):
        gid = new_game(client, word="apple")
        data = None
        for _ in range(6):
            data = client.post("/guess", json={"game_id": gid, "guess": "crane"}).json()
        assert data["status"] == "lost"
        assert data["current_round"] == 6
        assert data["target"] == "APPLE"

    def test_guess_after_game_over_rejected(self, client):
        gid = new_game(client, word="apple")
        client.post("/guess", json={"game_id": gid, "guess": "apple"})  # win
        resp = client.post("/guess", json={"game_id": gid, "guess": "crane"})
        assert resp.status_code == 400

    def test_wrong_length_guess_consumes_round(self, client):
        gid = new_game(client, word="apple")
        resp = client.post("/guess", json={"game_id": gid, "guess": "ab"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_round"] == 1
        assert data["rounds"][0]["error"] == "inadequate length"
        assert data["rounds"][0]["feedback"] == []

    def test_non_alpha_guess_consumes_round(self, client):
        gid = new_game(client, word="apple")
        resp = client.post("/guess", json={"game_id": gid, "guess": "12345"})
        assert resp.status_code == 200
        assert resp.json()["rounds"][0]["error"] == "inadequate length"

    def test_non_word_guess_consumes_round(self, client):
        gid = new_game(client, word="apple")
        resp = client.post("/guess", json={"game_id": gid, "guess": "zzzzz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["rounds"][0]["error"] == "out of vocabulary"
        assert data["rounds"][0]["feedback"] == []  # no free letter-probing

    def test_six_invalid_guesses_lose_and_reveal(self, client):
        gid = new_game(client, word="apple")
        data = None
        for _ in range(6):
            data = client.post("/guess", json={"game_id": gid, "guess": "zzzzz"}).json()
        assert data["status"] == "lost"
        assert data["current_round"] == 6
        assert data["target"] == "APPLE"

    def test_invalid_then_valid_history(self, client):
        gid = new_game(client, word="apple")
        client.post("/guess", json={"game_id": gid, "guess": "zzzzz"})
        data = client.post("/guess", json={"game_id": gid, "guess": "crane"}).json()
        assert data["rounds"][0]["error"] == "out of vocabulary"
        assert data["rounds"][1]["error"] is None
        assert data["rounds"][1]["guess"] == "CRANE"

    def test_unknown_game_id_404(self, client):
        resp = client.post("/guess", json={"game_id": "does-not-exist", "guess": "apple"})
        assert resp.status_code == 404


class TestState:
    def test_returns_current_state(self, client):
        gid = new_game(client, word="apple")
        client.post("/guess", json={"game_id": gid, "guess": "crane"})
        data = client.get(f"/state/{gid}").json()
        assert data["current_round"] == 1
        assert data["rounds"][0]["guess"] == "CRANE"
        assert data["target"] is None

    def test_does_not_mutate_game(self, client):
        gid = new_game(client, word="apple")
        client.get(f"/state/{gid}")
        client.get(f"/state/{gid}")
        assert client.get(f"/state/{gid}").json()["current_round"] == 0

    def test_unknown_game_id_404(self, client):
        assert client.get("/state/does-not-exist").status_code == 404


class TestFullPlaythrough:
    def test_win_on_second_guess(self, client):
        gid = new_game(client, word="apple")
        first = client.post("/guess", json={"game_id": gid, "guess": "crane"}).json()
        assert first["status"] == "in_progress"
        assert "feedback" not in first  # feedback lives under each round, not top-level
        second = client.post("/guess", json={"game_id": gid, "guess": "apple"}).json()
        assert second["status"] == "won"
        assert len(second["rounds"]) == 2
        assert second["target"] == "APPLE"
