"""FastAPI server wrapping the pure Wordle environment.

Run single-process (no ``--workers``) so the in-memory game store is shared:

    uv run --package game-wordle uvicorn games.wordle.server:app --reload

Endpoints are ``async`` — concurrency comes from the asyncio event loop, not from
worker processes. Every handler is trivial CPU work (feedback compute is
microseconds and never awaits mid-session), so the loop is never bottlenecked and
no locking is needed.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.wordle.game import (
    WORD_LENGTH,
    GameOverError,
    GameState,
    Mode,
    WordBank,
    WordleGame,
)

app = FastAPI(title="Wordle", description="A pure Wordle environment over HTTP.")

# Loaded once at startup: word list + deterministic train/val split.
word_bank = WordBank()

# In-memory game store, keyed by game_id. Single-process only (see module docstring).
games: dict[str, WordleGame] = {}


class ResetRequest(BaseModel):
    mode: Mode = "train"
    word: Optional[str] = None  # pin the target (testing/debugging only)


class GuessRequest(BaseModel):
    game_id: str
    guess: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    """Start a new game and return its (empty) state."""
    if req.word is not None:
        target = req.word.strip().lower()
        if len(target) != WORD_LENGTH or not target.isalpha():
            raise HTTPException(
                status_code=400,
                detail=f"Pinned word must be {WORD_LENGTH} alphabetic letters",
            )
    else:
        target = word_bank.sample(req.mode)

    game_id = str(uuid.uuid4())
    games[game_id] = WordleGame(
        target=target, game_id=game_id, validate_word=word_bank.is_valid
    )
    return games[game_id].state()


@app.post("/guess", response_model=GameState)
async def guess(req: GuessRequest) -> GameState:
    """Submit a guess; returns the full game state (all rounds + feedback).

    The game core owns validation: a wrong-length / non-word guess is not a 400 —
    it consumes a round and the latest ``RoundResult`` carries an ``error`` reason.
    """
    game = games.get(req.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {req.game_id}")

    try:
        game.guess(req.guess)
    except GameOverError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return game.state()


@app.get("/state/{game_id}", response_model=GameState)
async def state(game_id: str) -> GameState:
    """Observe the current game state without acting."""
    game = games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {game_id}")
    return game.state()
