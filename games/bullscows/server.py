"""FastAPI server wrapping the pure Bulls & Cows environment.

    uv run --package game-bullscows uvicorn games.bullscows.server:app --reload

Multi-turn API: ``/reset`` then repeated ``/guess``. Pin a secret with ``{"word": "1234"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.bullscows.game import BullsCowsBank, BullsCowsGame, GameOverError, GameState

app = FastAPI(title="BullsCows", description="A pure Bulls & Cows deduction environment over HTTP.")

bank = BullsCowsBank()
games: dict[str, BullsCowsGame] = {}


class ResetRequest(BaseModel):
    mode: str = "train"
    word: Optional[str] = None  # pin the secret (else sampled)


class GuessRequest(BaseModel):
    game_id: str
    guess: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    secret = req.word.strip() if req.word is not None else bank.sample(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = BullsCowsGame(secret=secret, game_id=game_id)
    return games[game_id].state()


@app.post("/guess", response_model=GameState)
async def guess(req: GuessRequest) -> GameState:
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
    game = games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {game_id}")
    return game.state()
