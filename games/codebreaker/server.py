"""FastAPI server wrapping the pure Codebreaker environment.

    uv run --package game-codebreaker uvicorn games.codebreaker.server:app --reload

Multi-turn API: ``/reset`` then repeated ``/guess``. Pin a secret with ``{"word": "ACEF"}``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from games.codebreaker.game import CodebreakerBank, CodebreakerGame, GameOverError, GameState

app = FastAPI(title="Codebreaker", description="A pure Mastermind-style code-breaking environment.")

bank = CodebreakerBank()
games: dict[str, CodebreakerGame] = {}


class ResetRequest(BaseModel):
    mode: str = "train"
    word: Optional[str] = None  # pin the secret code (else sampled)


class GuessRequest(BaseModel):
    game_id: str
    guess: str


@app.post("/reset", response_model=GameState)
async def reset(req: ResetRequest) -> GameState:
    secret = req.word.strip().upper() if req.word is not None else bank.sample(req.mode)
    game_id = str(uuid.uuid4())
    games[game_id] = CodebreakerGame(secret=secret, game_id=game_id)
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
