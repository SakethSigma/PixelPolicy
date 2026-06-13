"""Play Codebreaker in the terminal (multi-turn).

    uv run --package game-codebreaker python -m games.codebreaker.play
    uv run --package game-codebreaker python -m games.codebreaker.play --http URL
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.codebreaker.client import (
    CodebreakerClient,
    HTTPCodebreakerClient,
    LocalCodebreakerClient,
)
from games.codebreaker.game import CodebreakerBank
from games.codebreaker.render import render_observation, render_round


def play(client: CodebreakerClient) -> None:
    state = client.state()
    print(f"\n{render_observation(state)}")
    while state.status == "in_progress":
        try:
            g = input("Guess: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if g.strip().lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return
        state = client.guess(g)
        if state.rounds:
            print("  " + render_round(state.rounds[-1]))
    if state.status == "won":
        print(f"\n✓ Cracked it in {state.current_round} guesses!")
    else:
        print(f"\n✗ Out of rounds. The code was {state.secret}.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Codebreaker.")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--word", default=None, help="pin the secret code, e.g. ACEF")
    parser.add_argument("--http", metavar="URL", default=None)
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPCodebreakerClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return
    client = LocalCodebreakerClient(CodebreakerBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
