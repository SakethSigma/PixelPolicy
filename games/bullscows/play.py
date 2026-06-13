"""Play Bulls & Cows in the terminal (multi-turn).

    uv run --package game-bullscows python -m games.bullscows.play
    uv run --package game-bullscows python -m games.bullscows.play --http URL
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.bullscows.client import BullsCowsClient, HTTPBullsCowsClient, LocalBullsCowsClient
from games.bullscows.game import BullsCowsBank
from games.bullscows.render import render_observation, render_round


def play(client: BullsCowsClient) -> None:
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
        print(f"\n✓ Got it in {state.current_round} guesses!")
    else:
        print(f"\n✗ Out of rounds. The number was {state.secret}.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Bulls & Cows.")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--word", default=None, help="pin the secret, e.g. 1234")
    parser.add_argument("--http", metavar="URL", default=None)
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPBullsCowsClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return
    client = LocalBullsCowsClient(BullsCowsBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
