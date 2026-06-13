"""Play Candidate-consistency in the terminal.

    uv run --package game-consistency python -m games.consistency.play
    uv run --package game-consistency python -m games.consistency.play --http URL
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.consistency.client import (
    ConsistencyClient,
    HTTPConsistencyClient,
    LocalConsistencyClient,
)
from games.consistency.game import ConsistencyBank
from games.consistency.render import render_observation


def play(client: ConsistencyClient) -> None:
    state = client.state()
    print(f"\n{render_observation(state)}")
    try:
        answer = input("Answer (yes/no): ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return
    if answer.strip().lower() in {"q", "quit", "exit"}:
        print("Bye.")
        return
    state = client.step(answer)
    verdict = "✓ correct" if state.status == "correct" else "✗ incorrect"
    poss = "still possible" if state.solution else "ruled out"
    print(f"\n{verdict}. '{state.candidate}' is {poss}.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Candidate-consistency.")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--word", default=None, help="pin an encoded 'rows;candidate' target")
    parser.add_argument("--http", metavar="URL", default=None)
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPConsistencyClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return
    client = LocalConsistencyClient(ConsistencyBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
