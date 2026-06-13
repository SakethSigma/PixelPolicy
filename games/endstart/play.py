"""Play Ends-with → starts-with in the terminal.

    uv run --package game-endstart python -m games.endstart.play
    uv run --package game-endstart python -m games.endstart.play --http URL

Pick the option whose first letter matches word1's last letter.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.endstart.client import EndstartClient, HTTPEndstartClient, LocalEndstartClient
from games.endstart.game import EndstartBank
from games.endstart.render import render_observation


def play(client: EndstartClient) -> None:
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("Type the matching option; q to quit.")
    try:
        answer = input("Answer: ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return
    if answer.strip().lower() in {"q", "quit", "exit"}:
        print("Bye.")
        return
    state = client.step(answer)
    verdict = "✓ correct" if state.status == "correct" else "✗ incorrect"
    print(f"\n{verdict}. The match was '{state.solution}'.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Ends-with → starts-with.")
    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--word", default=None, help="pin an encoded 'word1;opt1,...' target")
    parser.add_argument("--http", metavar="URL", default=None)
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPEndstartClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return
    client = LocalEndstartClient(EndstartBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
