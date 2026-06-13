"""Play the Tower-deduction game in the terminal.

    uv run --package game-tower python -m games.tower.play              # a sampled challenge
    uv run --package game-tower python -m games.tower.play --http URL   # via the server

You're shown a proposed placement of three people + per-person floor/room feedback, and asked to
list every placement consistent with it. Enter lines like 'Alice: floor 3, Right' (prefix blocks
with 'solution 1:' / 'solution 2:' if there are two); finish with an empty line.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.tower.client import HTTPTowerClient, LocalTowerClient, TowerClient
from games.tower.game import TowerBank
from games.tower.render import render_observation, render_solutions


def play(client: TowerClient) -> None:
    """Pose one challenge, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("\nEnter your placement(s); finish with an empty line:")
    try:
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        answer = "\n".join(lines)
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return

    state = client.step(answer)
    verdict = "✓ correct" if state.status == "correct" else "✗ incorrect"
    print(f"\n{verdict}. Consistent placement(s):")
    print(render_solutions(state.solutions))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play the Tower-deduction game.")
    parser.add_argument("--mode", default="train", help="(unused split label; kept for parity)")
    parser.add_argument("--word", default=None, help="pin an encoded 'names;shown;feedback' target")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPTowerClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return

    client = LocalTowerClient(TowerBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
