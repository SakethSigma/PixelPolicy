"""Play the Character-set game in the terminal.

    uv run --package game-charset python -m games.charset.play                  # a sampled challenge
    uv run --package game-charset python -m games.charset.play --words cat,planet
    uv run --package game-charset python -m games.charset.play --http URL       # via the server

You're shown a few words and asked which letters of a-z are used (across all the words) and which
are unused. Type ``used: a c e ... / unused: b d ...``. The env scores it and reveals the truth.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.charset.client import CharsetClient, HTTPCharsetClient, LocalCharsetClient
from games.charset.game import CharsetBank
from games.charset.render import render_answer, render_observation


def play(client: CharsetClient) -> None:
    """Pose one challenge, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("List the used and unused letters (e.g. 'used: a c t / unused: b d ...'); q to quit.")
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
    print(f"\n{verdict}. For {', '.join(state.words)}:")
    print(render_answer(state.solution.used, state.solution.unused))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play the Character-set game in the terminal.")
    parser.add_argument("--mode", choices=["train", "val"], default="train",
                        help="which word pool to draw the challenge from")
    parser.add_argument("--words", default=None, help="pin the words, e.g. 'cat,planet'")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPCharsetClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.words)
            play(http_client)
        return

    client = LocalCharsetClient(CharsetBank())
    client.reset(mode=args.mode, word=args.words)
    play(client)


if __name__ == "__main__":
    main()
