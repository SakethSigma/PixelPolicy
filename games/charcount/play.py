"""Play Character-counts in the terminal.

    uv run --package game-charcount python -m games.charcount.play            # in-process
    uv run --package game-charcount python -m games.charcount.play --http URL # via the server

You're shown one word and asked for its analysis (length, vowels, consonants); type it as the
canonical block (``length: N`` / ``vowels: a, e`` / ``consonants: p, l, n, t``) — order and
spacing are forgiving. The env scores it correct/incorrect and reveals the answer. Single-turn,
so one prompt ends the game.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.charcount.client import (
    CharCountClient,
    HTTPCharCountClient,
    LocalCharCountClient,
)
from games.charcount.game import CharCountBank
from games.charcount.render import render_answer, render_observation


def play(client: CharCountClient) -> None:
    """Pose one challenge, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("Enter the analysis (e.g. 'length: 6, vowels: a e, consonants: p l n t'); q to quit.")
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
    print(f"\n{verdict}. The analysis of '{state.word}' is:")
    print(render_answer(state.solution))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Character-counts in the terminal.")
    parser.add_argument("--mode", choices=["train", "val"], default="train",
                        help="which word pool to draw the challenge from")
    parser.add_argument("--word", default=None, help="pin the challenge word")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPCharCountClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return

    client = LocalCharCountClient(CharCountBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
