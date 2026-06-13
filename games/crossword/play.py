"""Play Crossword-fill in the terminal.

    uv run --package game-crossword python -m games.crossword.play              # a sampled clue
    uv run --package game-crossword python -m games.crossword.play --word crane
    uv run --package game-crossword python -m games.crossword.play --http URL   # via the server

You're shown a definition, the word length, and a partially-revealed pattern, and asked for the
word. The env scores it (exact match to the seed word) and reveals the answer. Single-turn.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.crossword.client import CrosswordClient, HTTPCrosswordClient, LocalCrosswordClient
from games.crossword.game import CrosswordBank
from games.crossword.render import render_observation


def play(client: CrosswordClient) -> None:
    """Pose one clue, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("Enter the word; q to quit.")
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
    print(f"\n{verdict}. The word was '{state.solution.word}'.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Crossword-fill in the terminal.")
    parser.add_argument("--mode", default="train", help="(unused split label; kept for parity)")
    parser.add_argument("--word", default=None, help="pin the seed word")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPCrosswordClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return

    client = LocalCrosswordClient(CrosswordBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
