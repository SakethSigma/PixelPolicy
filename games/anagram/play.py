"""Play Anagrams in the terminal.

    uv run --package game-anagram python -m games.anagram.play                    # a sampled pair
    uv run --package game-anagram python -m games.anagram.play --pair listen,silent
    uv run --package game-anagram python -m games.anagram.play --http URL         # via the server

You're shown two words and asked whether they are anagrams. Type yes/no. The env scores it and
reveals the truth. Single-turn, so one prompt ends the game.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.anagram.client import AnagramClient, HTTPAnagramClient, LocalAnagramClient
from games.anagram.game import AnagramBank
from games.anagram.render import render_observation


def play(client: AnagramClient) -> None:
    """Pose one challenge, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("Answer yes or no; q to quit.")
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
    truth = "ARE" if state.solution.are_anagrams else "are NOT"
    print(f"\n{verdict}. '{state.word1}' and '{state.word2}' {truth} anagrams.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Anagrams in the terminal.")
    parser.add_argument("--mode", choices=["train", "val"], default="train",
                        help="which word pool to draw the pair from")
    parser.add_argument("--pair", default=None, help="pin the pair, e.g. 'listen,silent'")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPAnagramClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.pair)
            play(http_client)
        return

    client = LocalAnagramClient(AnagramBank())
    client.reset(mode=args.mode, word=args.pair)
    play(client)


if __name__ == "__main__":
    main()
