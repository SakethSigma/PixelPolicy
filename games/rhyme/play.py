"""Play Rhymes in the terminal.

    uv run --package game-rhyme python -m games.rhyme.play                  # free variant, in-process
    uv run --package game-rhyme python -m games.rhyme.play --variant mcq    # multiple-choice
    uv run --package game-rhyme python -m games.rhyme.play --http URL       # via the server

You're shown a word (and, for MCQ, five options) and asked for a rhyme. The env scores it
correct/incorrect and reveals the answer. Single-turn, so one prompt ends the game.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.rhyme.client import HTTPRhymeClient, LocalRhymeClient, RhymeClient
from games.rhyme.game import RhymeBank
from games.rhyme.render import render_observation


def play(client: RhymeClient) -> None:
    """Pose one challenge, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("Type a rhyming word; q to quit.")
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
    if state.variant == "mcq":
        reveal = f"the rhyming option was '{state.solution.correct_option}'"
    else:
        reveal = f"some words that rhyme: {', '.join(state.solution.examples) or '(none known)'}"
    print(f"\n{verdict}. For '{state.word}', {reveal}.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Rhymes in the terminal.")
    parser.add_argument("--mode", choices=["train", "val"], default="train",
                        help="which word pool to draw the challenge from")
    parser.add_argument("--variant", choices=["free", "mcq"], default="free")
    parser.add_argument("--word", default=None, help="pin the challenge word")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPRhymeClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word, variant=args.variant)
            play(http_client)
        return

    client = LocalRhymeClient(RhymeBank())
    client.reset(mode=args.mode, word=args.word, variant=args.variant)
    play(client)


if __name__ == "__main__":
    main()
