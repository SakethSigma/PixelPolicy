"""Play Validity + meaning in the terminal.

    uv run --package game-validity python -m games.validity.play              # a real word
    uv run --package game-validity python -m games.validity.play --kind invalid  # a pseudo-word
    uv run --package game-validity python -m games.validity.play --word kindle
    uv run --package game-validity python -m games.validity.play --http URL   # via the server

You're shown a word and asked whether it is a real English word; if valid, also give a meaning.
Type ``valid: <meaning>`` or ``invalid``. The env scores it and reveals the truth.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.validity.client import HTTPValidityClient, LocalValidityClient, ValidityClient
from games.validity.game import ValidityBank
from games.validity.render import render_answer


def play(client: ValidityClient) -> None:
    """Pose one challenge, read an answer, score it, reveal the truth."""
    state = client.state()
    print(f"\nWord: {state.word}")
    print("Is this a real English word? Type 'valid: <meaning>' or 'invalid'; q to quit.")
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
    truth = render_answer(state.solution.valid, state.solution.meaning)
    print(f"\n{verdict}. The truth for '{state.word}':\n{truth}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Validity + meaning in the terminal.")
    parser.add_argument("--mode", choices=["train", "val"], default="train",
                        help="which word pool to draw a valid challenge from")
    parser.add_argument("--kind", choices=["valid", "invalid"], default="valid",
                        help="ask the bank for a real word or a pseudo-word")
    parser.add_argument("--word", default=None, help="pin the challenge word (real or pseudo)")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPValidityClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word, kind=args.kind)
            play(http_client)
        return

    client = LocalValidityClient(ValidityBank())
    client.reset(mode=args.mode, word=args.word, kind=args.kind)
    play(client)


if __name__ == "__main__":
    main()
