"""Play the Mistake-identification game in the terminal.

    uv run --package game-mistakeid python -m games.mistakeid.play            # a sampled board
    uv run --package game-mistakeid python -m games.mistakeid.play --http URL # via the server

You're shown a Wordle board and a proposed next guess, and asked whether it repeats a known
grey/yellow mistake. Type 'mistakes: no' or 'mistakes: yes' followed by lines like
'position 4, letter R, grey'. The env scores it and reveals the truth.
"""

from __future__ import annotations

import argparse
from typing import Optional

from games.mistakeid.client import HTTPMistakeClient, LocalMistakeClient, MistakeClient
from games.mistakeid.game import MistakeBank
from games.mistakeid.render import render_observation


def play(client: MistakeClient) -> None:
    """Pose one challenge, read a report, score it, reveal the truth."""
    state = client.state()
    print(f"\n{render_observation(state)}")
    print("\nReport mistakes ('mistakes: no', or 'mistakes: yes' + 'position N, letter X, grey|yellow'); q to quit.")
    try:
        lines = []
        print("Enter your report; finish with an empty line:")
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        answer = "\n".join(lines)
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return
    if answer.strip().lower() in {"q", "quit", "exit"}:
        print("Bye.")
        return

    state = client.step(answer)
    verdict = "✓ correct" if state.status == "correct" else "✗ incorrect"
    sol = state.solution
    if sol.has_mistakes:
        truth = "; ".join(f"pos {e.position} {e.letter} {e.kind}" for e in sol.errors)
        print(f"\n{verdict}. Real mistakes: {truth}")
    else:
        print(f"\n{verdict}. The proposed guess repeats no known mistakes.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play the Mistake-identification game.")
    parser.add_argument("--mode", default="train", help="(unused split label; kept for parity)")
    parser.add_argument("--word", default=None, help="pin an encoded 'board;attempt' target")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    if args.http is not None:
        with HTTPMistakeClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client)
        return

    client = LocalMistakeClient(MistakeBank())
    client.reset(mode=args.mode, word=args.word)
    play(client)


if __name__ == "__main__":
    main()
