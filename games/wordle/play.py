"""Play Wordle in the terminal.

    uv run --package game-wordle python -m games.wordle.play            # in-process
    uv run --package game-wordle python -m games.wordle.play --http URL # via the server

Defaults to the in-process :class:`LocalWordleClient`, so you can just run it — no
server required. ``--http URL`` drives the same game through the HTTP server instead,
which is a handy by-hand proof that both transports behave identically.

Colored tiles use ``rich`` (install the extra: ``pip install game-wordle[tui]``). The
board layout mirrors :func:`games.wordle.render.render_observation` — the model and a
human see the same thing, only the human's is in color.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from games.wordle.game import GameState, LetterFeedback, WordBank
from games.wordle.client import HTTPWordleClient, LocalWordleClient, WordleClient

try:
    from rich.console import Console
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - exercised only without the extra
    print(
        "This terminal UI needs 'rich'. Install it with: pip install game-wordle[tui]",
        file=sys.stderr,
    )
    raise SystemExit(1)

_TILE_STYLE = {
    LetterFeedback.CORRECT: "bold white on green",
    LetterFeedback.WRONG_POS: "bold black on yellow",
    LetterFeedback.WRONG_LETTER: "bold white on grey37",
}
_MAX_GUESS_DISPLAY = 20  # clamp an over-long invalid guess so it can't break the layout


def render_board(state: GameState, console: Console) -> None:
    """Draw every round as colored tiles (invalid rounds shown in red, no tiles)."""
    for rnd in state.rounds:
        if rnd.error is not None:
            shown = rnd.guess[:_MAX_GUESS_DISPLAY]
            console.print(
                Text(f"  {shown}", style="bold red").append(
                    f"   invalid: {rnd.error.value} — counted as a round",
                    style="red",
                )
            )
            continue
        line = Text("  ")
        for letter, fb in zip(rnd.guess, rnd.feedback):
            line.append(f" {letter} ", style=_TILE_STYLE[fb])
            line.append(" ")
        console.print(line)


def step(client: WordleClient, raw: str) -> GameState:
    """Submit one raw guess and return the resulting state (validation is the env's)."""
    return client.guess(raw)


def play(client: WordleClient, console: Optional[Console] = None) -> GameState:
    """Run an interactive game to completion (or until the player quits)."""
    console = console or Console()
    state = client.state()
    console.print()
    render_board(state, console)

    while state.status == "in_progress":
        left = state.max_rounds - state.current_round
        try:
            raw = console.input(f"[bold]Guess[/] ([dim]{left} left, q to quit[/]): ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            return state
        if raw.strip().lower() in {"q", "quit", "exit"}:
            console.print("[dim]Bye.[/]")
            return state

        state = step(client, raw)
        console.print()
        render_board(state, console)

    if state.status == "won":
        console.print(f"\n[bold green]You won in {state.current_round} guesses![/]")
    else:
        console.print(f"\n[bold red]You lost.[/] The word was [bold]{state.target}[/].")
    return state


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play Wordle in the terminal.")
    parser.add_argument("--mode", choices=["train", "val"], default="train",
                        help="which word pool to draw the target from")
    parser.add_argument("--word", default=None,
                        help="pin the target word (debugging / fixed puzzles)")
    parser.add_argument("--http", metavar="URL", default=None,
                        help="play via the HTTP server at URL instead of in-process")
    args = parser.parse_args(argv)

    console = Console()
    client: WordleClient
    if args.http is not None:
        with HTTPWordleClient(args.http) as http_client:
            http_client.reset(mode=args.mode, word=args.word)
            play(http_client, console)
        return

    bank = WordBank()
    if args.word is not None and not bank.is_valid(args.word):
        console.print(
            f"[yellow]Warning:[/] '{args.word}' is not an allowed word — "
            "it can never be guessed, so this game is a guaranteed loss."
        )
    client = LocalWordleClient(bank)
    client.reset(mode=args.mode, word=args.word)
    play(client, console)


if __name__ == "__main__":
    main()
