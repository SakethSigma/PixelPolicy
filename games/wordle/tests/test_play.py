"""Tests for the terminal UI in ``games.wordle.play``.

The interactive ``input()`` loop is driven by a fake console so no TTY is needed.
"""

from __future__ import annotations

import io

import pytest

from games.wordle import play as play_mod
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank
from games.wordle.play import main, play, render_board, step


@pytest.fixture(scope="module")
def bank():
    return WordBank()


class FakeConsole:
    """Stands in for rich.Console: scripted input, captured output."""

    def __init__(self, inputs):
        self._inputs = iter(inputs)
        self.printed: list[str] = []

    def input(self, prompt=""):
        try:
            return next(self._inputs)
        except StopIteration:  # player walked away → behave like EOF
            raise EOFError

    def print(self, *args, **kwargs):
        self.printed.append(" ".join(str(a) for a in args))


def local(bank, word="apple"):
    c = LocalWordleClient(bank)
    c.reset(word=word)
    return c


class TestStep:
    def test_invalid_guess_yields_error_round(self, bank):
        st = step(local(bank), "zzzzz")
        assert st.rounds[0].error is not None
        assert st.current_round == 1


class TestRenderBoard:
    def test_renders_without_error(self, bank):
        from rich.console import Console

        c = local(bank)
        c.guess("crane")
        c.guess("zzzzz")  # invalid round — must render too
        console = Console(file=io.StringIO())
        render_board(c.state(), console)  # should not raise


class TestPlay:
    def test_terminates_on_win(self, bank):
        console = FakeConsole(["crane", "apple"])
        final = play(local(bank), console)
        assert final.status == "won"
        assert any("won" in p.lower() for p in console.printed)

    def test_quit_command_exits_in_progress(self, bank):
        console = FakeConsole(["q"])
        final = play(local(bank), console)
        assert final.status == "in_progress"

    def test_eof_exits_cleanly(self, bank):
        console = FakeConsole([])  # immediate EOF
        final = play(local(bank), console)
        assert final.status == "in_progress"


class TestMain:
    def test_main_terminates_on_win(self, monkeypatch):
        console = FakeConsole(["apple"])
        monkeypatch.setattr(play_mod, "Console", lambda *a, **k: console)
        main(["--word", "apple"])
        assert any("won" in p.lower() for p in console.printed)
