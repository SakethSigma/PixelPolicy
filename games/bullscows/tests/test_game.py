"""Tests for the pure Bulls & Cows logic + the unbiased solver."""

from __future__ import annotations

import random
import re

import pytest

from games.bullscows.game import (
    DIGITS,
    N_DIGITS,
    BullsCowsGame,
    BullsCowsSolver,
    GameOverError,
    compute_feedback,
    consistent_codes,
    is_valid_guess,
)


def _code(move: str) -> str:
    return re.search(r"<guess>(.*?)</guess>", move).group(1)


class TestFeedback:
    def test_all_bulls(self):
        assert compute_feedback("1234", "1234") == (4, 0)

    def test_bulls_and_cows(self):
        assert compute_feedback("1234", "1243") == (2, 2)   # 1,2 in place; 3,4 swapped

    def test_none(self):
        assert compute_feedback("1234", "5678") == (0, 0)


class TestGame:
    def test_win(self):
        g = BullsCowsGame("1234", "t")
        g.guess("1234")
        assert g.status == "won"

    def test_invalid_repeated_digit(self):
        assert not is_valid_guess("1123")
        g = BullsCowsGame("1234", "t")
        st = g.guess("1123")
        assert st.error == "invalid" and g.current_round == 1

    def test_lost(self):
        g = BullsCowsGame("1234", "t", max_rounds=1)
        g.guess("5678")
        assert g.status == "lost"

    def test_step_after_end_raises(self):
        g = BullsCowsGame("1234", "t")
        g.guess("1234")
        with pytest.raises(GameOverError):
            g.guess("5678")


class TestSolver:
    def test_solver_always_solves(self):
        for seed in range(20):
            secret = "".join(random.Random(seed).sample(DIGITS, N_DIGITS))
            g = BullsCowsGame(secret, "t")
            solver = BullsCowsSolver(random.Random(seed + 100))
            while g.status == "in_progress":
                g.guess(_code(solver.move(g.state())))
            assert g.status == "won", f"unsolved secret {secret}"

    def test_guesses_are_consistent(self):
        g = BullsCowsGame("4271", "t")
        solver = BullsCowsSolver(random.Random(5))
        while g.status == "in_progress":
            code = _code(solver.move(g.state()))
            prior = [(r.guess, r.bulls, r.cows) for r in g.state().rounds if r.error is None]
            if prior:
                assert code in consistent_codes(prior)
            g.guess(code)

    def test_opening_is_unbiased(self):
        from games.bullscows.game import GameState
        empty = GameState(game_id="t")
        openings = {_code(BullsCowsSolver(random.Random(s)).move(empty)) for s in range(25)}
        assert len(openings) > 1
        assert all(is_valid_guess(o) for o in openings)
