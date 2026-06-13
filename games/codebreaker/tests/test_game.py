"""Tests for the pure Codebreaker logic + the unbiased solver."""

from __future__ import annotations

import random
import re

import pytest

from games.codebreaker.game import (
    CODE_LENGTH,
    SYMBOLS,
    CodebreakerGame,
    CodebreakerSolver,
    GameOverError,
    compute_feedback,
    consistent_codes,
    is_valid_guess,
)


def _code(move: str) -> str:
    return re.search(r"<guess>(.*?)</guess>", move).group(1)


class TestFeedback:
    def test_all_correct(self):
        assert compute_feedback("ACEF", "ACEF") == "✓✓✓✓"

    def test_duplicate_handling(self):
        # guess AABB vs secret ABCD: A✓, second A has no copy left (x), one B present (-), other B x.
        assert compute_feedback("AABB", "ABCD") == "✓x-x"

    def test_wrong_position(self):
        assert compute_feedback("FECA", "ACEF") == "----"   # all present, all misplaced


class TestGame:
    def test_win(self):
        g = CodebreakerGame("ACEF", "t")
        st = g.guess("ACEF")
        assert g.status == "won" and st.feedback == "✓✓✓✓"

    def test_invalid_guess_consumes_round_no_feedback(self):
        g = CodebreakerGame("ACEF", "t")
        st = g.guess("ZZZZ")          # Z not in A-F
        assert st.feedback == "" and st.error == "invalid" and g.current_round == 1

    def test_lost_after_max_rounds(self):
        g = CodebreakerGame("ACEF", "t", max_rounds=2)
        g.guess("AAAA")
        g.guess("BBBB")
        assert g.status == "lost"

    def test_step_after_end_raises(self):
        g = CodebreakerGame("ACEF", "t")
        g.guess("ACEF")
        with pytest.raises(GameOverError):
            g.guess("BBBB")


class TestSolver:
    def test_solver_always_solves(self):
        for seed in range(20):
            secret = "".join(random.Random(seed).choice(SYMBOLS) for _ in range(CODE_LENGTH))
            g = CodebreakerGame(secret, "t")
            solver = CodebreakerSolver(random.Random(seed + 100))
            while g.status == "in_progress":
                g.guess(_code(solver.move(g.state())))
            assert g.status == "won", f"unsolved secret {secret}"

    def test_guesses_are_consistent_with_prior_feedback(self):
        secret = "ACEF"
        g = CodebreakerGame(secret, "t")
        solver = CodebreakerSolver(random.Random(3))
        codes = []
        while g.status == "in_progress":
            code = _code(solver.move(g.state()))
            prior = [(r.guess, r.feedback) for r in g.state().rounds if r.feedback]
            if prior:                       # every non-opening guess must be a live candidate
                assert code in consistent_codes(prior)
            codes.append(code)
            g.guess(code)

    def test_opening_is_unbiased(self):
        # Different seeds open with different codes — no fixed/ordered opening.
        from games.codebreaker.game import GameState
        empty = GameState(game_id="t")
        openings = {_code(CodebreakerSolver(random.Random(s)).move(empty)) for s in range(25)}
        assert len(openings) > 1
        assert all(is_valid_guess(o) for o in openings)
