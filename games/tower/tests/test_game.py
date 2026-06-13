"""Tests for the pure Tower-deduction logic in ``games.tower.game``."""

from __future__ import annotations

import pytest

from games.tower.game import (
    GameOverError,
    TowerGame,
    decode_target,
    encode_target,
    parse_answer,
    solve,
)
from games.tower.render import render_solutions

NAMES = ["Alice", "Bob", "Carol"]


class TestSolve:
    def test_all_floors_correct_one_solution(self):
        sols = solve([1, 2, 3], [0, 0, 0], [True, True, True], [True, True, True])
        assert len(sols) == 1
        assert sols[0][0] == (1, 2, 3)

    def test_one_floor_correct_one_solution(self):
        sols = solve([1, 2, 3], [0, 0, 0], [True, False, False], [True, True, True])
        assert len(sols) == 1
        assert sols[0][0] == (1, 3, 2)

    def test_all_floors_wrong_two_solutions(self):
        sols = solve([1, 2, 3], [0, 0, 0], [False, False, False], [True, True, True])
        assert len(sols) == 2
        assert {s[0] for s in sols} == {(2, 3, 1), (3, 1, 2)}    # the two derangements

    def test_rooms_are_flipped_when_wrong(self):
        sols = solve([1, 2, 3], [0, 1, 0], [True, True, True], [False, True, False])
        assert sols[0][1] == (1, 1, 1)    # person 0 and 2 rooms flipped (0->1), person 1 kept


class TestEncodeDecode:
    def test_round_trip(self):
        t = encode_target(NAMES, [1, 2, 3], [0, 1, 0], [True, False, False], [False, True, True])
        assert decode_target(t) == (NAMES, [1, 2, 3], [0, 1, 0],
                                    [True, False, False], [False, True, True])


class TestParseAnswer:
    def test_round_trip_two_solutions(self):
        g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [False, False, False], [True, True, True], "id")
        canonical = render_solutions(g.solution_placements())
        parsed = parse_answer(canonical, NAMES)
        assert parsed is not None and len(parsed) == 2

    def test_no_lines_is_none(self):
        assert parse_answer("I have no idea", NAMES) is None


class TestTowerGame:
    def _correct_answer(self, g: TowerGame) -> str:
        return render_solutions(g.solution_placements())

    def test_single_solution_correct(self):
        g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [True, True, True], [True, True, True], "a")
        assert g.step(self._correct_answer(g)).status == "correct"

    def test_two_solutions_must_list_both(self):
        g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [False, False, False], [True, True, True], "b")
        full = self._correct_answer(g)
        assert g.state().status == "in_progress"
        assert g.step(full).status == "correct"

    def test_listing_only_one_of_two_is_incorrect(self):
        g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [False, False, False], [True, True, True], "c")
        only_one = "solution 1:\nAlice: floor 2, Left\nBob: floor 3, Left\nCarol: floor 1, Left"
        assert g.step(only_one).status == "incorrect"

    def test_unparseable_incorrect(self):
        g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [True, True, True], [True, True, True], "d")
        assert g.step("dunno").status == "incorrect"

    def test_step_after_end_raises(self):
        g = TowerGame(NAMES, [1, 2, 3], [0, 0, 0], [True, True, True], [True, True, True], "e")
        g.step(self._correct_answer(g))
        with pytest.raises(GameOverError):
            g.step("anything")
