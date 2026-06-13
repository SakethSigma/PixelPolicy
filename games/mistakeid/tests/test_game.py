"""Tests for the pure Mistake-identification logic in ``games.mistakeid.game``."""

from __future__ import annotations

import pytest

from games.mistakeid.game import (
    GameOverError,
    MistakeGame,
    constraints_from_rounds,
    decode_target,
    encode_target,
    parse_report,
    score_feedback,
    true_errors,
)


class TestScoreFeedback:
    def test_greens_and_yellows(self):
        assert score_feedback("crane", "crane") == "ggggg"
        # slate vs crane: a and e land, s/l/t are absent.
        assert score_feedback("slate", "crane") == "xxgxg"

    def test_encode_decode(self):
        rounds = [("crane", "xxxxx"), ("slate", "xxgxg")]
        assert decode_target(encode_target(rounds, "moist")) == (rounds, "moist")


class TestConstraintsAndErrors:
    def test_grey_reuse_detected(self):
        rounds = [("crane", "xxxxx")]            # c r a n e all absent
        absent, yellow = constraints_from_rounds(rounds)
        assert absent == set("crane") and yellow == set()
        errs = true_errors(rounds, "track")      # r,a,c are absent -> grey errors
        assert [(e.position, e.letter, e.kind) for e in errs] == [
            (2, "R", "grey"), (3, "A", "grey"), (4, "C", "grey")]

    def test_yellow_repeat_detected(self):
        rounds = [("steam", "yxxxx")]            # s yellow at pos0; t,e,a,m absent
        errs = true_errors(rounds, "snork")      # s back at pos0 -> yellow repeat
        assert [(e.position, e.letter, e.kind) for e in errs] == [(1, "S", "yellow")]

    def test_clean_board_no_errors(self):
        rounds = [("crane", "xxxxx")]
        assert true_errors(rounds, "moist") == []


class TestParseReport:
    def test_yes_with_errors(self):
        flag, errs = parse_report("mistakes: yes\nposition 4, letter R, grey\nposition 1, letter A, yellow")
        assert flag is True
        assert errs == {(4, "R", "grey"), (1, "A", "yellow")}

    def test_no(self):
        assert parse_report("mistakes: no") == (False, set())

    def test_no_flag_is_none(self):
        assert parse_report("I am not sure") is None


class TestMistakeGame:
    def test_mistake_board_correct_report(self):
        g = MistakeGame([("crane", "xxxxx")], "track", "id1")
        report = "mistakes: yes\nposition 2, letter R, grey\nposition 3, letter A, grey\nposition 4, letter C, grey"
        state = g.step(report)
        assert state.status == "correct"
        assert state.solution.has_mistakes is True and len(state.solution.errors) == 3

    def test_mistake_board_incomplete_report_incorrect(self):
        g = MistakeGame([("crane", "xxxxx")], "track", "id2")
        assert g.step("mistakes: yes\nposition 2, letter R, grey").status == "incorrect"

    def test_clean_board_no_is_correct(self):
        g = MistakeGame([("crane", "xxxxx")], "moist", "id3")
        assert g.step("mistakes: no").status == "correct"

    def test_clean_board_false_positive_incorrect(self):
        g = MistakeGame([("crane", "xxxxx")], "moist", "id4")
        assert g.step("mistakes: yes\nposition 1, letter M, grey").status == "incorrect"

    def test_unparseable_incorrect(self):
        g = MistakeGame([("crane", "xxxxx")], "moist", "id5")
        assert g.step("dunno").status == "incorrect"

    def test_step_after_end_raises(self):
        g = MistakeGame([("crane", "xxxxx")], "moist", "id6")
        g.step("mistakes: no")
        with pytest.raises(GameOverError):
            g.step("mistakes: no")
