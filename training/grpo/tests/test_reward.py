"""Unit tests for the per-turn local Wordle reward (no verl, no model — pure functions)."""

from __future__ import annotations

from training.grpo.reward import (
    EpisodeOutcome,
    RewardWeights,
    compute_reward_sparse,
    is_format_valid,
    per_turn_rewards,
    reward_breakdown,
    win_discount,
)

W = RewardWeights()                      # defaults: win_bonus=0.0, novelty_decay=0.90
WIN = RewardWeights(win_bonus=2.0)       # terminal enabled (the safety valve)


def _turn(guess: str, feedback: list[str], *, error: str | None = None,
          format_valid: bool = True) -> dict:
    return {"guess": guess, "feedback": feedback, "error": error, "format_valid": format_valid}


def _episode(turns: list[dict], *, status: str = "in_progress", rounds_used: int | None = None,
             target: str = "vivid", max_rounds: int = 6) -> EpisodeOutcome:
    return {
        "target": target,
        "status": status,
        "rounds_used": rounds_used if rounds_used is not None else len(turns),
        "max_rounds": max_rounds,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Format gate
# ---------------------------------------------------------------------------
def test_format_valid_accepts_think_and_5char_guess():
    assert is_format_valid("<think>reasoning</think>\n<guess>crane</guess>")


def test_format_invalid_when_guess_not_5_chars():
    assert not is_format_valid("<think>x</think><guess>cat</guess>")
    assert not is_format_valid("<think>x</think><guess>cranes</guess>")


def test_format_invalid_without_think_when_required():
    assert not is_format_valid("<guess>crane</guess>", require_think=True)
    assert is_format_valid("<guess>crane</guess>", require_think=False)


# ---------------------------------------------------------------------------
# Per-turn shape: one reward per turn, purely local
# ---------------------------------------------------------------------------
def test_one_reward_per_turn():
    turns = [_turn("crane", ["x"] * 5), _turn("vivid", ["✓"] * 5)]
    R = per_turn_rewards(_episode(turns, status="won", rounds_used=2), W)
    assert len(R) == len(turns)


def test_turn_reward_is_local_only():
    # A format error in the LAST turn must not change the FIRST turn's reward.
    good_last = [_turn("vapor", ["✓", "x", "x", "x", "x"]), _turn("vivid", ["✓"] * 5)]
    bad_last = [_turn("vapor", ["✓", "x", "x", "x", "x"]),
                _turn("vivid", ["✓"] * 5, format_valid=False)]
    R_good = per_turn_rewards(_episode(good_last), W)
    R_bad = per_turn_rewards(_episode(bad_last), W)
    assert R_good[0] == R_bad[0]                  # round 1 untouched by round 2's error
    assert R_bad[1] < R_good[1]                   # only round 2 pays for its own bad format


# ---------------------------------------------------------------------------
# Information-gain: novel discovery pays once; re-confirmation free
# ---------------------------------------------------------------------------
def test_novel_green_pays_once_regreen_is_free():
    turns = [
        _turn("vapor", ["✓", "x", "x", "x", "x"]),   # new green at pos 0
        _turn("vexes", ["✓", "x", "x", "x", "x"]),   # SAME green at pos 0 -> free (only format)
    ]
    R = per_turn_rewards(_episode(turns), W)
    assert R[0] == W.format_ok + W.green_new                     # round 1: format + new green
    assert R[1] == W.format_ok                                   # round 2: format only (re-green free)


def test_yellow_pays_once_then_free():
    turns = [
        _turn("apple", ["-", "x", "x", "x", "x"]),   # new yellow 'a'
        _turn("table", ["x", "-", "x", "x", "x"]),   # 'a' yellow again -> free
    ]
    R = per_turn_rewards(_episode(turns), W)
    assert R[0] == W.format_ok + W.yellow_new
    assert R[1] == W.format_ok


def test_green_outweighs_yellow():
    assert W.green_new > W.yellow_new


# ---------------------------------------------------------------------------
# Round decay: same discovery earlier is worth more
# ---------------------------------------------------------------------------
def test_round_decay_makes_early_discovery_worth_more():
    early = per_turn_rewards(_episode([_turn("vivid", ["✓", "x", "x", "x", "x"])]), W)[0]
    # same single new green, but at round 3 (two no-op-but-valid turns before it can't re-discover,
    # so use distinct positions to keep them "novel")
    turns = [
        _turn("snail", ["x", "x", "x", "x", "x"]),   # round 1, no discovery
        _turn("third", ["x", "x", "x", "x", "x"]),   # round 2, no discovery
        _turn("vivid", ["✓", "x", "x", "x", "x"]),   # round 3: new green at pos 0
    ]
    late = per_turn_rewards(_episode(turns), W)[2]
    # late green is decayed by novelty_decay**2; subtract the flat format term to compare novelty
    assert (early - W.format_ok) > (late - W.format_ok)


# ---------------------------------------------------------------------------
# Invalid guesses
# ---------------------------------------------------------------------------
def test_invalid_guess_penalized_no_feedback_credit():
    R = per_turn_rewards(_episode([_turn("zzzzz", [], error="out of vocabulary")]), W)
    assert R[0] == W.format_ok + W.invalid_guess


# ---------------------------------------------------------------------------
# Terminal: OFF by default, on the winning turn only when enabled
# ---------------------------------------------------------------------------
def test_no_terminal_by_default():
    won = _episode([_turn("crane", ["x"] * 5), _turn("vivid", ["✓"] * 5)],
                   status="won", rounds_used=2)
    R = per_turn_rewards(won, W)               # default win_bonus = 0
    # last turn reward == its own local reward (format + 5 new greens), no terminal added
    assert R[-1] == W.format_ok + 5 * W.green_new * (W.novelty_decay ** 1)


def test_terminal_on_winning_turn_only_when_enabled():
    won = _episode([_turn("crane", ["x"] * 5), _turn("vivid", ["✓"] * 5)],
                   status="won", rounds_used=2)
    R0 = per_turn_rewards(won, W)
    R1 = per_turn_rewards(won, WIN)
    assert R1[0] == R0[0]                       # non-winning turn unchanged
    assert R1[-1] > R0[-1]                      # only the winning (last) turn gets the bonus
    assert R1[-1] - R0[-1] == WIN.win_bonus * win_discount(2, 6, WIN)


def test_terminal_absent_on_loss():
    lost = _episode([_turn("crane", ["x"] * 5)] * 6, status="lost", rounds_used=6)
    assert per_turn_rewards(lost, WIN) == per_turn_rewards(lost, W)  # no win term either way


def test_win_discount_monotonic():
    assert win_discount(1, 6, WIN) > win_discount(3, 6, WIN) > win_discount(6, 6, WIN)
    assert win_discount(1, 6, WIN) == 1.0


# ---------------------------------------------------------------------------
# Breakdown (wandb) consistency
# ---------------------------------------------------------------------------
def test_breakdown_sums_match_total():
    turns = [_turn("crane", ["x", "-", "x", "x", "x"]), _turn("vivid", ["✓"] * 5)]
    ep = _episode(turns, status="won", rounds_used=2)
    R = per_turn_rewards(ep, WIN)
    b = reward_breakdown(ep, WIN)
    assert abs(sum(R) - (b["novel"] + b["format"] + b["invalid"] + b["win"])) < 1e-9


# ---------------------------------------------------------------------------
# Sparse fallback (Arm C games)
# ---------------------------------------------------------------------------
def test_sparse_reward():
    o_ok = _episode([_turn("x", [])], status="correct")
    o_no = _episode([_turn("x", [])], status="lost")
    assert compute_reward_sparse(o_ok, good_status="correct") == 1.0
    assert compute_reward_sparse(o_no, good_status="correct") == 0.0
