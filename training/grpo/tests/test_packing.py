"""Tests for the pure (no-torch) per-turn sample packing logic."""

from __future__ import annotations

from training.grpo.rollout import TurnSample
from training.grpo.sample_packing import pack_rows, pad_left, pad_right


def test_pad_left_prompt():
    ids, mask = pad_left([5, 6, 7], 5, pad_id=0)
    assert ids == [0, 0, 5, 6, 7]
    assert mask == [0, 0, 1, 1, 1]


def test_pad_left_truncates_left_overflow():
    ids, mask = pad_left([1, 2, 3, 4, 5], 3, pad_id=0)
    assert ids == [3, 4, 5] and mask == [1, 1, 1]   # keeps the most recent context


def test_pad_right_response_mask_is_loss_mask():
    ids, mask = pad_right([9, 8], 4, pad_id=0)
    assert ids == [9, 8, 0, 0]
    assert mask == [1, 1, 0, 0]                       # loss only on real response tokens


def test_pad_right_truncates_tail():
    ids, mask = pad_right([1, 2, 3, 4, 5], 3, pad_id=0)
    assert ids == [1, 2, 3] and mask == [1, 1, 1]


def _s(prompt, resp, uid, reward, rnd):
    return TurnSample(prompt_ids=prompt, response_ids=resp, uid=uid, target="vivid",
                      game="wordle", round=rnd, reward=reward)


def test_pack_rows_shapes_and_fields():
    samples = [
        _s([1, 2], [3, 4, 5], "vivid#r1", 0.35, 1),
        _s([1, 2, 6, 7], [8], "vivid#r2", 0.05, 2),
    ]
    rows = pack_rows(samples, pad_id=0, max_prompt_len=4, max_response_len=3)
    assert all(len(p) == 4 for p in rows["prompt_ids"])
    assert all(len(r) == 3 for r in rows["response_ids"])
    assert rows["uid"] == ["vivid#r1", "vivid#r2"]
    assert rows["reward"] == [0.35, 0.05]
    assert rows["round"] == [1, 2]
    # prompt is left-padded, response right-padded
    assert rows["prompt_ids"][0] == [0, 0, 1, 2]
    assert rows["response_mask"][1] == [1, 0, 0]
