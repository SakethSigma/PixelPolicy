"""Tests for dataset construction — the critical guarantee is train/val disjointness."""

from __future__ import annotations

from games.wordle.game import WordBank
from training.grpo.data import ALL_GAMES, build_rows


def test_wordle_rows_shape_and_count():
    rows = build_rows("wordle", 200, seed=0)
    assert len(rows) == 200
    r = rows[0]
    assert r["data_source"] == "wordle"
    assert r["reward_model"]["ground_truth"] == r["extra_info"]["target"]
    assert r["prompt"][0]["role"] == "system"


def test_wordle_targets_distinct():
    rows = build_rows("wordle", 500, seed=0)
    targets = [r["extra_info"]["target"] for r in rows]
    assert len(set(targets)) == len(targets)  # sampled without replacement


def test_wordle_targets_are_train_only_disjoint_from_val():
    bank = WordBank()
    val = set(bank.val)
    train = set(bank.train)
    rows = build_rows("wordle", 1000, seed=0)
    for r in rows:
        tgt = r["extra_info"]["target"]
        assert tgt in train       # drawn from the train pool
        assert tgt not in val     # never an eval word


def test_seed_is_reproducible():
    a = [r["extra_info"]["target"] for r in build_rows("wordle", 100, seed=7)]
    b = [r["extra_info"]["target"] for r in build_rows("wordle", 100, seed=7)]
    assert a == b


def test_all_task_covers_every_game():
    rows = build_rows("all", 5, seed=0)
    seen = {r["data_source"] for r in rows}
    assert seen == set(ALL_GAMES)
