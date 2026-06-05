"""Tests for the pure Wordle game logic in ``games.wordle.game``."""

from __future__ import annotations

import pytest

from games.wordle.game import (
    DEFAULT_MAX_ROUNDS,
    WORD_LENGTH,
    GameOverError,
    GameState,
    InvalidReason,
    LetterFeedback,
    RoundResult,
    WordBank,
    WordleGame,
    _load_words,
    _WORDS_FILE,
    assign_pool,
    compute_feedback,
    generate_split_files,
)

C, P, X = LetterFeedback.CORRECT, LetterFeedback.WRONG_POS, LetterFeedback.WRONG_LETTER


def fb(guess: str, target: str) -> str:
    """Feedback as a compact string, e.g. ``"-x✓xx"``."""
    return "".join(f.value for f in compute_feedback(guess, target))


class TestLetterFeedback:
    def test_enum_values(self):
        assert LetterFeedback.CORRECT.value == "✓"
        assert LetterFeedback.WRONG_POS.value == "-"
        assert LetterFeedback.WRONG_LETTER.value == "x"


class TestComputeFeedback:
    def test_all_correct(self):
        assert compute_feedback("apple", "apple") == [C, C, C, C, C]

    def test_all_wrong(self):
        assert compute_feedback("crane", "moist") == [X, X, X, X, X]

    def test_puppy_apple_canonical_duplicate(self):
        # The example from the design doc: one P consumed by the green.
        assert fb("puppy", "apple") == "-x✓xx"

    def test_case_insensitive(self):
        assert compute_feedback("PuPpY", "ApPlE") == compute_feedback("puppy", "apple")

    def test_green_takes_precedence_over_yellow_for_duplicates(self):
        # target GEESE has three E; two are matched green, leaving one for a yellow.
        assert fb("eerie", "geese") == "-✓xx✓"

    def test_no_yellow_once_all_copies_matched_green(self):
        # target EAGLE has two E, both matched green → the middle E is gray, not yellow.
        assert fb("eerie", "eagle") == "✓xxx✓"

    def test_mixed_yellows_and_grays(self):
        # target ALERT, guess CRANE: C/N absent; R, A, E present but misplaced.
        assert fb("crane", "alert") == "x--x-"

    def test_length_preserved(self):
        assert len(compute_feedback("crane", "moist")) == WORD_LENGTH


class TestRoundResult:
    def test_model_roundtrip(self):
        r = RoundResult(guess="CRANE", feedback=[X, C, P, X, X])
        assert r.guess == "CRANE"
        assert [f.value for f in r.feedback] == ["x", "✓", "-", "x", "x"]


class TestWordleGame:
    def test_initial_state(self):
        g = WordleGame(target="apple", game_id="g1")
        assert g.status == "in_progress"
        assert g.current_round == 0
        state = g.state()
        assert state.game_id == "g1"
        assert state.rounds == []
        assert state.max_rounds == DEFAULT_MAX_ROUNDS
        assert state.target is None  # hidden until the game ends

    def test_target_stored_uppercase(self):
        assert WordleGame(target="apple", game_id="g1").target == "APPLE"

    def test_guess_appends_and_uppercases(self):
        g = WordleGame(target="apple", game_id="g1")
        result = g.guess("crane")
        assert isinstance(result, RoundResult)
        assert result.guess == "CRANE"
        assert g.current_round == 1
        assert len(g.state().rounds) == 1

    def test_win_sets_status_and_reveals_target(self):
        g = WordleGame(target="apple", game_id="g1")
        g.guess("apple")
        assert g.status == "won"
        assert g.state().target == "APPLE"

    def test_target_hidden_while_in_progress(self):
        g = WordleGame(target="apple", game_id="g1")
        g.guess("crane")
        assert g.status == "in_progress"
        assert g.state().target is None

    def test_lose_after_max_rounds(self):
        g = WordleGame(target="apple", game_id="g1", max_rounds=3)
        for _ in range(3):
            g.guess("crane")
        assert g.status == "lost"
        assert g.current_round == 3
        assert g.state().target == "APPLE"  # revealed on loss

    def test_win_on_final_round_is_not_a_loss(self):
        g = WordleGame(target="apple", game_id="g1", max_rounds=2)
        g.guess("crane")
        g.guess("apple")
        assert g.status == "won"

    def test_guess_after_game_over_raises(self):
        g = WordleGame(target="apple", game_id="g1")
        g.guess("apple")
        with pytest.raises(GameOverError):
            g.guess("crane")

    def test_guess_after_loss_raises(self):
        g = WordleGame(target="apple", game_id="g1", max_rounds=1)
        g.guess("crane")
        assert g.status == "lost"
        with pytest.raises(GameOverError):
            g.guess("crane")

    def test_custom_max_rounds_respected(self):
        g = WordleGame(target="apple", game_id="g1", max_rounds=10)
        assert g.state().max_rounds == 10


class TestInvalidGuess:
    """An invalid guess consumes a round, carries a reason, and gives no feedback."""

    def vocab_game(self, **kw):
        # Validator that only accepts the words used in these tests.
        allowed = {"APPLE", "CRANE", "MOIST"}
        return WordleGame(
            target="apple", game_id="g1", validate_word=lambda w: w in allowed, **kw
        )

    def test_inadequate_length_consumes_round_no_feedback(self):
        g = self.vocab_game()
        r = g.guess("ab")
        assert r.error is InvalidReason.LENGTH
        assert r.feedback == []
        assert g.current_round == 1
        assert g.status == "in_progress"

    def test_non_alpha_consumes_round(self):
        g = self.vocab_game()
        assert g.guess("12345").error is InvalidReason.LENGTH

    def test_out_of_vocab_consumes_round_no_feedback(self):
        g = self.vocab_game()
        r = g.guess("zzzzz")
        assert r.error is InvalidReason.VOCAB
        assert r.feedback == []          # no free letter-probing
        assert g.current_round == 1

    def test_invalid_round_is_not_a_win(self):
        # Guard against all([]) == True registering an empty-feedback round as a win.
        g = self.vocab_game()
        g.guess("zzzzz")
        assert g.status == "in_progress"

    def test_invalid_guess_on_final_round_loses_and_reveals(self):
        g = self.vocab_game(max_rounds=1)
        g.guess("zzzzz")
        assert g.status == "lost"
        assert g.state().target == "APPLE"

    def test_validator_skipped_when_none(self):
        # Bare game (no validator): a 5-alpha non-word is scored normally, not rejected.
        g = WordleGame(target="apple", game_id="g1")
        r = g.guess("zzzzz")
        assert r.error is None
        assert len(r.feedback) == WORD_LENGTH

    def test_validator_receives_normalized_word(self):
        seen = []
        g = WordleGame(
            target="apple", game_id="g1", validate_word=lambda w: seen.append(w) or True
        )
        g.guess("  CrAnE ")
        assert seen == ["CRANE"]

    def test_game_over_precedes_validation(self):
        g = self.vocab_game()
        g.guess("apple")  # win
        with pytest.raises(GameOverError):
            g.guess("zzzzz")  # invalid, but game-over check fires first


class TestAssignPool:
    def test_returns_train_or_val(self):
        assert assign_pool("apple") in ("train", "val")

    def test_deterministic_per_word(self):
        # Same word, many calls → same pool, always.
        assert all(assign_pool("apple") == assign_pool("apple") for _ in range(100))

    def test_case_and_whitespace_normalised(self):
        assert assign_pool("  APPLE ") == assign_pool("apple")

    def test_assignment_is_order_independent(self):
        # The whole point: a word's pool does not depend on any other word.
        words = ["apple", "crane", "moist", "eerie", "eagle", "geese", "alert"]
        first = {w: assign_pool(w) for w in words}
        second = {w: assign_pool(w) for w in reversed(words)}
        assert first == second

    def test_val_fraction_bounds(self):
        # Over the real vocabulary the empirical val share is close to the target.
        words = _load_words(_WORDS_FILE)
        share = sum(assign_pool(w) == "val" for w in words) / len(words)
        assert 0.17 < share < 0.23


class TestGenerateSplitFiles:
    def test_writes_disjoint_complete_split(self, tmp_path):
        words_file = tmp_path / "words.txt"
        words_file.write_text("# header\n" + "\n".join(
            ["apple", "crane", "moist", "eerie", "eagle", "geese", "alert", "speed"]
        ) + "\n")
        train_file = tmp_path / "train.txt"
        val_file = tmp_path / "val.txt"

        n_train, n_val = generate_split_files(words_file, train_file, val_file)

        train = set(_load_words(train_file))
        val = set(_load_words(val_file))
        assert n_train == len(train)
        assert n_val == len(val)
        assert train.isdisjoint(val)
        assert train | val == set(_load_words(words_file))

    def test_regeneration_is_byte_identical(self, tmp_path):
        words_file = tmp_path / "words.txt"
        words_file.write_text("\n".join(["apple", "crane", "moist", "eerie"]) + "\n")
        train_file = tmp_path / "train.txt"
        val_file = tmp_path / "val.txt"

        generate_split_files(words_file, train_file, val_file)
        first = (train_file.read_bytes(), val_file.read_bytes())
        generate_split_files(words_file, train_file, val_file)
        second = (train_file.read_bytes(), val_file.read_bytes())
        assert first == second


class TestLoadWords:
    def test_skips_header_blanks_and_junk(self, tmp_path):
        f = tmp_path / "w.txt"
        f.write_text("# comment\napple\nCRANE\n\nbad\ntoolong\n12345\nmoist\n")
        words = _load_words(f)
        assert words == ["apple", "crane", "moist"]  # lowercased; len!=5 / non-alpha dropped


class TestWordBank:
    @pytest.fixture(scope="class")
    def bank(self):
        return WordBank()

    def test_pools_disjoint_and_union_is_all(self, bank):
        assert set(bank.train).isdisjoint(bank.val)
        assert set(bank.train) | set(bank.val) == bank.all

    def test_matches_committed_split(self, bank):
        # WordBank reflects the on-disk artifacts produced by the generator.
        assert len(bank.train) == 10384
        assert len(bank.val) == 2588

    def test_deterministic_across_instances(self):
        a, b = WordBank(), WordBank()
        assert a.train == b.train
        assert a.val == b.val

    def test_is_valid(self, bank):
        assert bank.is_valid("apple")
        assert bank.is_valid("APPLE")       # case-insensitive
        assert bank.is_valid("  crane  ")   # whitespace-tolerant
        assert not bank.is_valid("zzzzz")
        assert not bank.is_valid("toolong")

    def test_sample_draws_from_correct_pool(self, bank):
        train_set, val_set = set(bank.train), set(bank.val)
        for _ in range(50):
            assert bank.sample("train") in train_set
            assert bank.sample("val") in val_set

    def test_missing_files_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            WordBank(train_file=tmp_path / "nope.txt", val_file=tmp_path / "nope2.txt")

    def test_empty_pool_raises(self, tmp_path):
        train_file = tmp_path / "train.txt"
        val_file = tmp_path / "val.txt"
        train_file.write_text("apple\n")
        val_file.write_text("# only a comment\n")
        with pytest.raises(ValueError):
            WordBank(train_file=train_file, val_file=val_file)


class TestGameStateModel:
    def test_defaults(self):
        s = GameState(game_id="g1")
        assert s.current_round == 0
        assert s.rounds == []
        assert s.status == "in_progress"
        assert s.target is None
        assert s.max_rounds == DEFAULT_MAX_ROUNDS
