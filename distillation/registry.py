"""Per-game wiring for the batch player — the ONE place a new game is added.

The batch player (`batch_play.py`) is game-agnostic: it only ever calls an agent's
``build_messages`` / ``parse_action`` / ``system_prompt`` and an env's ``step`` / ``state``.
A :class:`GameSpec` supplies the three game-specific bits — how to make an agent, how to make
an env already reset to a target, and how to sample target instances — plus the round cap.

Add a new game = add one entry to ``GAMES``. Nothing in ``batch_play.py`` changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from agents.anagram.agent import AnagramAgent, AnagramEnv
from agents.bullscows.agent import BullsCowsAgent, BullsCowsEnv
from agents.charcount.agent import CharCountAgent, CharCountEnv
from agents.charset.agent import CharsetAgent, CharsetEnv
from agents.codebreaker.agent import CodebreakerAgent, CodebreakerEnv
from agents.consistency.agent import ConsistencyAgent, ConsistencyEnv
from agents.crossword.agent import CrosswordAgent, CrosswordEnv
from agents.endstart.agent import EndstartAgent, EndstartEnv
from agents.mistakeid.agent import MistakeAgent, MistakeEnv
from agents.rhyme.agent import RhymeAgent, RhymeEnv
from agents.tower.agent import TowerAgent, TowerEnv
from agents.validity.agent import ValidityAgent, ValidityEnv
from agents.wordle.agent import WordleAgent, WordleEnv
from games.anagram.client import LocalAnagramClient
from games.anagram.game import AnagramBank
from games.bullscows.client import LocalBullsCowsClient
from games.bullscows.game import BullsCowsBank
from games.charcount.client import LocalCharCountClient
from games.charcount.game import CharCountBank
from games.charset.client import LocalCharsetClient
from games.charset.game import CharsetBank, encode_words
from games.codebreaker.client import LocalCodebreakerClient
from games.codebreaker.game import CodebreakerBank
from games.consistency.client import LocalConsistencyClient
from games.consistency.game import ConsistencyBank
from games.crossword.client import LocalCrosswordClient
from games.crossword.game import CrosswordBank
from games.endstart.client import LocalEndstartClient
from games.endstart.game import EndstartBank, encode_target as endstart_encode
from games.mistakeid.client import LocalMistakeClient
from games.mistakeid.game import MistakeBank
from games.rhyme.client import LocalRhymeClient
from games.rhyme.game import RhymeBank
from games.tower.client import LocalTowerClient
from games.tower.game import TowerBank
from games.validity.client import LocalValidityClient
from games.validity.game import ValidityBank
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank


@dataclass(frozen=True)
class GameSpec:
    """Everything the generic batch player needs to drive one game type.

    - ``make_agent()`` -> a stateless agent (``build_messages`` / ``parse_action`` / ``system_prompt``).
    - ``make_env(target)`` -> an env already reset to ``target`` (``step(action)`` / ``state()``);
      ``state().status == "in_progress"`` means the game is still going.
    - ``sample_targets(n, mode, rng)`` -> ``n`` distinct targets drawn with the caller's seeded ``rng``.
    - ``max_rounds`` -> hard cap on rounds per game.
    - ``good_status`` -> the terminal status that counts as a *solved* episode (the rejection
      gate / the unified ``valid`` flag). Wordle's is ``"won"``; single-turn games use ``"correct"``.
    """

    make_agent: Callable[[], Any]
    make_env: Callable[[str], Any]
    sample_targets: Callable[[int, str, random.Random], list[str]]
    max_rounds: int
    good_status: str = "won"
    # For reasoning games (the prompt asks the model to think): a solved trace that contains NO
    # <think> block is unusable as an SFT target, so it is marked invalid alongside wrong traces.
    # (Anagram and crossword are distilled at high adaptive-thinking effort to elicit reasoning.)
    require_think: bool = False


def _wordle_spec() -> GameSpec:
    """Build the Wordle spec, loading the train/val word lists once (shared across the run)."""
    bank = WordBank()

    def make_env(target: str):
        env = WordleEnv(LocalWordleClient(bank))
        env.reset(word=target)  # pin the secret word; mode is irrelevant once the word is fixed
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        pool = bank.train if mode == "train" else bank.val
        if n > len(pool):
            raise ValueError(f"asked for {n} distinct {mode} targets but the pool has {len(pool)}")
        return rng.sample(pool, n)  # distinct words, deterministic for a given seed

    return GameSpec(make_agent=WordleAgent, make_env=make_env, sample_targets=sample_targets, max_rounds=6)


def _charcount_spec() -> GameSpec:
    """Build the Character-counts spec (single-turn), loading the shared vocabulary once.

    Charcount is a programmatic game (see ``distillation/programmatic.py``); this spec lets the
    generic machinery — and any future live/batch path — drive it the same way as Wordle. The
    only structural difference is ``max_rounds == 1``: one ``step`` ends the episode.
    """
    bank = CharCountBank()

    def make_env(target: str):
        env = CharCountEnv(LocalCharCountClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        pool = bank.train if mode == "train" else bank.val
        if n > len(pool):
            raise ValueError(f"asked for {n} distinct {mode} targets but the pool has {len(pool)}")
        return rng.sample(pool, n)  # distinct words, deterministic for a given seed

    return GameSpec(make_agent=CharCountAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _validity_spec() -> GameSpec:
    """Build the Validity + meaning spec (single-turn, programmatic).

    ``sample_targets`` returns a 50/50 mix of real words (valid) and synthesized pseudo-words
    (invalid); the env computes the verdict from the word itself, so the target string is all the
    batch/eval driver needs. Programmatic generation lives in ``distillation/programmatic.py``.
    """
    bank = ValidityBank()

    def make_env(target: str):
        env = ValidityEnv(LocalValidityClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        n_valid = n // 2
        pool = bank.train if mode == "train" else bank.val
        if n_valid > len(pool):
            raise ValueError(f"asked for {n_valid} distinct {mode} valid words but the pool has {len(pool)}")
        out = rng.sample(pool, n_valid)
        seen = set(out)
        while len(out) < n:
            cand = bank.make_pseudo_word(rng)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
        rng.shuffle(out)
        return out

    return GameSpec(make_agent=ValidityAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _anagram_spec() -> GameSpec:
    """Build the Anagrams spec (single-turn, Claude-distilled + sort-rejection).

    Targets are encoded ``"w1,w2"`` pairs in a 40/60 positive/negative mix; the env scores the
    yes/no verdict against the pure ``sorted()`` check, which is the rejection gate.
    """
    bank = AnagramBank()

    def make_env(target: str):
        env = AnagramEnv(LocalAnagramClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return bank.sample_targets(n, mode, rng, pos_fraction=0.4)

    return GameSpec(make_agent=AnagramAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct", require_think=True)


def _rhyme_spec() -> GameSpec:
    """Build the Rhymes spec (single-turn, programmatic). The registry path uses the free
    variant (single-word targets); the MCQ variant is built in ``distillation/programmatic.py``."""
    bank = RhymeBank()

    def make_env(target: str):
        env = RhymeEnv(LocalRhymeClient(bank))
        env.reset(word=target, variant="free")
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        pool = bank.train if mode == "train" else bank.val
        out: list[str] = []
        seen: set[str] = set()
        for _ in range(100000):
            if len(out) >= n:
                break
            w = rng.choice(pool)
            if w not in seen and bank.has_rhyme(w):
                seen.add(w)
                out.append(w)
        if len(out) < n:
            raise ValueError(f"could only find {len(out)} rhymable {mode} words (asked {n})")
        return out

    return GameSpec(make_agent=RhymeAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _crossword_spec() -> GameSpec:
    """Build the Crossword-fill spec (single-turn, Claude-distilled + reasoning).

    Targets are seed words (half Wordle vocab, half general); ``make_env`` reconstructs the clue
    (definition + masked pattern) deterministically from the word. The env scores an exact match
    to the seed word, and ``require_think`` keeps only traces that actually reasoned.
    """
    bank = CrosswordBank()

    def make_env(target: str):
        env = CrosswordEnv(LocalCrosswordClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return bank.sample_targets(n, mode, rng)

    return GameSpec(make_agent=CrosswordAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct", require_think=True)


def _charset_spec() -> GameSpec:
    """Build the Character-set spec (single-turn, programmatic).

    Targets are comma-joined word lists (one Wordle word + non-five-letter words); ``make_env``
    decodes them. Programmatic generation lives in ``distillation/programmatic.py``.
    """
    bank = CharsetBank()

    def make_env(target: str):
        env = CharsetEnv(LocalCharsetClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for _ in range(100000):
            if len(out) >= n:
                break
            enc = encode_words(bank.make_words(mode, rng))
            if enc not in seen:
                seen.add(enc)
                out.append(enc)
        if len(out) < n:
            raise ValueError(f"could only build {len(out)} distinct charset challenges (asked {n})")
        return out

    return GameSpec(make_agent=CharsetAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _mistakeid_spec() -> GameSpec:
    """Build the Mistake-identification spec (single-turn, Claude-distilled + reasoning).

    Targets are encoded ``board;attempt`` challenges drawn 50/50 mistake/clean from the committed
    real-Wordle challenge set; the env scores the reported error set against the truth computed
    from the board feedback, and ``require_think`` keeps only traces that actually reasoned.
    """
    bank = MistakeBank()

    def make_env(target: str):
        env = MistakeEnv(LocalMistakeClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return bank.sample_targets(n, mode, rng)

    return GameSpec(make_agent=MistakeAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct", require_think=True)


def _tower_spec() -> GameSpec:
    """Build the Tower-deduction spec (single-turn, programmatic).

    Targets encode ``names;shown;feedback``; ``make_env`` decodes them. The model lists every
    placement consistent with the feedback (1 or 2). Programmatic generation lives in
    ``distillation/programmatic.py``.
    """
    bank = TowerBank()

    def make_env(target: str):
        env = TowerEnv(LocalTowerClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return bank.sample_targets(n, mode, rng)

    return GameSpec(make_agent=TowerAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _endstart_spec() -> GameSpec:
    """Ends-with → starts-with (single-turn, programmatic). Targets encode word1 + 5 options."""
    bank = EndstartBank()

    def make_env(target: str):
        env = EndstartEnv(LocalEndstartClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for _ in range(n * 50):
            if len(out) >= n:
                break
            enc = endstart_encode(*bank.make_challenge(mode, rng))
            if enc not in seen:
                seen.add(enc)
                out.append(enc)
        return out

    return GameSpec(make_agent=EndstartAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _consistency_spec() -> GameSpec:
    """Candidate-consistency (single-turn, programmatic). Targets encode board + candidate."""
    bank = ConsistencyBank()

    def make_env(target: str):
        env = ConsistencyEnv(LocalConsistencyClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return bank.sample_targets(n, mode, rng)

    return GameSpec(make_agent=ConsistencyAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=1, good_status="correct")


def _codebreaker_spec() -> GameSpec:
    """Codebreaker / Mastermind (multi-turn, programmatic). Targets are secret codes."""
    bank = CodebreakerBank()

    def make_env(target: str):
        env = CodebreakerEnv(LocalCodebreakerClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return [bank.make_secret(rng) for _ in range(n)]

    return GameSpec(make_agent=CodebreakerAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=12, good_status="won")


def _bullscows_spec() -> GameSpec:
    """Bulls & Cows (multi-turn, programmatic). Targets are secret distinct-digit codes."""
    bank = BullsCowsBank()

    def make_env(target: str):
        env = BullsCowsEnv(LocalBullsCowsClient(bank))
        env.reset(word=target)
        return env

    def sample_targets(n: int, mode: str, rng: random.Random) -> list[str]:
        return [bank.make_secret(rng) for _ in range(n)]

    return GameSpec(make_agent=BullsCowsAgent, make_env=make_env, sample_targets=sample_targets,
                    max_rounds=10, good_status="won")


# name -> zero-arg factory that builds the spec (loads shared resources lazily, once per run).
GAMES: dict[str, Callable[[], GameSpec]] = {
    "wordle": _wordle_spec,
    "charcount": _charcount_spec,
    "validity": _validity_spec,
    "anagram": _anagram_spec,
    "rhyme": _rhyme_spec,
    "crossword": _crossword_spec,
    "charset": _charset_spec,
    "mistakeid": _mistakeid_spec,
    "tower": _tower_spec,
    "endstart": _endstart_spec,
    "codebreaker": _codebreaker_spec,
    "bullscows": _bullscows_spec,
    "consistency": _consistency_spec,
}

# Stable game number per game name — a unified-schema column (wordle is 0; the six word-skill
# games are 1-6 per games/DATA_SOURCING.md). Used by the SFT writers and push normalization.
GAME_NUMBERS: dict[str, int] = {
    "wordle": 0,
    "charcount": 1,
    "validity": 2,
    "anagram": 3,
    "rhyme": 5,
    "crossword": 6,
    "charset": 7,
    "mistakeid": 8,
    "tower": 9,
    "endstart": 4,
    "codebreaker": 10,
    "bullscows": 11,
    "consistency": 12,
}
