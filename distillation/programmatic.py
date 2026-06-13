"""Programmatic SFT generation — the "synthetic teacher" for no-reasoning word games.

For games whose label is cheap and exact (char counts, validity, ends→starts, rhymes), no
Claude is needed: step the env, read the gold answer the core computed, and format it straight
into the completion. The rows are byte-identical in shape to the distilled ones (the unified
schema in ``distillation/schema.py``), so the combine + ``push.py`` step is unchanged.

This module implements game #1, **charcount**. It does NOT touch the generic Claude pipeline.

    uv run --package distillation python -m distillation.programmatic   # default 14k rows

Determinism: the sample is drawn with a seeded RNG; the labels are pure functions of the word,
so a given seed reproduces the exact dataset. Every row is **self-checked** (its own answer is
fed back through ``env.step`` and must score ``"correct"``) before being written — rejection by
construction, the same quality gate the distilled games pass by filtering.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from agents.base import Completion
from agents.bullscows.agent import BullsCowsAgent, BullsCowsEnv
from agents.charcount.agent import CharCountAgent
from agents.charset.agent import CharsetAgent
from agents.codebreaker.agent import CodebreakerAgent, CodebreakerEnv
from agents.consistency.agent import ConsistencyAgent
from agents.endstart.agent import EndstartAgent
from agents.rhyme.agent import RhymeAgent
from agents.tower.agent import TowerAgent
from agents.validity.agent import ValidityAgent
from agents.rollout import run_episode
from games.bullscows.client import LocalBullsCowsClient
from games.bullscows.game import BullsCowsBank, BullsCowsSolver
from games.charcount.game import CharCountBank, analyze
from games.charcount.render import render_answer
from games.charset.game import CharsetBank, CharsetGame, encode_words
from games.charset.game import analyze as charset_analyze
from games.charset.render import render_answer as render_charset_answer
from games.codebreaker.client import LocalCodebreakerClient
from games.codebreaker.game import CodebreakerBank, CodebreakerSolver
from games.consistency.game import ConsistencyBank, ConsistencyGame, decode_target as consistency_decode, is_consistent
from games.consistency.render import render_reasoning as render_consistency_reasoning
from games.endstart.game import EndstartBank, EndstartGame, correct_option, encode_target as endstart_encode
from games.rhyme.game import RhymeBank, RhymeGame, is_rhyme
from games.tower.game import TowerBank, TowerGame, decode_target as tower_decode
from games.tower.render import render_solutions as render_tower_solutions
from games.validity.game import ValidityBank, ValidityGame, lookup_meaning
from games.validity.render import render_answer as render_validity_answer
from games.wordle.game import WordBank
from distillation.registry import GAME_NUMBERS
from distillation.schema import sft_row

MIN_LEN, MAX_LEN = 3, 20


def _stratified_words(
    bank: CharCountBank, *, n: int, wordle_min: int, rng: random.Random
) -> list[str]:
    """Draw ``n`` distinct charcount-**train** words: ≥``wordle_min`` from the Wordle vocab,
    the rest from WordNet-origin words spread across lengths 3-20 (so length variety is real).

    Generating only from the charcount-train pool keeps charcount-val clean for eval. Wordle's
    own train/val words both land in charcount-train ~80% of the time (the salted split), which
    is exactly the cross-game design — the model learns to analyze words it also plays Wordle on.
    """
    wordle_vocab = WordBank().all  # all 12,972 Wordle words (every one is length 5)
    train = set(bank.train)
    wordle_train = sorted(train & wordle_vocab)
    wn_by_len: dict[int, list[str]] = defaultdict(list)
    for w in sorted(train - wordle_vocab):
        if MIN_LEN <= len(w) <= MAX_LEN:
            wn_by_len[len(w)].append(w)

    n_wordle = min(wordle_min, len(wordle_train), n)
    picked_wordle = rng.sample(wordle_train, n_wordle)

    # Fill the remainder from WordNet words, round-robin across lengths for an even spread.
    n_wn = n - n_wordle
    lengths = sorted(wn_by_len)
    for L in lengths:
        rng.shuffle(wn_by_len[L])
    picked_wn: list[str] = []
    cursors = {L: 0 for L in lengths}
    while len(picked_wn) < n_wn and any(cursors[L] < len(wn_by_len[L]) for L in lengths):
        for L in lengths:
            if cursors[L] < len(wn_by_len[L]):
                picked_wn.append(wn_by_len[L][cursors[L]])
                cursors[L] += 1
                if len(picked_wn) >= n_wn:
                    break

    words = picked_wordle + picked_wn
    rng.shuffle(words)
    return words


def generate(n: int, wordle_min: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` charcount SFT rows in the unified schema and write them to ``out_path``."""
    bank = CharCountBank()
    agent = CharCountAgent()
    rng = random.Random(seed)
    words = _stratified_words(bank, n=n, wordle_min=wordle_min, rng=rng)

    wordle_vocab = WordBank().all
    game_no = GAME_NUMBERS["charcount"]
    rows: list[dict] = []
    for i, word in enumerate(words):
        state = bank_state(word)
        messages = agent.build_messages(state)
        analysis = analyze(word)
        completion = f"<answer>\n{render_answer(analysis)}\n</answer>"

        # Self-check (rejection by construction): the answer we wrote must score "correct".
        if agent.parse_action(completion) != render_answer(analysis):
            raise AssertionError(f"answer round-trip failed for {word!r}")
        assert state.status == "in_progress"

        rows.append(sft_row(
            game_name="charcount", game_no=game_no, round=1, target=word,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=i,
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_wordle = sum(1 for w in words if w in wordle_vocab)
    hist = Counter(len(w) for w in words)
    print(f"wrote {len(rows)} charcount rows -> {out_path}")
    print(f"  wordle-vocab words: {n_wordle}   wordnet-only words: {len(words) - n_wordle}")
    print("  length spread:", {L: hist[L] for L in sorted(hist)})
    return out_path


def bank_state(word: str):
    """A fresh in-progress GameState for ``word`` (challenge posed, nothing answered yet)."""
    from games.charcount.game import CharCountGame

    return CharCountGame(word=word, game_id=f"gen-{word}").state()


def _write_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_validity(n: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` validity+meaning SFT rows (≈50/50 valid/invalid) in the unified schema.

    Valid challenges are real Wordle words that carry a WordNet definition (the gold
    ``<meaning>``); invalid challenges are synthesized pseudo-words confirmed absent from both
    WordNet and the Wordle vocab. Both pools draw from the full Wordle vocabulary — Wordle val
    words are deliberately included, since meaning recall is a different skill from Wordle play.
    Every row is self-checked through ``ValidityGame.step`` (rejection by construction).
    """
    bank = ValidityBank()
    agent = ValidityAgent()
    rng = random.Random(seed)
    game_no = GAME_NUMBERS["validity"]

    # Balanced 50/50 by construction: capped at the number of Wordle words that carry a WordNet
    # definition (≈6.6k), so the valid and invalid classes are always equal in size.
    n_valid = min(n // 2, len(bank.valid_words))
    valid_words = rng.sample(bank.valid_words, n_valid)
    n_invalid = n_valid
    invalids: list[str] = []
    seen: set[str] = set()
    while len(invalids) < n_invalid:
        cand = bank.make_pseudo_word(rng)
        if cand not in seen:
            seen.add(cand)
            invalids.append(cand)

    items = [(w, True) for w in valid_words] + [(w, False) for w in invalids]
    rng.shuffle(items)

    rows: list[dict] = []
    for i, (word, valid) in enumerate(items):
        meaning = lookup_meaning(word) if valid else None
        completion = render_validity_answer(valid, meaning)
        messages = agent.build_messages(ValidityGame(word=word, game_id=f"gen-{word}").state())
        check = ValidityGame(word=word, game_id=f"chk-{word}").step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"validity self-check failed for {word!r} (valid={valid})")
        rows.append(sft_row(
            game_name="validity", game_no=game_no, round=1, target=word,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=i,
        ))

    _write_rows(rows, out_path)
    print(f"wrote {len(rows)} validity rows -> {out_path}")
    print(f"  valid: {n_valid}   invalid (pseudo-words): {n_invalid}")
    return out_path


def generate_rhyme(n: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` rhyme SFT rows split evenly between MCQ and free-generation.

    Seeds are distinct rhymable words from the ``rhyme`` train pool. MCQ rows pick the single
    rhyming option among five; free rows name one valid rhyme. Both are self-checked through
    ``RhymeGame.step``. Needs the bundled CMU dict (``pronouncing``).
    """
    bank = RhymeBank()
    agent = RhymeAgent()
    rng = random.Random(seed)
    game_no = GAME_NUMBERS["rhyme"]

    n_mcq = n // 2
    n_free = n - n_mcq
    needed = n_mcq + n_free
    pool = list(bank.train)
    rng.shuffle(pool)
    seeds: list[str] = []
    for w in pool:
        if bank.has_rhyme(w):
            seeds.append(w)
            if len(seeds) >= needed:
                break
    if len(seeds) < needed:
        print(f"  WARNING: only {len(seeds)} rhymable train words found (asked {needed}); capping.")
        n_mcq = min(n_mcq, len(seeds) // 2)
        n_free = len(seeds) - n_mcq
    mcq_seeds = seeds[:n_mcq]
    free_seeds = seeds[n_mcq:n_mcq + n_free]

    rows: list[dict] = []
    idx = 0
    for word in mcq_seeds:
        options = bank.mcq_options(word, rng)
        if options is None:
            continue
        correct = next(o for o in options if is_rhyme(word, o))
        completion = f"<answer>{correct}</answer>"
        state = RhymeGame(word=word, game_id=f"gen-{word}", variant="mcq", options=options).state()
        messages = agent.build_messages(state)
        check = RhymeGame(word=word, game_id=f"chk-{word}", variant="mcq",
                          options=options).step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"rhyme MCQ self-check failed for {word!r}")
        rows.append(sft_row(
            game_name="rhyme", game_no=game_no, round=1, target=word,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=idx,
        ))
        idx += 1
    for word in free_seeds:
        rhyme = bank.a_rhyme(word, rng)
        if rhyme is None:
            continue
        completion = f"<answer>{rhyme}</answer>"
        state = RhymeGame(word=word, game_id=f"gen-{word}", variant="free").state()
        messages = agent.build_messages(state)
        check = RhymeGame(word=word, game_id=f"chk-{word}",
                          variant="free").step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"rhyme free self-check failed for {word!r}")
        rows.append(sft_row(
            game_name="rhyme", game_no=game_no, round=1, target=word,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=idx,
        ))
        idx += 1

    _write_rows(rows, out_path)
    n_mcq_rows = sum(1 for r in rows if r["messages"][-1]["content"].lower().startswith("which"))
    print(f"wrote {len(rows)} rhyme rows -> {out_path}")
    print(f"  mcq: {n_mcq_rows}   free: {len(rows) - n_mcq_rows}")
    return out_path


def generate_charset(n: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` used/unused character-set SFT rows in the unified schema.

    Each challenge is a small word list (one Wordle word + non-five-letter words); the gold answer
    is the union of letters used and its complement. Every row is self-checked through
    ``CharsetGame.step`` (rejection by construction). Distinct challenges only.
    """
    bank = CharsetBank()
    agent = CharsetAgent()
    rng = random.Random(seed)
    game_no = GAME_NUMBERS["charset"]

    rows: list[dict] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < n and guard < n * 50:
        guard += 1
        words = bank.make_words("train", rng)
        enc = encode_words(words)
        if enc in seen:
            continue
        seen.add(enc)
        used, unused = charset_analyze(words)
        completion = f"<answer>\n{render_charset_answer(used, unused)}\n</answer>"
        messages = agent.build_messages(CharsetGame(words=words, game_id=f"gen-{len(rows)}").state())
        check = CharsetGame(words=words, game_id=f"chk-{len(rows)}").step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"charset self-check failed for {words!r}")
        rows.append(sft_row(
            game_name="charset", game_no=game_no, round=1, target=enc,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=len(rows),
        ))

    _write_rows(rows, out_path)
    from collections import Counter as _C
    nwords = _C(len(decode) for decode in (r["target"].split(",") for r in rows))
    print(f"wrote {len(rows)} charset rows -> {out_path}")
    print("  words-per-challenge spread:", {k: nwords[k] for k in sorted(nwords)})
    return out_path


def generate_tower(n: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` tower-deduction SFT rows in the unified schema.

    Each challenge is a proposed placement + per-person feedback (with random names for surface
    variety); the gold completion lists every consistent placement (1 or 2). Self-checked through
    ``TowerGame.step``. The logic space is only 1,920 distinct structures, so beyond that the
    variety comes from the names.
    """
    bank = TowerBank()
    agent = TowerAgent()
    rng = random.Random(seed)
    game_no = GAME_NUMBERS["tower"]
    targets = bank.sample_targets(n, "train", rng)

    rows: list[dict] = []
    multi = 0
    for i, target in enumerate(targets):
        names, sf, sr, fok, rok = tower_decode(target)
        game = TowerGame(names, sf, sr, fok, rok, game_id=f"gen-{i}")
        placements = game.solution_placements()
        completion = f"<answer>\n{render_tower_solutions(placements)}\n</answer>"
        messages = agent.build_messages(game.state())
        check = TowerGame(names, sf, sr, fok, rok, game_id=f"chk-{i}").step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"tower self-check failed for {target!r}")
        if len(placements) == 2:
            multi += 1
        rows.append(sft_row(
            game_name="tower", game_no=game_no, round=1, target=target,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=i,
        ))

    _write_rows(rows, out_path)
    print(f"wrote {len(rows)} tower rows -> {out_path}")
    print(f"  single-solution: {len(rows) - multi}   two-solution: {multi}")
    return out_path


def _ok_budget(messages: list[dict], completion: str, max_chars: int = 16000) -> bool:
    """Cheap guard: keep prompt+completion under ~4k tokens (≈4 chars/token)."""
    return sum(len(m["content"]) for m in messages) + len(completion) <= max_chars


def _play_multiturn(agent, env, solver):
    """Drive one multi-turn episode with a deterministic solver as the 'teacher'.

    Reuses ``agents.rollout.run_episode``: the injected ``generate`` returns the solver's move as
    a ``Completion`` (the solver reads ``env.state()`` — the feedback so far — to pick its guess).
    """
    def generate(prompts):
        return [Completion(text=solver.move(env.state()))]

    return run_episode(agent, env, generate)


def generate_endstart(n: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` endstart MCQ rows (single-turn): pick the candidate starting with word1's
    last letter. Self-checked through ``EndstartGame.step``."""
    bank = EndstartBank()
    agent = EndstartAgent()
    rng = random.Random(seed)
    game_no = GAME_NUMBERS["endstart"]

    rows: list[dict] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < n and guard < n * 50:
        guard += 1
        word1, options = bank.make_challenge("train", rng)
        enc = endstart_encode(word1, options)
        if enc in seen:
            continue
        seen.add(enc)
        completion = f"<answer>{correct_option(word1, options)}</answer>"
        messages = agent.build_messages(EndstartGame(word1, options, f"gen-{len(rows)}").state())
        check = EndstartGame(word1, options, "chk").step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"endstart self-check failed for {enc!r}")
        rows.append(sft_row(
            game_name="endstart", game_no=game_no, round=1, target=enc,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=len(rows),
        ))

    _write_rows(rows, out_path)
    print(f"wrote {len(rows)} endstart rows -> {out_path}")
    return out_path


def generate_consistency(n: int, seed: int, out_path: Path) -> Path:
    """Generate ``n`` candidate-consistency yes/no rows (single-turn, balanced)."""
    bank = ConsistencyBank()
    agent = ConsistencyAgent()
    rng = random.Random(seed)
    game_no = GAME_NUMBERS["consistency"]
    targets = bank.sample_targets(n, "train", rng)

    rows: list[dict] = []
    pos = 0
    for i, target in enumerate(targets):
        board, candidate = consistency_decode(target)
        truth = is_consistent(candidate, board)
        completion = f"{render_consistency_reasoning(board, candidate)}\n<answer>{'yes' if truth else 'no'}</answer>"
        messages = agent.build_messages(ConsistencyGame(board, candidate, f"gen-{i}").state())
        check = ConsistencyGame(board, candidate, "chk").step(agent.parse_action(completion))
        if check.status != "correct":
            raise AssertionError(f"consistency self-check failed for {target!r}")
        assert _ok_budget(messages, completion)
        pos += int(truth)
        rows.append(sft_row(
            game_name="consistency", game_no=game_no, round=1, target=target,
            system=agent.system_prompt, messages=messages, completion=completion,
            valid=True, episode=i,
        ))

    _write_rows(rows, out_path)
    print(f"wrote {len(rows)} consistency rows -> {out_path}")
    print(f"  consistent (yes): {pos}   inconsistent (no): {len(rows) - pos}")
    return out_path


def _generate_multiturn(game: str, n: int, seed: int, out_path: Path, *, make_env, make_solver,
                        agent, max_rows: Optional[int] = None):
    """Shared driver for the programmatic multi-turn games (codebreaker, bullscows).

    Plays up to ``n`` episodes with an unbiased solver; emits one SFT row per turn for every
    solved episode (the solver always solves, so episodes are valid). Skips any rare unsolved or
    over-budget episode.

    ``max_rows`` caps the output **at an episode boundary** — a whole episode is either kept or
    dropped, never truncated mid-trajectory. So the round distribution is preserved (we never drop
    "only the first turns" or "only the last turns"); the cap just keeps fewer complete episodes.
    """
    base_rng = random.Random(seed)
    game_no = GAME_NUMBERS[game]
    rows: list[dict] = []
    n_turns = skipped = episodes = 0
    for ep in range(n):
        secret, env = make_env(base_rng)            # advance base_rng even if we later stop, for reproducibility
        if max_rows is not None and len(rows) >= max_rows:
            continue
        solver = make_solver(random.Random((seed + 1) * 1_000_003 + ep))
        traj = _play_multiturn(agent, env, solver)
        if traj.final.status != "won" or not all(_ok_budget(t.messages, t.response) for t in traj.turns):
            skipped += 1
            continue
        episode_rows = [
            sft_row(game_name=game, game_no=game_no, round=idx + 1, target=secret,
                    system=agent.system_prompt, messages=turn.messages, completion=turn.response,
                    valid=True, episode=ep)
            for idx, turn in enumerate(traj.turns)
        ]
        if max_rows is not None and len(rows) + len(episode_rows) > max_rows:
            continue                                 # whole-episode cap: skip rather than truncate
        rows.extend(episode_rows)
        n_turns += len(traj.turns)
        episodes += 1

    _write_rows(rows, out_path)
    print(f"wrote {len(rows)} {game} rows ({episodes} episodes, {n_turns / max(1, episodes):.1f} "
          f"turns/episode avg; {skipped} skipped) -> {out_path}")
    return out_path


def generate_codebreaker(n: int, seed: int, out_path: Path, max_rows: Optional[int] = None) -> Path:
    bank = CodebreakerBank()

    def make_env(rng):
        secret = bank.make_secret(rng)
        env = CodebreakerEnv(LocalCodebreakerClient(bank))
        env.reset(word=secret)
        return secret, env

    return _generate_multiturn("codebreaker", n, seed, out_path, make_env=make_env,
                               make_solver=CodebreakerSolver, agent=CodebreakerAgent(), max_rows=max_rows)


def generate_bullscows(n: int, seed: int, out_path: Path, max_rows: Optional[int] = None) -> Path:
    bank = BullsCowsBank()

    def make_env(rng):
        secret = bank.make_secret(rng)
        env = BullsCowsEnv(LocalBullsCowsClient(bank))
        env.reset(word=secret)
        return secret, env

    return _generate_multiturn("bullscows", n, seed, out_path, make_env=make_env,
                               make_solver=BullsCowsSolver, agent=BullsCowsAgent(), max_rows=max_rows)


_DEFAULT_OUT = {
    "charcount": "distillation/data/charcount_sft.jsonl",
    "validity": "distillation/data/validity_sft.jsonl",
    "rhyme": "distillation/data/rhyme_sft.jsonl",
    "charset": "distillation/data/charset_sft.jsonl",
    "tower": "distillation/data/tower_sft.jsonl",
    "endstart": "distillation/data/endstart_sft.jsonl",
    "codebreaker": "distillation/data/codebreaker_sft.jsonl",
    "bullscows": "distillation/data/bullscows_sft.jsonl",
    "consistency": "distillation/data/consistency_sft.jsonl",
}
_DEFAULT_N = {"charcount": 14000, "validity": 14000, "rhyme": 10000, "charset": 12000, "tower": 5000,
              "endstart": 6000, "codebreaker": 5000, "bullscows": 5000, "consistency": 10000}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Programmatically generate SFT data for a word-skill game.")
    ap.add_argument("--game", default="charcount",
                    choices=["charcount", "validity", "rhyme", "charset", "tower",
                             "endstart", "codebreaker", "bullscows", "consistency"],
                    help="which programmatic game to generate")
    ap.add_argument("--episodes", type=int, default=None, help="number of samples (default per game)")
    ap.add_argument("--wordle-min", type=int, default=4000,
                    help="charcount only: at least this many Wordle-vocab words")
    ap.add_argument("--seed", type=int, default=0, help="seed for reproducible sampling")
    ap.add_argument("--out", default=None, help="output JSONL path (default per game)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="multi-turn games only: cap output rows at an episode boundary (keeps "
                         "whole episodes, so the round distribution stays unbiased)")
    args = ap.parse_args(argv)

    n = args.episodes if args.episodes is not None else _DEFAULT_N[args.game]
    out = Path(args.out) if args.out is not None else Path(_DEFAULT_OUT[args.game])
    if args.game == "charcount":
        generate(n, args.wordle_min, args.seed, out)
    elif args.game == "validity":
        generate_validity(n, args.seed, out)
    elif args.game == "rhyme":
        generate_rhyme(n, args.seed, out)
    elif args.game == "charset":
        generate_charset(n, args.seed, out)
    elif args.game == "tower":
        generate_tower(n, args.seed, out)
    elif args.game == "endstart":
        generate_endstart(n, args.seed, out)
    elif args.game == "codebreaker":
        generate_codebreaker(n, args.seed, out, max_rows=args.max_rows)
    elif args.game == "bullscows":
        generate_bullscows(n, args.seed, out, max_rows=args.max_rows)
    elif args.game == "consistency":
        generate_consistency(n, args.seed, out)


if __name__ == "__main__":
    main()
