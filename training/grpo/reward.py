"""Dense, information-gain reward shaping for GRPO on Wordle (and the other games).

Why dense: sparse correctness (1.0 win / 0.0 else) starves GRPO at a ~2-7% solve rate —
most rollout groups all-fail, so the group-relative advantage is zero and there is nothing
to learn from. We add shaping that pays the model for *making progress* every turn, while
keeping the terminal win term dominant so it optimizes *solving*, not color-farming.

The shaping is deliberately **information-gain**, not raw color count:

* a green (``✓``) pays only the FIRST time a position is confirmed — correctly *keeping* an
  already-pinned letter on a later turn earns nothing;
* a yellow (``-``) pays only the FIRST time a letter is discovered to be in the word.

Re-confirming letters you already know is free. That is the "explore, don't just exploit the
words you already know fit" signal the experiment is after.

The single source of truth for *solved* stays ``status == "won"`` — identical to what
``inference/evaluate.py`` reports — so training and eval never disagree on the headline metric.

This module has two entry points:

* :func:`compute_reward` — pure function over an :class:`EpisodeOutcome`; unit-tested, no verl.
* :func:`compute_score`  — the verl ``custom_reward_function`` signature; reads the structured
  outcome the agent loop threads into ``extra_info`` and dispatches per game.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional, TypedDict

# ---------------------------------------------------------------------------
# Format check (one turn's raw reply)
# ---------------------------------------------------------------------------
# With thinking enabled the Qwen3.5 chat template opens ``<think>`` in the *prompt*, so the
# model's reply carries its reasoning, a closing ``</think>``, then the ``<guess>``. We therefore
# match the CLOSING think tag followed (eventually) by a single 5-letter ``<guess>`` — mirroring
# how ``agents/wordle/agent.py`` parses, and how the SFT data was shaped. ``_GUESS5`` is the
# stricter "5 alpha chars inside the last guess tag" check the user asked for.
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_GUESS5 = re.compile(r"<guess>\s*([A-Za-z]{5})\s*</guess>", re.IGNORECASE | re.DOTALL)


def is_format_valid(response: str, *, require_think: bool = True) -> bool:
    """True iff a turn's reply is well formed: a closed ``</think>`` (when required) and a
    ``<guess>`` carrying exactly 5 letters. This is the per-turn format reward gate."""
    if not response:
        return False
    if require_think and not _THINK_CLOSE.search(response):
        return False
    return _GUESS5.search(response) is not None


# ---------------------------------------------------------------------------
# Reward weights (all CLI-overridable for ablation)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardWeights:
    """Shaping weights. Greens > yellows; novelty is round-decayed; the terminal win is OFF by default.

    The reward is **purely per-turn local** (see :func:`per_turn_rewards`): each turn earns its own
    format + invalid + (round-decayed) novelty. ``win_bonus`` defaults to **0.0** — novel-green already
    encodes winning and a near-never-firing terminal would only add a low-variance, collapse-prone,
    trajectory-coupled term. It is kept as a safety-valve flag (added to the winning turn only).
    """

    green_new: float = 0.30      # NEW green: a ✓ at a position never before confirmed
    yellow_new: float = 0.10     # NEW yellow: an in-word letter discovered for the first time
    novelty_decay: float = 0.90  # per-round decay on novelty: novelty*decay**(round-1) → discover early
    format_ok: float = 0.05      # well-formed <think>…</think> + 5-char <guess> this turn
    format_bad: float = -0.05    # malformed turn
    invalid_guess: float = -0.20 # RoundResult.error set (wrong length / not a word)
    win_bonus: float = 0.0       # terminal, OFF by default (safety valve; added to the winning turn only)
    win_discount_k: float = 0.10 # linear discount; see win_discount below
    win_discount_mode: str = "linear"  # "linear" -> bonus*(1 - k*(r-1)/m); "geometric" -> bonus*gamma**(r-1)
    win_gamma: float = 0.9       # geometric discount base (used when win_discount_mode == "geometric")
    require_think: bool = True   # whether the format gate insists on a </think> block

    @classmethod
    def from_overrides(cls, **kw: Any) -> "RewardWeights":
        """Build from a dict, ignoring ``None`` values (CLI flags left unset)."""
        base = asdict(cls())
        base.update({k: v for k, v in kw.items() if v is not None})
        return cls(**base)


def win_discount(rounds_used: int, max_rounds: int, w: RewardWeights) -> float:
    """Multiplier in (0, 1]: a win on round 1 is worth the full bonus; later wins are discounted."""
    r = max(1, rounds_used)
    if w.win_discount_mode == "geometric":
        return float(w.win_gamma ** (r - 1))
    # linear: full at r=1, shrinking by k per round as a fraction of the horizon
    return max(0.0, 1.0 - w.win_discount_k * (r - 1) / max(1, max_rounds))


# ---------------------------------------------------------------------------
# Structured episode outcome (threaded from the agent loop)
# ---------------------------------------------------------------------------
class TurnOutcome(TypedDict):
    guess: str               # the parsed action ("" if no <guess> tag)
    feedback: list[str]      # [f.value for f in RoundResult.feedback] -> "✓"/"-"/"x"; [] if invalid
    error: Optional[str]     # RoundResult.error.value if the guess was invalid, else None
    format_valid: bool       # is_format_valid(turn.response)


class EpisodeOutcome(TypedDict):
    target: str
    status: str              # "won" | "lost" | "in_progress"
    rounds_used: int
    max_rounds: int
    turns: list[TurnOutcome]


# ---------------------------------------------------------------------------
# The dense Wordle reward — PURELY PER-TURN LOCAL (one R_t per turn, no broadcast)
# ---------------------------------------------------------------------------
def per_turn_rewards(outcome: EpisodeOutcome, weights: RewardWeights) -> list[float]:
    """One local reward ``R_t`` per turn — the live training signal.

    ``R_t = format_t + invalid_t + novelty_t·decay**(round-1)``, where novelty counts only NEW
    greens/yellows (re-confirmations pay nothing — tracked across the episode). No broadcast: a turn's
    reward depends only on that turn, so an early turn is never blamed for a later mistake, and
    reward-std stays high even at ~0% win rate. If ``weights.win_bonus > 0`` (default 0.0), a
    round-discounted terminal is added to the **winning turn only** (the last turn of a won episode).
    """
    known_green_pos: set[int] = set()   # positions ever scored ✓ (positional information)
    known_letters: set[str] = set()     # letters ever known in-word (✓ or -); membership information
    rewards: list[float] = []

    for idx, t in enumerate(outcome["turns"]):
        decay = weights.novelty_decay ** idx          # idx 0 = round 1 → full; later rounds decayed
        r = weights.format_ok if t["format_valid"] else weights.format_bad
        if t["error"]:
            # Invalid guess: consumes a round, no per-letter feedback -> format term + penalty only.
            rewards.append(r + weights.invalid_guess)
            continue
        guess = (t["guess"] or "").upper()
        novel = 0.0
        for i, fb in enumerate(t["feedback"]):
            if i >= len(guess):
                break
            letter = guess[i]
            if fb == "✓":
                if i not in known_green_pos:
                    novel += weights.green_new        # new positional confirmation
                    known_green_pos.add(i)
                known_letters.add(letter)             # a green also fixes membership
            elif fb == "-":
                if letter not in known_letters:
                    novel += weights.yellow_new       # new letter-membership discovery
                    known_letters.add(letter)
            # "x" (gray): elimination, not a positive discovery here -> no reward
        rewards.append(r + novel * decay)

    # Terminal win — OFF by default; if enabled, only the winning (last) turn gets it.
    if weights.win_bonus > 0 and outcome["status"] == "won" and rewards:
        rewards[-1] += weights.win_bonus * win_discount(
            outcome["rounds_used"], outcome["max_rounds"], weights)

    return rewards


def reward_breakdown(outcome: EpisodeOutcome, weights: RewardWeights) -> dict[str, float]:
    """Episode-level component sums {novel, format, invalid, win} for wandb logging (decayed novelty)."""
    known_green_pos: set[int] = set()
    known_letters: set[str] = set()
    novel = fmt = inval = 0.0
    for idx, t in enumerate(outcome["turns"]):
        decay = weights.novelty_decay ** idx
        fmt += weights.format_ok if t["format_valid"] else weights.format_bad
        if t["error"]:
            inval += weights.invalid_guess
            continue
        guess = (t["guess"] or "").upper()
        for i, fb in enumerate(t["feedback"]):
            if i >= len(guess):
                break
            letter = guess[i]
            if fb == "✓":
                if i not in known_green_pos:
                    novel += weights.green_new * decay
                    known_green_pos.add(i)
                known_letters.add(letter)
            elif fb == "-":
                if letter not in known_letters:
                    novel += weights.yellow_new * decay
                    known_letters.add(letter)
    win = 0.0
    if weights.win_bonus > 0 and outcome["status"] == "won":
        win = weights.win_bonus * win_discount(outcome["rounds_used"], outcome["max_rounds"], weights)
    return {"novel": novel, "format": fmt, "invalid": inval, "win": win}


def compute_reward_sparse(outcome: EpisodeOutcome, *, good_status: str) -> float:
    """Sparse correctness for the non-wordle games (Arm C): 1.0 solved, else 0.0.

    Single-turn games have one turn and ``good_status == "correct"``; multi-turn deduction
    games (codebreaker / bullscows) use ``"won"``. This matches the eval harness's notion of
    *solved* so training optimizes exactly what eval reports.
    """
    return 1.0 if outcome["status"] == good_status else 0.0


# ---------------------------------------------------------------------------
# verl entry point
# ---------------------------------------------------------------------------
# Process-global weights, set once by train.py at startup (verl imports this module by path
# and calls compute_score with no place to inject config, so we stash it here).
_WEIGHTS: RewardWeights = RewardWeights()
# Per-game good_status for the sparse fallback (Arm C). Populated from the registry by
# configure(); kept here so this module has no hard import of distillation at call time.
_GOOD_STATUS: dict[str, str] = {}


def configure(weights: Optional[RewardWeights] = None,
              good_status: Optional[dict[str, str]] = None) -> None:
    """Install the reward weights / per-game good_status used by :func:`compute_score`."""
    global _WEIGHTS, _GOOD_STATUS
    if weights is not None:
        _WEIGHTS = weights
    if good_status is not None:
        _GOOD_STATUS = dict(good_status)


# verl imports this module by file path in a *fresh* process, so in-process configure() does not
# reach it. The trainer therefore writes weights + good_status to a JSON sidecar and points
# GRPO_REWARD_WEIGHTS_JSON at it; we load it once on first scoring.
_ENV_LOADED = False


def _load_from_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    import json
    import os

    path = os.getenv("GRPO_REWARD_WEIGHTS_JSON")
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        blob = json.load(f)
    configure(
        weights=RewardWeights.from_overrides(**blob.get("weights", {})),
        good_status=blob.get("good_status"),
    )


def dump_config(path: str, weights: RewardWeights, good_status: dict[str, str]) -> None:
    """Write the weights + per-game good_status sidecar the reward process reads via the env var."""
    import json

    with open(path, "w") as f:
        json.dump({"weights": asdict(weights), "good_status": good_status}, f, indent=2)


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: Optional[dict] = None) -> float:
    """verl ``custom_reward_function`` entry point.

    The agent loop threads the structured :class:`EpisodeOutcome` into
    ``extra_info['episode_outcome']``. Wordle gets the dense shaping; every other game gets the
    sparse solved/not-solved reward (so easy games can't be color-farmed). ``data_source`` is
    the game name; ``ground_truth`` is the target (unused here — the outcome already carries it).
    """
    _load_from_env_once()
    extra_info = extra_info or {}
    outcome: Optional[EpisodeOutcome] = extra_info.get("episode_outcome")
    if outcome is None:
        # No structured trajectory (e.g. a mis-wired run): fall back to "did the final answer win".
        return 1.0 if (extra_info.get("status") == _GOOD_STATUS.get(data_source, "won")) else 0.0

    if data_source == "wordle":
        # Episode total = sum of the per-turn local rewards (the live path uses the per-turn list;
        # this scalar is only verl's reward-fn slot / offline sanity).
        return float(sum(per_turn_rewards(outcome, _WEIGHTS)))
    good = _GOOD_STATUS.get(data_source, "won")
    return compute_reward_sparse(outcome, good_status=good)
