"""Per-turn rollout driver — turn each multi-turn episode into one training sample PER MOVE.

This is the heart of the GRPO bridge. We do **not** use verl's single-sequence agent loop (it can't
represent stripped multi-turn without the Qwen3 delta-tokenization bug). Instead, for each episode we
drive the game **turn-by-turn exactly like inference** (`WordleAgent.build_messages` strips prior
`<think>`), and emit **one sample per turn**: `(prompt_ids_t, response_ids_t, reward R_t, uid)`.

* `prompt_ids_t` = the stripped conversation through turn *t* (byte-identical to what the model sees at
  inference turn *t*). Loss is masked to the completion only — the prompt, **including the feedback**,
  never receives gradient (same as SFT's prompt/completion split).
* `R_t` = a **purely per-turn local** reward (see `reward.per_turn_rewards`): no broadcast, no terminal
  by default. An early turn is never blamed for a later mistake.
* `uid = (target, round)` — the GRPO group key. Round-1 samples across the `G` episodes of a target
  share the *identical* prior → a perfect baseline; round-*t* samples share the round (best-available
  baseline). The group mean is an action-independent baseline, so the policy gradient is **unbiased**
  regardless of the priors diverging past round 1 (differing priors only cost variance).

`roll_batch` advances all episodes in lockstep so generation is **batched per turn** (one call across
all live episodes), which is what verl's rollout engine wants. `batch_generate` is injected (verl's
engine in training, a stub in tests), so this module needs neither verl nor a GPU to be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agents.base import Turn
from agents.wordle.agent import WordleAgent, WordleEnv
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank
from training.grpo.reward import (
    EpisodeOutcome,
    RewardWeights,
    compute_reward_sparse,
    is_format_valid,
    per_turn_rewards,
)

# prompts -> responses, both as token-id lists. Batched (one call for many in-flight turns).
BatchGenerate = Callable[[list[list[int]]], list[list[int]]]


@dataclass
class TurnSample:
    """One per-move training row handed to verl's GRPO update."""

    prompt_ids: list[int]
    response_ids: list[int]
    uid: str                       # GRPO group key = f"{target}#r{round}"
    target: str
    game: str
    round: int                     # 1-based
    reward: float = 0.0            # filled once the episode finishes


# ---------------------------------------------------------------------------
# (agent, env) construction — reuse the registry; wordle gets a shared WordBank
# ---------------------------------------------------------------------------
_SHARED_BANK: WordBank | None = None
_SPEC_CACHE: dict[str, Any] = {}


def make_pair(game: str, target: str) -> tuple[Any, Any, int, str]:
    """Return (agent, env-pinned-to-target, max_rounds, good_status) for ``game``."""
    if game == "wordle":
        global _SHARED_BANK
        if _SHARED_BANK is None:
            _SHARED_BANK = WordBank()
        env = WordleEnv(LocalWordleClient(_SHARED_BANK))
        env.reset(word=target)
        return WordleAgent(), env, 6, "won"
    # generic games via the registry (Arm C)
    from distillation.registry import GAMES

    if game not in _SPEC_CACHE:
        _SPEC_CACHE[game] = GAMES[game]()
    spec = _SPEC_CACHE[game]
    return spec.make_agent(), spec.make_env(target), spec.max_rounds, spec.good_status


# ---------------------------------------------------------------------------
# One in-flight episode
# ---------------------------------------------------------------------------
@dataclass
class _EpisodeRun:
    game: str
    target: str
    enable_thinking: bool = True
    require_think: bool = True
    agent: Any = None
    env: Any = None
    max_rounds: int = 6
    good_status: str = "won"
    history: list[Turn] = field(default_factory=list)
    samples: list[TurnSample] = field(default_factory=list)
    turn_outcomes: list[dict] = field(default_factory=list)
    state: Any = None

    def start(self) -> None:
        self.agent, self.env, self.max_rounds, self.good_status = make_pair(self.game, self.target)
        self.state = self.env.state()

    @property
    def live(self) -> bool:
        return self.state.status == "in_progress" and len(self.history) < self.max_rounds

    def prompt_ids(self, tokenizer) -> list[int]:
        messages = self.agent.build_messages(self.state, self.history)   # STRIPPED context (= inference)
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=self.enable_thinking)

    def step(self, tokenizer, prompt_ids: list[int], response_ids: list[int]) -> None:
        text = tokenizer.decode(response_ids)
        action = self.agent.parse_action(text)
        self.state = self.env.step(action)
        rnd = len(self.history) + 1
        self.history.append(Turn(messages=[], response=text, action=action, state=self.state))
        self.samples.append(TurnSample(
            prompt_ids=prompt_ids, response_ids=response_ids,
            uid=f"{self.target}#r{rnd}", target=self.target, game=self.game, round=rnd))
        rr = self.state.rounds[-1]
        self.turn_outcomes.append({
            "guess": action,
            "feedback": [f.value for f in rr.feedback],
            "error": rr.error.value if rr.error else None,
            "format_valid": is_format_valid(text, require_think=self.require_think),
        })

    def outcome(self) -> EpisodeOutcome:
        return {
            "target": self.target,
            "status": self.state.status,
            "rounds_used": self.state.current_round,
            "max_rounds": self.max_rounds,
            "turns": self.turn_outcomes,
        }

    def assign_rewards(self, weights: RewardWeights) -> None:
        outcome = self.outcome()
        if self.game == "wordle":
            rewards = per_turn_rewards(outcome, weights)            # per-turn local
        else:
            # Non-wordle: no dense per-turn signal -> sparse solved/not-solved, same value each turn.
            sparse = compute_reward_sparse(outcome, good_status=self.good_status)
            rewards = [sparse] * len(self.samples)
        for s, r in zip(self.samples, rewards):
            s.reward = r


# ---------------------------------------------------------------------------
# Vectorized rollout: advance every episode in lockstep, batching generation per turn
# ---------------------------------------------------------------------------
def roll_batch(
    specs: list[tuple[str, str]],          # [(game, target)] — one episode each (G copies of a target = a group)
    *,
    tokenizer: Any,
    batch_generate: BatchGenerate,
    weights: RewardWeights,
    enable_thinking: bool = True,
    require_think: bool = True,
    max_turns: int = 6,
) -> tuple[list[TurnSample], list[EpisodeOutcome]]:
    """Roll all episodes, returning the flat per-turn samples + per-episode outcomes.

    Generation is batched once per turn across all still-live episodes. The number of samples an
    episode contributes equals the number of turns it actually played (2 here, 5 there) — verl
    consumes a flat, row-oriented batch, so variable turn counts are just variable row counts.
    """
    runs = [_EpisodeRun(game=g, target=t, enable_thinking=enable_thinking, require_think=require_think)
            for g, t in specs]
    for r in runs:
        r.start()

    for _ in range(max_turns):
        live = [r for r in runs if r.live]
        if not live:
            break
        prompts = [r.prompt_ids(tokenizer) for r in live]
        responses = batch_generate(prompts)
        if len(responses) != len(live):
            raise RuntimeError(f"batch_generate returned {len(responses)} responses for {len(live)} prompts")
        for r, p, resp in zip(live, prompts, responses):
            r.step(tokenizer, p, resp)

    samples: list[TurnSample] = []
    outcomes: list[EpisodeOutcome] = []
    for r in runs:
        r.assign_rewards(weights)
        samples.extend(r.samples)
        outcomes.append(r.outcome())
    return samples, outcomes


def make_groups(targets: list[str], group_size: int, *, game: str = "wordle") -> list[tuple[str, str]]:
    """Expand each target into ``group_size`` episode specs (the GRPO group shares the target)."""
    specs: list[tuple[str, str]] = []
    for t in targets:
        specs.extend([(game, t)] * group_size)
    return specs
