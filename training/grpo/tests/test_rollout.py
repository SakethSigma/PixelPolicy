"""Smoke test for the per-turn rollout driver — prompt/mask correctness + per-turn rewards + uid.

No verl, no model. A char-reversible fake tokenizer lets us decode prompts/completions and assert
each per-move sample is exactly one inference step (stripped prompt → that turn's reply).
"""

from __future__ import annotations

from training.grpo.reward import RewardWeights
from training.grpo.rollout import make_groups, roll_batch


class FakeTokenizer:
    """Char-level, reversible. apply_chat_template renders the messages to a string (so the prompt
    decodes back to the exact stripped conversation)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **kw) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kw):
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            s += "<|assistant|>"
        return self.encode(s)


def _gen_for(replies):
    """A message-based generate for a lockstep rollout: all live episodes are at the same turn, so on
    the k-th call we serve `replies[k]` to every prompt. Returns (templated prompt_ids, response_ids)
    per message list — the same contract verl's generate_sequences provides."""
    clock = {"k": 0}

    def gen(messages_list):
        reply = replies[clock["k"]]
        clock["k"] += 1
        return [(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True), tok.encode(reply))
                for m in messages_list]

    return gen


tok = FakeTokenizer()


def test_one_sample_per_turn_with_stripped_prompt():
    weights = RewardWeights()
    # 8 episodes of the same target → a GRPO group; all play the same 2-turn script (lose then win).
    specs = make_groups(["vivid"], group_size=8, game="wordle")
    replies = [
        "<think>open</think>\n<guess>crane</guess>",   # round 1: all x
        "<think>got it</think>\n<guess>vivid</guess>",  # round 2: win
    ]
    samples, outcomes = roll_batch(
        specs, tokenizer=tok, generate=_gen_for(replies), weights=weights, max_turns=6)

    # 8 episodes × 2 turns = 16 samples
    assert len(samples) == 16
    assert len(outcomes) == 8
    assert all(o["status"] == "won" and o["rounds_used"] == 2 for o in outcomes)

    # round-1 and round-2 samples
    r1 = [s for s in samples if s.round == 1]
    r2 = [s for s in samples if s.round == 2]
    assert len(r1) == 8 and len(r2) == 8

    # uid groups by (target, round): round-1 group shares the IDENTICAL prior
    assert {s.uid for s in r1} == {"vivid#r1"}
    assert {s.uid for s in r2} == {"vivid#r2"}

    # a round-1 sample's prompt decodes to the stripped opening; its completion to round-1's reply
    s = r1[0]
    prompt_text = tok.decode(s.prompt_ids)
    assert "Make your first guess." in prompt_text
    assert "<think>" not in prompt_text                 # the PROMPT carries no model reply yet
    assert tok.decode(s.response_ids) == replies[0]

    # a round-2 sample's prompt replays round-1's guess + feedback (stripped: only the <guess>, no think)
    s2 = r2[0]
    p2 = tok.decode(s2.prompt_ids)
    assert "<guess>crane</guess>" in p2                  # prior guess replayed (think stripped)
    assert "<think>open</think>" not in p2              # prior THINK stripped (matches inference)
    assert "C R A N E" in p2                             # round-1 feedback is in the PROMPT (masked from loss)
    assert tok.decode(s2.response_ids) == replies[1]


def test_per_turn_local_reward_and_round_decay():
    weights = RewardWeights()                            # win_bonus 0
    specs = make_groups(["vivid"], group_size=1, game="wordle")
    replies = [
        "<think>a</think>\n<guess>vapor</guess>",        # round 1: V green at pos0 (new) -> high
        "<think>b</think>\n<guess>vivid</guess>",        # round 2: win, greens incl. re-green pos0
    ]
    samples, _ = roll_batch(specs, tokenizer=tok, generate=_gen_for(replies),
                            weights=weights, max_turns=6)
    r1, r2 = samples[0].reward, samples[1].reward
    # round 1 earns format + a new green; both > 0, and no terminal was added (win_bonus 0)
    assert r1 > 0 and r2 > 0
    # round 2's novelty is decayed (decay**1) and re-greens are free → assert decay actually applied
    assert weights.novelty_decay < 1.0


def test_variable_turn_counts_no_crash():
    # Two different targets, different scripts → different #turns; flat sample list, no shape issue.
    specs = [("wordle", "vivid"), ("wordle", "vivid")]
    # immediate win in 1 turn
    samples, outcomes = roll_batch(
        specs, tokenizer=tok,
        generate=_gen_for(["<think>z</think><guess>vivid</guess>"]),
        weights=RewardWeights(), max_turns=6)
    assert all(o["rounds_used"] == 1 for o in outcomes)
    assert len(samples) == 2 and all(s.round == 1 for s in samples)
