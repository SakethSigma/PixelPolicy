"""Pack per-turn `TurnSample`s into the tensors verl's GRPO update consumes.

Each `TurnSample` is one (prompt, completion) row. verl trains the policy/KL loss on the **response**
tokens only (the prompt — including the feedback — is masked), and computes the GRPO advantage by
normalizing the per-sample reward within each `uid` group. So we produce, per batch:

* `input_ids`      = left-padded(prompt) ++ right-padded(response)   [B, max_prompt+max_resp]
* `attention_mask` = 1 on real tokens, 0 on pad
* `position_ids`   = cumsum(attention_mask)-1 (standard)
* `responses`      = right-padded response ids                       [B, max_resp]
* `response_mask`  = 1 on real response tokens (the loss/advantage mask)
* non-tensor: `uid` (group key), `reward` (per-sample scalar), `target`, `round`

The pure padding/mask logic lives in `pad_left`/`pad_right` and is unit-tested with plain lists; the
torch assembly is import-local so the rest of the package needs no torch.

NOTE (VERL-VERSION): the exact `DataProto` field names verl expects can shift across releases — verify
against the installed verl's `verl.protocol.DataProto` / the DAPO recipe at the `--max-steps 5` probe.
The grouping key field is `uid` (a.k.a. `index`) consumed by `compute_grpo_outcome_advantage`.
"""

from __future__ import annotations

from typing import Any

from training.grpo.rollout import TurnSample


def pad_left(seq: list[int], length: int, pad_id: int) -> tuple[list[int], list[int]]:
    """Left-pad (for prompts): returns (padded_ids, attention_mask)."""
    seq = seq[-length:]                       # truncate from the LEFT-most overflow if needed
    n_pad = length - len(seq)
    return [pad_id] * n_pad + seq, [0] * n_pad + [1] * len(seq)


def pad_right(seq: list[int], length: int, pad_id: int) -> tuple[list[int], list[int]]:
    """Right-pad (for responses): returns (padded_ids, mask) — mask is the loss/response mask."""
    seq = seq[:length]                        # truncate the tail of an over-long completion
    n_pad = length - len(seq)
    return seq + [pad_id] * n_pad, [1] * len(seq) + [0] * n_pad


def pack_rows(samples: list[TurnSample], *, pad_id: int, max_prompt_len: int,
              max_response_len: int) -> dict[str, Any]:
    """Pure (list-based) packing — produces the per-row arrays verl needs, no torch.

    Returns a dict of equal-length lists (one entry per sample). The torch tensorization is done by
    `to_dataproto`; keeping this pure makes the padding/mask logic unit-testable.
    """
    rows = {
        "prompt_ids": [], "prompt_mask": [],
        "response_ids": [], "response_mask": [],
        "uid": [], "reward": [], "target": [], "round": [], "game": [],
    }
    for s in samples:
        p_ids, p_mask = pad_left(s.prompt_ids, max_prompt_len, pad_id)
        r_ids, r_mask = pad_right(s.response_ids, max_response_len, pad_id)
        rows["prompt_ids"].append(p_ids)
        rows["prompt_mask"].append(p_mask)
        rows["response_ids"].append(r_ids)
        rows["response_mask"].append(r_mask)
        rows["uid"].append(s.uid)
        rows["reward"].append(float(s.reward))
        rows["target"].append(s.target)
        rows["round"].append(s.round)
        rows["game"].append(s.game)
    return rows


def to_dataproto(samples: list[TurnSample], *, pad_id: int, max_prompt_len: int,
                 max_response_len: int):
    """Build a verl `DataProto` from per-turn samples. Import-local (needs torch + verl).

    The reward is placed as a sequence-level `token_level_scores`/reward verl turns into the
    group-normalized advantage (`adv_estimator=grpo`, grouped by `uid`). VERL-VERSION: confirm the
    exact tensor keys (`input_ids`/`responses`/`response_mask`/`token_level_scores`) and the non-tensor
    `uid` key name against the installed verl.
    """
    import torch
    from verl import DataProto  # type: ignore

    rows = pack_rows(samples, pad_id=pad_id, max_prompt_len=max_prompt_len,
                     max_response_len=max_response_len)
    prompt_ids = torch.tensor(rows["prompt_ids"], dtype=torch.long)
    prompt_mask = torch.tensor(rows["prompt_mask"], dtype=torch.long)
    response_ids = torch.tensor(rows["response_ids"], dtype=torch.long)
    response_mask = torch.tensor(rows["response_mask"], dtype=torch.long)

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0) * attention_mask

    # token_level_scores: [B, max_resp], the per-sample reward placed at each row's LAST real response
    # token (verl's GRPO sums these per sequence, normalizes within the uid group → advantage).
    token_level_scores = torch.zeros_like(response_ids, dtype=torch.float32)
    lengths = response_mask.sum(dim=1)                       # real response length per row
    rewards = torch.tensor(rows["reward"], dtype=torch.float32)
    for i, ln in enumerate(lengths.tolist()):
        if ln > 0:
            token_level_scores[i, int(ln) - 1] = rewards[i]

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response_ids,
        "response_mask": response_mask,
        "token_level_scores": token_level_scores,
    }
    non_tensor = {
        "uid": rows["uid"],                 # GRPO group key
        "reward": rows["reward"],           # per-sample scalar reward → group-normalized advantage
        "target": rows["target"],
        "round": rows["round"],
        "game": rows["game"],
    }
    return DataProto.from_dict(tensors=batch, non_tensors=non_tensor)
