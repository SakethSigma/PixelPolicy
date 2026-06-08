"""Stage 2 — turn raw Trajectories into per-move SFT samples (filter + explode).

Two pure steps over the raw JSONL dicts (no Claude calls, cheap and re-runnable):

  FILTER  keep only solved episodes — rejection-sampling distillation. Filter on the
          final GameState alone (status == "won", optionally won in <= max_guesses), so
          no reward module is needed.

  EXPLODE one sample per *move*, so we train on every move with loss on the completion
          only. The per-move prompt is ALREADY stored: run_episode saved
          `messages = build_messages(state, history_so_far)` into each Turn.messages. So a
          sample is just {messages: turn.messages, completion: turn.response} — byte-
          identical to what the student sees at inference (prior turns already have <think>
          stripped; the current move keeps its full <think>…</think><guess>…</guess>).
          This is why we explode rather than pack one multi-turn sequence — see PLAN.md.

Works on raw JSONL dicts (not re-validated Pydantic) because Trajectory.final and
Turn.state are typed `Any` and round-trip as plain dicts. So: final["status"],
final["rounds"], turn["messages"], turn["response"]. This keeps the converter fully
game-agnostic — it never imports a game.
"""

from __future__ import annotations

# TODO imports:
#   import json
#   from distillation.config import DistillConfig


# TODO: def is_solved(traj: dict, max_guesses: int) -> bool:
#   - return traj["final"]["status"] == "won" and len(traj["final"]["rounds"]) <= max_guesses
#   - (max_guesses == 6 means "any win"; lower it to keep only efficient wins.)


# TODO: def explode(traj: dict, game: str) -> Iterator[dict]:
#   - for turn in traj["turns"]:
#       optional per-turn guard: skip turns whose stored move was invalid/empty, so the
#       student never imitates a wasted guess — e.g. skip if turn["action"] == "" or the
#       turn's round in turn["state"] carries an `error`. (A strong teacher's won episodes
#       are usually clean, so this is a safety net.)
#       yield {"game": game, "messages": turn["messages"], "completion": turn["response"]}


# TODO: def build_game(cfg, game) -> Path:
#   1. read cfg.raw_dir / f"{game}.jsonl" line by line -> json.loads
#   2. keep = [t for t in trajs if is_solved(t, cfg.max_guesses)]
#   3. samples = [s for t in keep for s in explode(t, game)]
#   4. write samples to cfg.sft_dir / f"{game}.jsonl" (mkdir parents=True, exist_ok=True)
#   5. print kept/total episodes and total samples; return the output path.
#
# Sanity check worth asserting once (verification step 4): for a fresh agent,
#   agent.build_messages(state_before_move_i, history[:i]) == turn.messages   # train == inference
