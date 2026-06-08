"""CLI: `generate` (teacher rollouts) -> `build` (filter + explode) -> `push` (to Hub).

    uv run --package distillation python -m distillation.run generate --game wordle --n 500 --mode train
    uv run --package distillation python -m distillation.run build    --game wordle
    uv run --package distillation python -m distillation.run push

Mirrors agents/run.py: argparse + config from .env, CLI flags override. Each subcommand
is a thin wrapper over the matching module so the pieces stay independently testable.
"""

from __future__ import annotations

# TODO imports:
#   import argparse
#   from distillation.config import DistillConfig
#   from distillation.generate import generate_game
#   from distillation.dataset import build_game
#   from distillation.push import push


# TODO: def main(argv=None) -> None:
#   1. parser = argparse.ArgumentParser(...); sub = parser.add_subparsers(dest="cmd", required=True)
#
#   2. `generate` subcommand:
#        --game (required), --n (int, default 100), --mode {train,val} (default train),
#        --word (default None, pin a target for deterministic smoke tests),
#        --concurrency (int, override cfg), --model (override TEACHER_MODEL), --effort (override).
#        -> generate_game(cfg, args.game, args.n, mode=args.mode, word=args.word)
#
#   3. `build` subcommand:
#        --game (required), --max-guesses (int, override cfg.max_guesses).
#        -> build_game(cfg, args.game)
#
#   4. `push` subcommand:
#        --repo-id (override cfg.hub_repo_id), --private/--public flag, --dry-run
#        (build_dataset + print stats, skip the actual push).
#        -> push(cfg, ...)
#
#   5. cfg = DistillConfig.from_env(); apply any CLI overrides onto cfg; dispatch on args.cmd.


# TODO:
# if __name__ == "__main__":
#     main()
