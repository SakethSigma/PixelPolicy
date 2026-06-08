"""CLI: drive the Wordle agent in a visible demo or a headless eval batch.

    # watch one game on a colored board
    uv run --package agents python -m agents.run --demo --word crane

    # run a silent batch, print a win-rate summary
    uv run --package agents python -m agents.run --episodes 20

Both share one loop; only the observer (Terminal vs none) and number of episodes differ.
The model endpoint comes from ``.env`` (see :mod:`agents.config`); ``--base-url`` /
``--model`` override.
"""

from __future__ import annotations

import argparse
from typing import Optional

from agents.config import AgentConfig
from agents.rollout import TerminalObserver, run_episode, run_eval, win_rate
from agents.wordle.agent import WordleAgent, WordleEnv
from games.wordle.client import LocalWordleClient
from games.wordle.game import WordBank


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run the Wordle agent.")
    p.add_argument("--demo", action="store_true",
                   help="watch one game on a colored board")
    p.add_argument("--episodes", type=int, default=10,
                   help="headless: how many games")
    p.add_argument("--mode", choices=["train", "val"], default="val")
    p.add_argument("--word", default=None, help="pin the target word")
    p.add_argument("--concurrency", type=int, default=8,
                   help="headless: parallel games")
    p.add_argument("--pace", type=float, default=0.0,
                   help="demo: seconds between moves")
    p.add_argument("--step", action="store_true",
                   help="demo: wait for Enter per move")
    p.add_argument("--base-url", default=None, help="override OPENAI_BASE_URL")
    p.add_argument("--model", default=None, help="override INFERENCE_MODEL")
    args = p.parse_args(argv)

    cfg = AgentConfig.from_env()
    # local: only running needs openai
    from agents.backend import OpenAICompatBackend, AnthropicBackend

    backend = OpenAICompatBackend(
        base_url=args.base_url or cfg.base_url,
        model=args.model or cfg.model,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        top_p=cfg.top_p,
        presence_penalty=cfg.presence_penalty,
        top_k=cfg.top_k,
        enable_thinking=cfg.enable_thinking,
    )
    backend = AnthropicBackend(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        effort="high"
    )

    bank = WordBank()

    def make_env() -> WordleEnv:
        env = WordleEnv(LocalWordleClient(bank))
        env.reset(mode=args.mode, word=args.word)
        return env

    if args.demo:
        from games.wordle.play import render_board

        env = make_env()
        observer = TerminalObserver(
            render_board, pace=args.pace, step=args.step)
        run_episode(WordleAgent(), env, backend.generate, observer)
        return

    pairs = [(WordleAgent(), make_env()) for _ in range(args.episodes)]
    trajs = run_eval(pairs, backend.generate, concurrency=args.concurrency)
    print(
        f"win rate: {win_rate(trajs):.1%}  ({len(trajs)} games, mode={args.mode})")


if __name__ == "__main__":
    main()
