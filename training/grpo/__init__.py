"""GRPO RL training of Qwen3.5-0.8B on the word games via verl.

This package continues an SFT checkpoint with Group-Relative Policy Optimization (GRPO),
reusing the existing game env / agent / eval machinery unchanged. The only game-aware code
here is the reward shaping (``reward.py``) and the multi-turn rollout bridge
(``wordle_agent_loop.py`` / ``game_loop.py``) that adapts our episode loop onto verl's
Agent Loop abstraction.

See ``PLAN.md`` (the canonical, crash-recoverable plan) and ``README.md`` (how to install
the isolated verl env and run the three experiment arms).
"""
