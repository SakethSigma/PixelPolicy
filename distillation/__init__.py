"""Distillation: Claude plays the games as a teacher; export SFT data; push to the Hub.

This package only *drives* the existing layers — it adds no game/agent/model logic of
its own. The data flow is:

    AnthropicBackend (agents/)  ──▶  run_eval (agents/rollout.py)  ──▶  list[Trajectory]
            │ teacher                         │ reuses the real game loop
            ▼                                 ▼
    generate.py  ──▶  data/raw/<game>.jsonl  ──▶  dataset.py (filter + explode)
                                                        │
                                                        ▼
                                              data/sft/<game>.jsonl
                                                        │
                                                  push.py ──▶  HuggingFace Hub

See PLAN.md in this folder for the full design and rationale.
"""
