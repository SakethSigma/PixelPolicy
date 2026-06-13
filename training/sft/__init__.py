"""SFT harness: load the distillation dataset, fine-tune via TRL `SFTTrainer`, push checkpoints.

Modules:
- `format`          — row → (prompt, completion), byte-identical to inference (the only chat-template seam)
- `data_flat`       — non-curriculum loader (serves the wordle-only baseline and the full set)
- `data_curriculum` — curriculum loader (widening / sorted / weighted strategies)
- `train`           — the SFTTrainer entrypoint (CLI, 4 epochs, save + push every checkpoint)
- `upload`          — push one local checkpoint dir to the Hub (per-epoch revision)
"""
