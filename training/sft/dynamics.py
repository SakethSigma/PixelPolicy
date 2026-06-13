"""Learning-dynamics instrumentation (method #1 from CURRICULUM/LEARNING_DYNAMICS notes).

A `TrainerCallback` that logs, every N optimizer steps, the **per-layer gradient norm** and the
**per-layer update norm** (‖Δparams‖ from the optimizer step) to **wandb** — the live "where is
learning happening right now" view, nearly free. Layers are grouped by transformer block index
(`layer_00`…), plus `embed` / `lm_head` / `final_norm` / `other`.

Wired into `train.py` via `--gradlog-steps` (0 disables); active only when `--report-to wandb`.
The other visualizations in `LEARNING_DYNAMICS_NOTES.md` are NOT implemented yet — brainstorm later.

Notes / limitations:
- Grad norm is read in `on_pre_optimizer_step` (after backward + clipping, before the step), so it
  reflects the gradients that actually drive the update. Update norm diffs a param snapshot taken
  there against the params after `on_optimizer_step`.
- Single-process / single-GPU view (norms are per local process). Snapshot adds ~one model's worth
  of memory transiently, but only on logging steps.
"""

from __future__ import annotations


def _build_callback():
    import re

    from transformers import TrainerCallback

    _LAYER_RE = re.compile(r"layers\.(\d+)\.")

    def _buckets(name: str) -> list[str]:
        """Buckets a param contributes to. Per transformer block we log the WHOLE block
        (`layer_NN`) AND its attention (`attn_NN`) / MLP (`mlp_NN`) sub-norms separately, so the
        viz can split attention vs feed-forward. (Layernorm params count toward `layer_NN` only.)"""
        m = _LAYER_RE.search(name)
        if m:
            nn = f"{int(m.group(1)):02d}"
            out = [f"layer_{nn}"]
            if "self_attn" in name:
                out.append(f"attn_{nn}")
            elif "mlp" in name:
                out.append(f"mlp_{nn}")
            return out
        if "embed" in name:
            return ["embed"]
        if "lm_head" in name:
            return ["lm_head"]
        if "norm" in name:
            return ["final_norm"]
        return ["other"]

    class GradUpdateNormCallback(TrainerCallback):
        def __init__(self, every: int = 50):
            self.every = max(1, every)
            self._model = None
            self._snap: dict | None = None
            self._grad_norm: dict = {}

        def on_train_begin(self, args, state, control, model=None, **kw):
            self._model = model
            return control

        def _logging_step(self, state) -> bool:
            return state.global_step > 0 and state.global_step % self.every == 0

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            m = model or self._model
            if m is None or not self._logging_step(state):
                self._snap = None
                return control
            grad_sq: dict = {}
            snap: dict = {}
            for n, p in m.named_parameters():
                if p.grad is None:
                    continue
                sq = p.grad.detach().float().pow(2).sum().item()
                for g in _buckets(n):
                    grad_sq[g] = grad_sq.get(g, 0.0) + sq
                snap[n] = p.detach().clone()
            self._grad_norm = {g: v ** 0.5 for g, v in grad_sq.items()}
            self._snap = snap
            return control

        def on_optimizer_step(self, args, state, control, model=None, **kw):
            m = model or self._model
            if m is None or self._snap is None:
                return control
            upd_sq: dict = {}
            for n, p in m.named_parameters():
                prev = self._snap.get(n)
                if prev is None:
                    continue
                sq = (p.detach() - prev).float().pow(2).sum().item()
                for g in _buckets(n):
                    upd_sq[g] = upd_sq.get(g, 0.0) + sq
            upd_norm = {g: v ** 0.5 for g, v in upd_sq.items()}
            self._snap = None
            self._log(state, self._grad_norm, upd_norm)
            return control

        @staticmethod
        def _log(state, gnorm: dict, unorm: dict) -> None:
            try:
                import wandb
            except ImportError:
                return
            if wandb.run is None:
                return
            d = {f"gradnorm/{g}": v for g, v in gnorm.items()}
            d.update({f"updnorm/{g}": v for g, v in unorm.items()})
            if gnorm:
                d["gradnorm/_total"] = sum(v * v for v in gnorm.values()) ** 0.5
            if unorm:
                d["updnorm/_total"] = sum(v * v for v in unorm.values()) ** 0.5
            wandb.log(d, step=state.global_step)

    return GradUpdateNormCallback


def __getattr__(name: str):
    """Lazily build the callback on first access (defers the transformers import)."""
    if name == "GradUpdateNormCallback":
        return _build_callback()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
