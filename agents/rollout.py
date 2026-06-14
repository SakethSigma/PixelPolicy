"""Generic episode drivers + observers. Game-agnostic and model-agnostic.

``run_episode`` drives one episode to completion through the three seams
(:class:`~agents.base.GameAgent`, :class:`~agents.base.Env`, an injected ``generate``)
and returns a :class:`~agents.base.Trajectory`. ``run_eval`` runs many episodes
concurrently for an eval harness.

The only difference between "watch it play" and "run N silently" is the
:class:`Observer` passed in. Generation (``generate``) is injected: the HTTP backend at
inference, the policy engine in training. Nothing here imports a game or a provider.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Protocol

from agents.base import Completion, Env, GameAgent, Trajectory, Turn

Generate = Callable[[list[list[dict]]], list[Completion]]


# ── Observers ──────────────────────────────────────────────────────────────────--


class Observer(Protocol):
    def on_start(self, state: Any) -> None: ...
    def on_step(self, turn: Turn, completion: Completion) -> None: ...
    def on_end(self, final: Any) -> None: ...


class NullObserver:
    """Headless — does nothing. Used by eval/training rollouts."""

    def on_start(self, state: Any) -> None: ...
    def on_step(self, turn: Turn, completion: Completion) -> None: ...
    def on_end(self, final: Any) -> None: ...


class TerminalObserver:
    """Demo — prints the model's reply + parsed action, then the game's own board.

    ``render_fn(state, console)`` is injected by the caller (Wordle wires in
    ``games.wordle.play.render_board``), so this observer stays game-agnostic. ``rich``
    is imported lazily, so importing this module never requires the ``[tui]`` extra.
    """

    def __init__(self, render_fn, *, pace: float = 0.0, step: bool = False, console=None):
        from rich.console import Console  # lazy: only the demo needs rich

        self._render = render_fn
        self._pace = pace
        self._step = step
        self._console = console or Console()

    def on_start(self, state: Any) -> None:
        self._console.print()
        self._render(state, self._console)

    def on_step(self, turn: Turn, completion: Completion) -> None:
        self._console.print(f"\n[dim]model:[/] {completion.text.strip()}")
        self._console.print(f"[bold]→ guess:[/] {turn.action}\n")
        self._render(turn.state, self._console)
        if self._step:
            try:
                self._console.input("[dim](enter for next move)[/]")
            except (EOFError, KeyboardInterrupt):
                pass
        elif self._pace:
            import time

            time.sleep(self._pace)

    def on_end(self, final: Any) -> None:
        if getattr(final, "status", None) == "won":
            self._console.print(f"\n[bold green]Won in {final.current_round} guesses![/]")
        else:
            tgt = getattr(final, "target", None)
            self._console.print(f"\n[bold red]Lost.[/] The word was [bold]{tgt}[/].")


# ── Drivers ──────────────────────────────────────────────────────────────────────


def run_episode(
    agent: GameAgent,
    env: Env,
    generate: Generate,
    observer: Observer | None = None,
) -> Trajectory:
    """Drive one episode to completion; record and return its :class:`Trajectory`.

    ``env`` must already be reset. Each step builds the prompt from the current state
    *and the turns so far* (``history``), so the conversation is threaded from the
    trajectory — the agent stays stateless.
    """
    observer = observer or NullObserver()
    state = env.state()
    turns: list[Turn] = []

    observer.on_start(state)
    while state.status == "in_progress":
        messages = agent.build_messages(state, turns)
        completion = generate([messages])[0]
        action = agent.parse_action(completion.text)
        state = env.step(action)
        turn = Turn(messages=messages, response=completion.text, action=action, state=state)
        turns.append(turn)
        observer.on_step(turn, completion)
    observer.on_end(state)

    return Trajectory(turns=turns, final=state)


def run_eval(
    pairs: Iterable[tuple[GameAgent, Env]],
    generate: Generate,
    *,
    concurrency: int = 8,
    on_result=None,
) -> list[Trajectory]:
    """Run many already-reset ``(agent, env)`` episodes concurrently.

    Concurrency is at the episode level (a thread per in-flight game); the inference
    server batches the overlapping requests itself (e.g. vLLM continuous batching), so
    we don't hand-vectorize generation here.

    ``on_result(i, traj)`` (if given) is called on the MAIN thread the instant each
    episode finishes — used by the eval to flush every completed episode to disk
    immediately, so a crash/Ctrl-C never loses finished work (resume picks up the rest).
    """
    pairs = list(pairs)
    results: list[Trajectory | None] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(run_episode, agent, env, generate): i
            for i, (agent, env) in enumerate(pairs)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            tr = fut.result()
            results[i] = tr
            if on_result is not None:
                on_result(i, tr)
    return [r for r in results if r is not None]


def win_rate(trajectories: Iterable[Trajectory]) -> float:
    """Fraction of finished trajectories whose final status is ``"won"``."""
    trajs = list(trajectories)
    if not trajs:
        return 0.0
    won = sum(getattr(t.final, "status", None) == "won" for t in trajs)
    return won / len(trajs)
