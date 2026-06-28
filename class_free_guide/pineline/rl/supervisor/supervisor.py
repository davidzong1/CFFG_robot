"""The supervisor daemon — orchestrates the per-cycle loop."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import SupervisorConfig
from .guardrails import Guardrails
from .llm_client import LLMClient, build_client
from .metric_collector import MetricCollector
from .objective import TrainingObjective
from .patcher import RewardPatcher
from .rollback import RollbackEvaluator
from .schema import RewardSchema
from .video_sampler import VideoSampler

log = logging.getLogger(__name__)


class Supervisor:
    """LLM-driven reward auto-tuner running in a daemon thread."""

    def __init__(
        self,
        env: Any,
        writer: Any | None,
        log_dir: Path,
        config: SupervisorConfig,
        llm: LLMClient | None = None,
        iter_getter: Callable[[], int] | None = None,
        writer_getter: Callable[[], Any] | None = None,
        objective: TrainingObjective | None = None,
    ):
        self.env = env
        self.log_dir = Path(log_dir)
        self.cfg = config
        # ``writer`` may not exist yet (rsl_rl creates it inside ``learn``).
        # The patcher resolves it lazily via ``writer_getter`` on each apply.
        self._writer_getter = writer_getter or (lambda: writer)

        # Resolve the training objective: explicit arg > config path > default.
        if objective is not None:
            self.objective = objective
        else:
            self.objective = TrainingObjective.load(config.objective_path)

        base_env = env.unwrapped
        self.reward_manager = base_env.reward_manager
        self._known_terms: list[str] = list(self.reward_manager.active_terms)

        # Resolve schema path: explicit > packaged default.
        if config.schema_path:
            schema_path = Path(config.schema_path)
        else:
            schema_path = Path(__file__).parent / "config" / "schema.yaml"
        self.schema = RewardSchema.load(schema_path)

        self.collector = MetricCollector(self.log_dir, window=config.metric_window)
        self.video_sampler = VideoSampler(
            self.log_dir / "videos" / "train",
            clips_per_cycle=config.clips_per_cycle,
            frames_per_clip=config.video_frames_per_clip,
        )
        self.llm = llm or build_client(config)
        self.patcher = RewardPatcher(
            self.log_dir,
            weight_setter=self._set_weight,
            weight_getter=self._get_weight,
            tb_writer=None,
            all_terms_getter=lambda: [n for n in self._known_terms if n in self.schema.bounds],
        )
        # Hand the patcher a late-binding writer reference so it picks up the
        # SummaryWriter that rsl_rl creates inside ``learn``.
        self.patcher.tb_writer = _LazyWriter(self._writer_getter)
        self.guardrails = Guardrails(self.schema, config, self.log_dir)
        self.rollback = RollbackEvaluator(self.patcher, self.guardrails)

        # Iteration counter: callers can supply one; otherwise we fall back
        # to ``env.unwrapped.common_step_counter`` which all mjlab envs have.
        if iter_getter is not None:
            self._iter_getter = iter_getter
        else:
            self._iter_getter = lambda: int(getattr(base_env, "common_step_counter", 0))

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        (self.log_dir / "supervisor").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        # Persist v0 baseline so we always have a rollback target.
        self.patcher.write_initial_snapshot(self._current_weights())
        # Record the objective up front so the audit log is self-contained.
        self._write_audit(
            {
                "kind": "objective",
                "objective": {
                    "name": self.objective.name,
                    "summary": self.objective.summary,
                    "priorities": self.objective.priorities,
                    "avoid": self.objective.avoid,
                    "notes": self.objective.notes,
                },
            }
        )
        self._thread = threading.Thread(target=self._loop, name="rl-supervisor", daemon=True)
        self._thread.start()
        log.info(
            "[supervisor] started: provider=%s model=%s interval=%.0fmin objective=%s",
            self.cfg.provider,
            self.cfg.model,
            self.cfg.interval_min,
            self.objective.name,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        # Warmup: let training produce some tfevents before the first cycle.
        min_to_sec = int(self.cfg.interval_min * 60)
        warm_until = time.time() + min_to_sec
        while not self._stop.is_set():
            if time.time() >= warm_until:
                break
            self._stop.wait(timeout=1.0)

        while not self._stop.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                log.exception("[supervisor] cycle failed: %s", exc)
                self._write_audit({"kind": "cycle_error", "error": repr(exc)})
            # Sleep in small chunks so stop() returns promptly.
            slept = 0.0
            while slept < min_to_sec and not self._stop.is_set():
                time.sleep(min(60.0, min_to_sec - slept))
                slept += 60.0

    def _run_cycle(self) -> None:
        if self.guardrails.killswitch_present():
            self._write_audit({"kind": "paused"})
            return

        current_iter = self._iter_getter()
        if current_iter < self.cfg.warmup_iters:
            return

        snapshot = self.collector.snapshot()
        if not snapshot.series:
            self._write_audit({"kind": "skipped", "reason": "no tfevents yet", "iter": current_iter})
            return
        snapshot_summary = snapshot.summary(self.cfg.metric_downsample)

        current_weights = self._current_weights()
        frames = self.video_sampler.frames(overlay={"iter": str(current_iter), "v": str(self.patcher.version)})

        diag = self.llm.diagnose(
            snapshot_summary,
            current_weights,
            frames,
            objective_block=self.objective.to_prompt_block(),
        )
        proposal = self.llm.propose(
            diag,
            self._schema_summary(),
            current_weights,
            objective_block=self.objective.to_prompt_block(),
        )

        guard = self.guardrails.evaluate(proposal, current_weights, current_iter)
        if not guard.ok:
            self._write_audit(
                {
                    "kind": "rejected",
                    "iter": current_iter,
                    "reason": guard.reason,
                    "diagnose": diag,
                    "proposal": proposal,
                }
            )
            # Still evaluate rollback against any previously armed rule.
            self.rollback.maybe_rollback(snapshot_summary, current_iter)
            return

        record = self.patcher.apply(
            patch=guard.clamped_patch,
            rationale=str(proposal.get("rationale", "")),
            expected_effect=str(proposal.get("expected_effect", "")),
            rollback_if=str(proposal.get("rollback_if", "")),
            diagnose=diag,
            current_iter=current_iter,
        )
        self.guardrails.note_apply(current_iter)
        if record.rollback_if:
            self.rollback.watch(record.rollback_if)

    # ------------------------------------------------------------------
    # Reward-manager glue
    # ------------------------------------------------------------------

    def _get_weight(self, name: str) -> float:
        return float(self.reward_manager.get_term_cfg(name).weight)

    def _set_weight(self, name: str, value: float) -> None:
        self.reward_manager.get_term_cfg(name).weight = float(value)

    def _current_weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self._known_terms:
            if name in self.schema.bounds:
                out[name] = self._get_weight(name)
        return out

    def _schema_summary(self) -> dict[str, dict[str, float]]:
        return {name: {"min": b.min, "max": b.max, "default": b.default} for name, b in self.schema.bounds.items() if name in self._known_terms}

    def _write_audit(self, payload: dict[str, Any]) -> None:
        payload = {"ts": time.time(), **payload}
        with open(self.log_dir / "supervisor" / "audit.jsonl", "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")


class _LazyWriter:
    """Resolves a SummaryWriter on each call so we tolerate late init."""

    def __init__(self, getter: Callable[[], Any]):
        self._getter = getter

    def _w(self):
        try:
            return self._getter()
        except Exception:
            return None

    def add_scalar(self, *args, **kwargs):
        w = self._w()
        if w is not None and hasattr(w, "add_scalar"):
            w.add_scalar(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        w = self._w()
        if w is not None and hasattr(w, "add_text"):
            w.add_text(*args, **kwargs)
