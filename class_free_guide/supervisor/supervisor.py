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
from .interfaces import SupervisorCallbacks
from .llm_client import LLMClient, build_client
from .objective import TrainingObjective
from .patcher import RewardPatcher
from .rollback import RollbackEvaluator
from .schema import RewardSchema

log = logging.getLogger(__name__)


class Supervisor:
    """LLM-driven auto-tuning daemon, framework-agnostic via callback injection.

    Instantiate directly with a ``SupervisorCallbacks`` bundle for non-RL
    frameworks, or use ``from_rl_env()`` for backward-compatible RL wiring.

    Example (non-RL, tuning learning rate + weight decay)::

        params = {"learning_rate": 1e-3, "weight_decay": 1e-4}

        callbacks = SupervisorCallbacks(
            metrics_getter=my_metrics_fn,
            frames_getter=lambda overlay=None: [],
            params_getter=lambda: dict(params),
            param_setter=lambda n, v: params.__setitem__(n, v),
            known_params_getter=lambda: list(params.keys()),
        )

        sup = Supervisor(
            log_dir=Path("./tuning_logs"),
            config=SupervisorConfig(...),
            callbacks=callbacks,
            iter_getter=lambda: epoch,
        )
        sup.start()
    """

    def __init__(
        self,
        log_dir: Path,
        config: SupervisorConfig,
        callbacks: SupervisorCallbacks,
        *,
        llm: LLMClient | None = None,
        iter_getter: Callable[[], int],
        writer_getter: Callable[[], Any] | None = None,
        objective: TrainingObjective | None = None,
        stopping_event: threading.Event | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.cfg = config

        # ---- callbacks (framework-agnostic I/O) -----------------------------
        self._callbacks = callbacks
        self._known_terms: list[str] = list(callbacks.known_params_getter())
        self._iter_getter = iter_getter
        self._writer_getter = writer_getter

        # ---- objective ------------------------------------------------------
        if objective is not None:
            self.objective = objective
        else:
            self.objective = TrainingObjective.load(config.objective_path)

        # ---- schema ---------------------------------------------------------
        if config.schema_path:
            schema_path = Path(config.schema_path)
        else:
            schema_path = Path(__file__).parent / "config" / "schema.yaml"
        self.schema = RewardSchema.load(schema_path)

        # ---- LLM client -----------------------------------------------------
        self.llm = llm or build_client(config)

        # ---- stopping -------------------------------------------------------
        self._stopping_event = stopping_event

        # ---- patcher (wired to callbacks) -----------------------------------
        self.patcher = RewardPatcher(
            self.log_dir,
            weight_setter=callbacks.param_setter,
            weight_getter=self._get_weight,
            tb_writer=None,
            all_terms_getter=lambda: [n for n in self._known_terms if n in self.schema.bounds],
        )
        if writer_getter is not None:
            self.patcher.tb_writer = _LazyWriter(writer_getter)

        # ---- guardrails & rollback ------------------------------------------
        self.guardrails = Guardrails(self.schema, config, self.log_dir)
        self.rollback = RollbackEvaluator(self.patcher, self.guardrails)

        # ---- threading ------------------------------------------------------
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        (self.log_dir / "supervisor").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Factory: RL backward-compat
    # ------------------------------------------------------------------

    @classmethod
    def from_rl_env(
        cls,
        env: Any,
        writer: Any | None,
        log_dir: Path,
        config: SupervisorConfig,
        *,
        llm: LLMClient | None = None,
        iter_getter: Callable[[], int] | None = None,
        writer_getter: Callable[[], Any] | None = None,
        objective: TrainingObjective | None = None,
        stopping_event: threading.Event | None = None,
    ) -> "Supervisor":
        """Construct a Supervisor wired to an RL environment's RewardManager.

        This convenience factory preserves backward compatibility with the
        original ``Supervisor(env=..., writer=..., ...)`` signature.  It
        imports ``MetricCollector`` and ``VideoSampler`` lazily so
        non-RL users of the raw ``__init__`` don't need TensorBoard.
        """
        from .metric_collector import MetricCollector
        from .video_sampler import VideoSampler

        base_env = env.unwrapped
        reward_manager = base_env.reward_manager

        collector = MetricCollector(log_dir, window=config.metric_window)
        video_sampler = VideoSampler(
            log_dir / "videos" / "train",
            clips_per_cycle=config.clips_per_cycle,
            frames_per_clip=config.video_frames_per_clip,
        )

        callbacks = SupervisorCallbacks(
            metrics_getter=collector.snapshot,
            frames_getter=video_sampler.frames,
            param_setter=lambda name, val: setattr(reward_manager.get_term_cfg(name), "weight", float(val)),
            params_getter=lambda: {name: float(reward_manager.get_term_cfg(name).weight) for name in reward_manager.active_terms},
            known_params_getter=lambda: list(reward_manager.active_terms),
        )

        if iter_getter is None:
            iter_getter = lambda: int(getattr(base_env, "common_step_counter", 0))

        _writer_getter = writer_getter or (lambda: writer)

        return cls(
            log_dir=log_dir,
            config=config,
            callbacks=callbacks,
            llm=llm,
            iter_getter=iter_getter,
            writer_getter=_writer_getter,
            objective=objective,
            stopping_event=stopping_event,
        )

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
                "objective": self.objective.as_dict(),
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
        # Warmup: let training produce some data before the first cycle.
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

    def _check_completion(self, diag: dict[str, Any], current_iter: int) -> None:
        """If the LLM-assigned score meets the pass threshold and we've trained
        long enough, signal the training loop to stop early."""
        if not self.cfg.early_stopping:
            return
        score = diag.get("score")
        if score is None:
            return
        try:
            score_val = int(score)
        except (TypeError, ValueError):
            return
        if not (0 <= score_val <= 100):
            return
        if score_val < self.cfg.pass_score or current_iter < self.cfg.min_training_iters:
            return

        # All criteria met — signal early stop.
        self._write_audit(
            {
                "kind": "early_stop",
                "iter": current_iter,
                "score": score_val,
                "pass_score": self.cfg.pass_score,
                "min_training_iters": self.cfg.min_training_iters,
            }
        )
        (self.log_dir / "supervisor" / "EARLY_STOP").touch()
        if self._stopping_event is not None:
            self._stopping_event.set()

    def _run_cycle(self) -> None:
        if self.guardrails.killswitch_present():
            self._write_audit({"kind": "paused"})
            return

        current_iter = self._iter_getter()
        if current_iter < self.cfg.warmup_iters:
            return

        # ---- collect observations via callbacks -----------------------------
        snapshot = self._callbacks.metrics_getter()
        if not snapshot.series:
            self._write_audit({"kind": "skipped", "reason": "no metrics yet", "iter": current_iter})
            return
        snapshot_summary = snapshot.summary(self.cfg.metric_downsample)

        current_weights = self._current_weights()
        frames = self._callbacks.frames_getter(overlay={"iter": str(current_iter), "v": str(self.patcher.version)})

        # ---- LLM diagnose ---------------------------------------------------
        diag = self.llm.diagnose(
            snapshot_summary,
            current_weights,
            frames,
            objective_block=self.objective.to_prompt_block(),
        )

        # ---- early-stop check -----------------------------------------------
        self._check_completion(diag, current_iter)
        if self._stopping_event is not None and self._stopping_event.is_set():
            return  # training is done — skip propose/patch this cycle

        # ---- LLM propose ----------------------------------------------------
        proposal = self.llm.propose(
            diag,
            self._schema_summary(),
            current_weights,
            objective_block=self.objective.to_prompt_block(),
        )

        # ---- validate & apply -----------------------------------------------
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
    # Parameter helpers (delegate to callbacks)
    # ------------------------------------------------------------------

    def _get_weight(self, name: str) -> float:
        return float(self._callbacks.params_getter()[name])

    def _current_weights(self) -> dict[str, float]:
        all_params = self._callbacks.params_getter()
        return {name: all_params[name] for name in self._known_terms if name in self.schema.bounds}

    def _schema_summary(self) -> dict[str, dict[str, float]]:
        return {name: {"min": b.min, "max": b.max, "default": b.default} for name, b in self.schema.bounds.items() if name in self._known_terms}

    def _write_audit(self, payload: dict[str, Any]) -> None:
        payload = {"ts": time.time(), **payload}
        with open(self.log_dir / "supervisor" / "audit.jsonl", "a") as f:
            f.write(json.dumps(payload, default=str, indent=2, ensure_ascii=False) + "\n\n")


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
