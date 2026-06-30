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
from .safety import SafetyConfig, SafetyMonitor
from .schema import RewardSchema
import dzipc

log = logging.getLogger(__name__)


def _is_fatal_error(exc: BaseException) -> bool:
    """Return True for errors that will never heal by retrying
    (e.g. non-vision model, invalid API key)."""
    msg = str(exc).lower()
    return "vision-capable" in msg or "image_url" in msg


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

        # ---- safety monitor -------------------------------------------------
        safety_cfg = SafetyConfig.from_dict(config.safety)
        self.safety = SafetyMonitor(
            log_dir,
            safety_cfg,
            audit_writer=self._write_audit,
        )

        # ---- threading ------------------------------------------------------
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # ---- IPC (dzipc shared-memory publisher) ---------------------------
        self._ipc_pub: "dzipc.PublisherIPC" | None = None
        self._ipc_msg_template: "dzipc.Supervisor" | None = None
        self._ipc_topic_data: "dzipc.TopicData" | None = None
        self._next_cycle_at: float = 0.0       # timestamp when the next cycle fires
        self._ipc_heartbeat_ready = threading.Event()
        self._ipc_heartbeat_thread: threading.Thread | None = None
        self._last_ipc_event: tuple[str, str] | None = None  # (update_time, additional_info) pending for subscriber
        self._last_ipc_event_lock = threading.Lock()

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
        # Wire up the SHM publisher so external tooling can monitor us.
        self._init_ipc_publisher()
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
        if self._ipc_heartbeat_thread is not None:
            self._ipc_heartbeat_thread.join(timeout=min(timeout, 2.0))
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        # Warmup: let training produce some data before the first cycle.
        min_to_sec = int(self.cfg.interval_min * 60)
        self._next_cycle_at = time.time() + min_to_sec
        self._ipc_heartbeat_ready.set()
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
                self._publish_status(
                    update_time="error",
                    additional_info=json.dumps({"event": "cycle_error", "error": repr(exc)}),
                )
                # Fatal errors (e.g. non-vision model) must stop training
                # so GPU hours are not wasted on a supervisor that can never work.
                if _is_fatal_error(exc):
                    log.critical(
                        "[supervisor] FATAL error — signalling training to stop. %s",
                        exc,
                    )
                    self._stop.set()
                    if self._stopping_event is not None:
                        self._stopping_event.set()
                    return  # exit the daemon loop immediately
            # Schedule next cycle; heartbeat reads _next_cycle_at.
            self._next_cycle_at = time.time() + min_to_sec
            # Sleep in small chunks so stop() returns promptly, interleaving
            # safety checks at the configured cadence.
            safety_interval = min(self.safety.config.check_interval_s, min_to_sec)
            slept = 0.0
            while slept < min_to_sec and not self._stop.is_set():
                # Run safety checks at configured cadence
                if self.safety.config.enabled:
                    self._run_safety_check()
                sleep_chunk = min(safety_interval, min_to_sec - slept)
                self._stop.wait(timeout=sleep_chunk)
                slept += sleep_chunk

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

    def _run_safety_check(self) -> bool:
        """Run programmatic safety checks against TB scalars.

        Returns True if a stop action was triggered (caller should abort
        the current LLM cycle).
        """
        if not self.safety.config.enabled:
            return False
        try:
            current_iter = self._iter_getter()
            violations = self.safety.check(current_iter)
            for v in violations:
                self.safety.execute(v, self)
                if v.threshold.action == "stop":
                    return True
        except Exception:
            log.exception("[supervisor] safety check failed")
        return False

    def _run_cycle(self) -> None:
        if self.guardrails.killswitch_present():
            self._write_audit({"kind": "paused"})
            self._publish_status(
                update_time="paused",
                additional_info=json.dumps({"event": "paused"}),
            )
            return

        # Run safety checks before the LLM cycle — stop/rollback take priority
        # over LLM diagnosis.
        if self._run_safety_check():
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
        t0 = time.monotonic()
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
        thinking_time = time.monotonic() - t0

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
            self._publish_status(
                update_time=str(current_iter),
                additional_info=json.dumps(
                    {"event": "rejected", "iter": current_iter, "reason": guard.reason}
                ),
            )
            return

        record = self.patcher.apply(
            patch=guard.clamped_patch,
            rationale=str(proposal.get("rationale", "")),
            expected_effect=str(proposal.get("expected_effect", "")),
            rollback_if=str(proposal.get("rollback_if", "")),
            diagnose=diag,
            current_iter=current_iter,
            thinking_time=thinking_time,
        )
        self.guardrails.note_apply(current_iter)
        if record.rollback_if:
            self.rollback.watch(record.rollback_if)
        self._publish_status(
            update_time=str(current_iter),
            additional_info=json.dumps(
                {
                    "event": "patched",
                    "iter": current_iter,
                    "version": self.patcher.version,
                    "weights": current_weights,
                    "clamped": guard.clamped_patch,
                }
            ),
        )

    # ------------------------------------------------------------------
    # IPC (shared-memory publisher for external monitoring)
    # ------------------------------------------------------------------

    def _init_ipc_publisher(self) -> None:
        """Create a SHM publisher so external tooling can watch supervisor state.

        Call once, typically from ``start()``.  No-op when IPC is disabled in
        config or when the dzipc native module cannot be loaded.
        """
        if not self.cfg.ipc_enabled:
            return
        try:
            self._ipc_msg_template = dzipc.Supervisor()
            self._ipc_topic_data = dzipc.make_topic_data(self._ipc_msg_template)
            self._ipc_pub = dzipc.PublisherIPCPtrMake(
                self._ipc_topic_data,
                self.cfg.ipc_topic,
                self.cfg.ipc_domain,
                dzipc.IPC_SHM,
                verbose=False,
            )
            self._ipc_pub.InitChannel()
            log.info(
                "[supervisor] ipc publisher ready: topic=%s domain=%s transport=shm",
                self.cfg.ipc_topic,
                self.cfg.ipc_domain,
            )
            # Start a background thread that publishes heartbeat every second.
            self._ipc_heartbeat_thread = threading.Thread(
                target=self._ipc_heartbeat_loop, name="supervisor-ipc", daemon=True
            )
            self._ipc_heartbeat_thread.start()
        except Exception as exc:
            log.warning("[supervisor] ipc publisher init failed: %s", exc)
            self._ipc_pub = None
            self._ipc_msg_template = None
            self._ipc_topic_data = None

    def _ipc_heartbeat_loop(self) -> None:
        """Background thread: publish status every 1 s so external monitors see
        up-to-date remaining time until the next supervisor cycle.

        When a subscriber first connects (or reconnects after a gap) the most
        recent event-driven status is replayed so late-joining monitors receive
        the current state immediately.
        """
        # Wait until _loop() has written the first _next_cycle_at.
        self._ipc_heartbeat_ready.wait()
        was_subscribed = False
        while not self._stop.is_set():
            try:
                pub = self._ipc_pub
                subscribed = pub is not None and pub.has_subscribed()
                if subscribed:
                    # Subscriber just appeared — replay last event first.
                    if not was_subscribed:
                        self._flush_pending_event()
                    remaining = max(0.0, self._next_cycle_at - time.time())
                    msg = dzipc.Supervisor()
                    msg.update_time = str(int(remaining))
                    msg.additional_info = json.dumps(
                        {
                            "event": "heartbeat",
                            "iter": self._iter_getter(),
                            "version": self.patcher.version,
                            "next_cycle_in_s": int(remaining),
                        }
                    )
                    if not pub.publish(msg):
                        log.warning(
                            "[supervisor] heartbeat publish failed (topic=%s)",
                            self.cfg.ipc_topic,
                        )
                was_subscribed = subscribed
            except Exception:
                pass  # swallow — don't crash the heartbeat thread
            self._stop.wait(timeout=1.0)

    def _flush_pending_event(self) -> None:
        """Replay the last stored event so a late-joining subscriber gets current state."""
        with self._last_ipc_event_lock:
            event = self._last_ipc_event
        if event is None or self._ipc_pub is None or self._ipc_msg_template is None:
            return
        update_time, additional_info = event
        try:
            msg = dzipc.Supervisor()
            msg.update_time = update_time
            msg.additional_info = additional_info
            if not self._ipc_pub.publish(msg):
                log.warning(
                    "[supervisor] flush pending event failed (topic=%s)",
                    self.cfg.ipc_topic,
                )
        except Exception:
            pass

    def _publish_status(self, update_time: str, additional_info: str = "") -> None:
        """Publish a Supervisor message over SHM (fire-and-forget).

        The event is always stored so the heartbeat thread can replay it when a
        subscriber connects later.  When a subscriber is already present the
        publish happens immediately (no C++ stderr warning).
        """
        with self._last_ipc_event_lock:
            self._last_ipc_event = (update_time, additional_info)
        if self._ipc_pub is None or self._ipc_msg_template is None:
            return
        if not self._ipc_pub.has_subscribed():
            return
        try:
            msg = dzipc.Supervisor()
            msg.update_time = update_time
            msg.additional_info = additional_info
            if not self._ipc_pub.publish(msg):
                log.warning(
                    "[supervisor] ipc publish failed (topic=%s, event=%s)",
                    self.cfg.ipc_topic,
                    update_time,
                )
        except Exception as exc:
            log.debug("[supervisor] ipc publish failed: %s", exc)

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
