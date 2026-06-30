"""Programmatic safety monitor: evaluates TB scalars against thresholds at
high frequency, independent of LLM cycles."""

from __future__ import annotations

import json
import logging
import operator
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from .config import SafetyConfig, SafetyThreshold
from .metric_collector import MetricCollector

if TYPE_CHECKING:
    from .supervisor import Supervisor

log = logging.getLogger(__name__)

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass
class SafetyViolation:
    """A single triggered safety threshold with context."""

    threshold: SafetyThreshold
    actual_value: float
    tag_stats: dict[str, float]
    iter: int
    ts: float


class SafetyMonitor:
    """Checks TensorBoard scalars against configured thresholds.

    Operates at a higher frequency (default 60 s) than the LLM diagnosis
    cycle (default 15-30 min).  Designed to catch rapid degradation
    (e.g. physics instabilities, policy collapse) before the next
    scheduled LLM cycle.
    """

    def __init__(
        self,
        log_dir: Path,
        config: SafetyConfig,
        *,
        audit_writer: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.config = config
        self._audit_writer = audit_writer
        self._collector = MetricCollector(log_dir, window=config.metrics_window)
        self._last_check_iter: int = -1  # suppress duplicate triggers

    def check(self, current_iter: int) -> list[SafetyViolation]:
        """Read TB scalars, evaluate all thresholds, return triggered violations.

        Returns an empty list when no thresholds are breached or the check
        has already been run for the current training iteration.
        """
        if not self.config.enabled or not self.config.thresholds:
            return []
        if current_iter == self._last_check_iter:
            return []  # already checked this iteration

        self._last_check_iter = current_iter
        snapshot = self._collector.snapshot()
        if not snapshot.series:
            return []

        violations: list[SafetyViolation] = []
        for thresh in self.config.thresholds:
            series = snapshot.series.get(thresh.tag)
            if series is None:
                continue
            stats = series.stats()
            actual = stats.get("last", stats.get("mean", 0.0))
            if _OPS[thresh.op](actual, thresh.value):
                violations.append(
                    SafetyViolation(
                        threshold=thresh,
                        actual_value=actual,
                        tag_stats=stats,
                        iter=current_iter,
                        ts=time.time(),
                    )
                )
        return violations

    def execute(self, violation: SafetyViolation, supervisor: "Supervisor") -> None:
        """Execute the action prescribed by a triggered safety threshold."""
        thresh = violation.threshold
        self._write_audit(
            {
                "kind": "safety_violation",
                "iter": violation.iter,
                "tag": thresh.tag,
                "op": thresh.op,
                "threshold": thresh.value,
                "actual": violation.actual_value,
                "action": thresh.action,
                "description": thresh.description,
                "stats": violation.tag_stats,
            }
        )
        log.warning(
            "[supervisor] Safety violation: %s %s %s (actual %.4f at iter %d), action=%s",
            thresh.tag,
            thresh.op,
            thresh.value,
            violation.actual_value,
            violation.iter,
            thresh.action,
        )

        if thresh.action == "stop":
            self._do_stop(supervisor, violation)
        elif thresh.action == "rollback":
            self._do_rollback(supervisor, violation)

    def _do_stop(self, supervisor: "Supervisor", violation: SafetyViolation) -> None:
        """Signal the training loop to stop and write a marker file."""
        if supervisor._stopping_event is not None:
            supervisor._stopping_event.set()
        (self.log_dir / "supervisor" / "SAFETY_STOP").touch()
        self._write_audit(
            {
                "kind": "safety_stop_triggered",
                "iter": violation.iter,
                "tag": violation.threshold.tag,
            }
        )
        log.critical(
            "[supervisor] Safety threshold triggered EMERGENCY STOP: %s %s %s",
            violation.threshold.tag,
            violation.threshold.op,
            violation.threshold.value,
        )

    def _do_rollback(
        self, supervisor: "Supervisor", violation: SafetyViolation
    ) -> None:
        """Restore weights from the most recent patch version."""
        patcher = supervisor.patcher
        if not patcher.history:
            log.warning(
                "[supervisor] Safety rollback requested but no patch history exists"
            )
            return
        last_record = patcher.history[-1]
        prev_version = max(last_record.version - 1, 0)
        patcher.restore(prev_version, violation.iter)
        self._write_audit(
            {
                "kind": "safety_rollback",
                "iter": violation.iter,
                "tag": violation.threshold.tag,
                "from_version": last_record.version,
                "to_version": prev_version,
            }
        )
        log.warning(
            "[supervisor] Safety threshold triggered ROLLBACK to v%d at iter %d",
            prev_version,
            violation.iter,
        )

    def _write_audit(self, payload: dict[str, Any]) -> None:
        if self._audit_writer is not None:
            self._audit_writer(payload)
        else:
            # Fallback: write directly to audit.jsonl
            entry = {"ts": time.time(), **payload}
            audit_dir = self.log_dir / "supervisor"
            audit_dir.mkdir(parents=True, exist_ok=True)
            with open(audit_dir / "audit.jsonl", "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
