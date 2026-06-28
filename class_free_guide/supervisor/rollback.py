"""Evaluates the ``rollback_if`` condition attached to each applied patch."""

from __future__ import annotations

import operator
import re
from typing import Any

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


_RULE_RE = re.compile(r"^\s*(\S+)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


class RollbackEvaluator:
    """Holds the most recent applied patch and checks its trigger."""

    def __init__(self, patcher, guardrails):
        self.patcher = patcher
        self.guardrails = guardrails
        self._last_rule: str | None = None
        self._consecutive: dict[str, int] = {}

    def watch(self, rule: str) -> None:
        self._last_rule = rule

    def maybe_rollback(self, snapshot_summary: dict[str, Any], current_iter: int) -> bool:
        rule = self._last_rule
        if not rule:
            return False
        m = _RULE_RE.match(rule)
        if not m:
            return False
        tag, op_str, threshold = m.group(1), m.group(2), float(m.group(3))
        scalars = snapshot_summary.get("scalars", {})
        series = scalars.get(tag)
        if not series:
            return False
        last = series.get("stats", {}).get("last")
        if last is None:
            return False
        if not _OPS[op_str](float(last), threshold):
            return False

        # Trigger: restore the previous version.
        if not self.patcher.history:
            return False
        last_record = self.patcher.history[-1]
        prev_version = max(last_record.version - 1, 0)
        self.patcher.restore(prev_version, current_iter)

        # Bookkeep consecutive rollbacks per term.
        for name in last_record.patch.keys():
            n = self._consecutive.get(name, 0) + 1
            self._consecutive[name] = n
            if n >= self.guardrails.cfg.max_consecutive_rollbacks:
                self.guardrails.note_rollback([name])
        self._last_rule = None
        return True
