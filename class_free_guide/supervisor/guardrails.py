"""Validation rules applied before a patch is allowed through."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SupervisorConfig
from .schema import RewardSchema


@dataclass
class GuardResult:
    ok: bool
    reason: str
    clamped_patch: dict[str, float]


class Guardrails:
    """Stateless validators + a cooldown/blacklist tracker."""

    def __init__(self, schema: RewardSchema, cfg: SupervisorConfig, log_dir: Path):
        self.schema = schema
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.last_apply_iter: int = -10**9
        self.blacklist: set[str] = set()

    def killswitch_present(self) -> bool:
        return (self.log_dir / "supervisor" / "PAUSE").exists()

    def evaluate(
        self,
        response: dict[str, Any],
        current_weights: dict[str, float],
        current_iter: int,
    ) -> GuardResult:
        if self.killswitch_present():
            return GuardResult(False, "killswitch active (supervisor/PAUSE exists)", {})

        if current_iter - self.last_apply_iter < self.cfg.cooldown_iters:
            return GuardResult(
                False,
                f"in cooldown (iter {current_iter}, last apply {self.last_apply_iter})",
                {},
            )

        if not isinstance(response, dict):
            return GuardResult(False, "response is not a JSON object", {})

        required = {"patch", "rationale", "expected_effect", "rollback_if"}
        missing = required - response.keys()
        if missing:
            return GuardResult(False, f"missing fields: {sorted(missing)}", {})

        patch = response.get("patch") or {}
        if not isinstance(patch, dict):
            return GuardResult(False, "patch is not a JSON object", {})
        if not patch:
            return GuardResult(False, "empty patch (no-op)", {})

        if len(patch) > self.cfg.max_patch_fields:
            return GuardResult(
                False,
                f"patch touches {len(patch)} fields > limit {self.cfg.max_patch_fields}",
                {},
            )

        blacklisted = self.blacklist & patch.keys()
        if blacklisted:
            return GuardResult(False, f"blacklisted terms: {sorted(blacklisted)}", {})

        ok, reason, clamped = self.schema.validate_patch(
            patch, current_weights, self.cfg.max_rel_change
        )
        if not ok:
            return GuardResult(False, reason, {})

        return GuardResult(True, "ok", clamped)

    def note_apply(self, current_iter: int) -> None:
        self.last_apply_iter = current_iter

    def note_rollback(self, terms: list[str]) -> None:
        for t in terms:
            self.blacklist.add(t)
