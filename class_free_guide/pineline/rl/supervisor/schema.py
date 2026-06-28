"""Schema loader and patch validator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .config import RewardBound


@dataclass
class RewardSchema:
    """In-memory representation of ``reward_schema.yaml``."""

    bounds: dict[str, RewardBound]

    @staticmethod
    def load(path: str | Path) -> "RewardSchema":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        terms = data.get("terms", {})
        bounds: dict[str, RewardBound] = {}
        for name, spec in terms.items():
            bounds[name] = RewardBound(
                min=float(spec["min"]),
                max=float(spec["max"]),
                default=float(spec["default"]),
                description=str(spec.get("description", "")),
            )
        return RewardSchema(bounds=bounds)

    def names(self) -> Iterable[str]:
        return self.bounds.keys()

    def validate_patch(
        self,
        patch: dict[str, float],
        current: dict[str, float],
        max_rel_change: float,
    ) -> tuple[bool, str, dict[str, float]]:
        """Validate a patch dict ``{term_name: new_weight}``.

        Returns ``(ok, reason, clamped_patch)``. ``clamped_patch`` is the
        applied patch after clamping to bounds (only set if ``ok``).
        """
        if not isinstance(patch, dict):
            return False, "patch is not a dict", {}
        clamped: dict[str, float] = {}
        for name, new_val in patch.items():
            if name not in self.bounds:
                return False, f"unknown reward term: {name}", {}
            try:
                new_val_f = float(new_val)
            except (TypeError, ValueError):
                return False, f"non-numeric value for {name}: {new_val!r}", {}
            bound = self.bounds[name]
            if not bound.contains(new_val_f):
                return False, (
                    f"{name}={new_val_f} out of bounds [{bound.min}, {bound.max}]"
                ), {}
            old_val = current.get(name, bound.default)
            denom = max(abs(old_val), 1e-6)
            rel = abs(new_val_f - old_val) / denom
            if rel > max_rel_change:
                return False, (
                    f"{name} relative change {rel:.2f} exceeds limit {max_rel_change}"
                ), {}
            clamped[name] = new_val_f
        return True, "ok", clamped
