"""Applies a validated patch atomically and logs the result."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class AppliedPatch:
    version: int
    iter: int
    patch: dict[str, float]
    before: dict[str, float]
    after: dict[str, float]
    rationale: str
    expected_effect: str
    rollback_if: str
    diagnose: dict[str, Any] = field(default_factory=dict)


class RewardPatcher:
    """Owns the on-disk version history and live-weight mutation.

    ``weight_setter`` is a callback ``(term_name, new_value) -> None``;
    typically ``lambda n, v: env.unwrapped.reward_manager.get_term_cfg(n).weight = v``.
    """

    def __init__(
        self,
        log_dir: Path,
        weight_setter: Callable[[str, float], None],
        weight_getter: Callable[[str], float],
        tb_writer: Any | None = None,
        all_terms_getter: Callable[[], list[str]] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.cfg_dir = self.log_dir / "reward_cfg"
        self.audit_dir = self.log_dir / "supervisor"
        self.cfg_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.weight_setter = weight_setter
        self.weight_getter = weight_getter
        self.tb_writer = tb_writer
        self._all_terms_getter = all_terms_getter
        self.version: int = self._discover_version()
        self.history: list[AppliedPatch] = []

    def _discover_version(self) -> int:
        existing = sorted(self.cfg_dir.glob("v*.yaml"))
        if not existing:
            return 0
        try:
            return max(int(p.stem[1:]) for p in existing if p.stem[1:].isdigit())
        except ValueError:
            return 0

    def write_initial_snapshot(self, current_weights: dict[str, float]) -> None:
        """Persist v0 = the baseline weights so rollbacks have a target."""
        if self.version > 0:
            return
        self._write_yaml(0, current_weights, meta={"baseline": True})

    def apply(
        self,
        patch: dict[str, float],
        rationale: str,
        expected_effect: str,
        rollback_if: str,
        diagnose: dict[str, Any],
        current_iter: int,
        *,
        thinking_time: float | None = None,
    ) -> AppliedPatch:
        before = {k: self.weight_getter(k) for k in patch.keys()}
        for name, new_val in patch.items():
            self.weight_setter(name, float(new_val))
        after = {k: self.weight_getter(k) for k in patch.keys()}
        self.version += 1

        # Persist all weights for replay/rollback, not just the diff.
        all_weights = {k: self.weight_getter(k) for k in self._known_keys(before)}
        meta: dict[str, Any] = {
            "iter": current_iter,
            "patch": patch,
            "rationale": rationale,
            "expected_effect": expected_effect,
            "rollback_if": rollback_if,
        }
        if thinking_time is not None:
            meta["thinking_time_s"] = round(thinking_time, 2)
        self._write_yaml(
            self.version,
            all_weights,
            meta=meta,
        )

        record = AppliedPatch(
            version=self.version,
            iter=current_iter,
            patch=patch,
            before=before,
            after=after,
            rationale=rationale,
            expected_effect=expected_effect,
            rollback_if=rollback_if,
            diagnose=diagnose,
        )
        self.history.append(record)
        self._audit("applied", record)
        self._tb_log(record, current_iter)
        return record

    def restore(self, target_version: int, current_iter: int) -> None:
        """Restore weights from a previously saved version on disk."""
        path = self.cfg_dir / f"v{target_version:02d}.yaml"
        if not path.exists():
            return
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        weights = data.get("weights", {})
        for name, val in weights.items():
            try:
                self.weight_setter(name, float(val))
            except Exception:
                continue
        self.version += 1
        self._write_yaml(
            self.version,
            weights,
            meta={"iter": current_iter, "rolled_back_to": target_version},
        )
        self._audit(
            "rollback",
            {
                "version": self.version,
                "iter": current_iter,
                "rolled_back_to": target_version,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _known_keys(self, before: dict[str, float]) -> list[str]:
        if self._all_terms_getter is not None:
            try:
                return list(self._all_terms_getter())
            except Exception:
                pass
        return list(before.keys())

    def _write_yaml(
        self, version: int, weights: dict[str, float], meta: dict[str, Any]
    ) -> None:
        body = {
            "version": version,
            "ts": time.time(),
            "meta": meta,
            "weights": {k: float(v) for k, v in weights.items()},
        }
        path = self.cfg_dir / f"v{version:02d}.yaml"
        tmp = path.with_suffix(".yaml.tmp")
        with open(tmp, "w") as f:
            yaml.safe_dump(body, f, sort_keys=True)
        os.replace(tmp, path)

        current = self.cfg_dir / "current.yaml"
        tmp_link = self.cfg_dir / "current.yaml.tmp"
        if tmp_link.exists():
            tmp_link.unlink()
        try:
            os.symlink(path.name, tmp_link)
            os.replace(tmp_link, current)
        except OSError:
            # FS without symlink support: fall back to a copy.
            with open(current, "w") as f:
                yaml.safe_dump(body, f, sort_keys=True)

    def _audit(self, kind: str, payload: Any) -> None:
        line = {"kind": kind, "ts": time.time()}
        if hasattr(payload, "__dict__"):
            line.update({k: v for k, v in payload.__dict__.items()})
        elif isinstance(payload, dict):
            line.update(payload)
        with open(self.audit_dir / "audit.jsonl", "a") as f:
            f.write(json.dumps(line, default=str) + "\n")

    def _tb_log(self, record: AppliedPatch, current_iter: int) -> None:
        if self.tb_writer is None:
            return
        try:
            self.tb_writer.add_scalar(
                "reward_cfg/version", record.version, current_iter
            )
            for name, val in record.after.items():
                self.tb_writer.add_scalar(
                    f"reward_cfg/{name}_weight", val, current_iter
                )
            self.tb_writer.add_text(
                "reward_cfg/rationale",
                f"v{record.version}: {record.rationale}",
                current_iter,
            )
        except Exception:
            pass
