"""Reads TensorBoard scalars and returns a compact snapshot for the LLM."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except Exception:  # pragma: no cover
    EventAccumulator = None  # type: ignore


@dataclass
class ScalarSeries:
    name: str
    steps: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def stats(self) -> dict[str, float]:
        if not self.values:
            return {"n": 0}
        n = len(self.values)
        mean = sum(self.values) / n
        var = sum((v - mean) ** 2 for v in self.values) / max(n - 1, 1)
        slope = 0.0
        if n >= 2 and self.steps[-1] != self.steps[0]:
            slope = (self.values[-1] - self.values[0]) / (
                self.steps[-1] - self.steps[0]
            )
        return {
            "n": float(n),
            "mean": mean,
            "std": math.sqrt(var),
            "last": self.values[-1],
            "min": min(self.values),
            "max": max(self.values),
            "slope_per_step": slope,
        }


@dataclass
class MetricSnapshot:
    step: int
    series: dict[str, ScalarSeries]

    def summary(self, downsample: int) -> dict[str, Any]:
        """Compact JSON-serialisable summary for prompting."""
        out: dict[str, Any] = {"step": self.step, "scalars": {}}
        for name, s in self.series.items():
            ds_steps, ds_values = _downsample(s.steps, s.values, downsample)
            out["scalars"][name] = {
                "stats": s.stats(),
                "points": list(zip(ds_steps, ds_values)),
            }
        return out


def _downsample(
    steps: list[int], values: list[float], target: int
) -> tuple[list[int], list[float]]:
    n = len(values)
    if n <= target:
        return steps, values
    step = n / target
    idx = [int(i * step) for i in range(target)]
    return [steps[i] for i in idx], [values[i] for i in idx]


class MetricCollector:
    """Wraps ``EventAccumulator`` to extract recent training scalars."""

    # Scalar tags worth shipping. Substring-matched (case-sensitive).
    _PREFIXES = (
        "Rewards/",
        "Train/",
        "Loss/",
        "Policy/",
        "Episode_Reward/",
        "Episode_Termination/",
        "Episode/",
        "Curriculum/",
    )

    def __init__(self, log_dir: Path, window: int = 200):
        self.log_dir = Path(log_dir)
        self.window = window
        self._acc: Any | None = None

    def _accumulator(self):
        if EventAccumulator is None:
            raise RuntimeError("tensorboard is required for MetricCollector")
        # Recreate each call so we pick up new events without holding open
        # file handles between cycles. ``Reload`` is the supported API.
        acc = EventAccumulator(
            str(self.log_dir),
            size_guidance={"scalars": 0},  # 0 → keep everything
        )
        acc.Reload()
        return acc

    def snapshot(self) -> MetricSnapshot:
        acc = self._accumulator()
        tags = acc.Tags().get("scalars", [])
        series: dict[str, ScalarSeries] = {}
        max_step = 0
        for tag in tags:
            if not any(tag.startswith(p) for p in self._PREFIXES):
                continue
            events = acc.Scalars(tag)
            if not events:
                continue
            events = events[-self.window :]
            s = ScalarSeries(name=tag)
            for ev in events:
                s.steps.append(int(ev.step))
                s.values.append(float(ev.value))
            series[tag] = s
            if s.steps:
                max_step = max(max_step, s.steps[-1])
        return MetricSnapshot(step=max_step, series=series)
