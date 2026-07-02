"""Callback interfaces that decouple the Supervisor from any training framework.

All callbacks are called from the supervisor's daemon thread and must be
thread-safe if the training loop accesses the same state concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .metric_collector import MetricSnapshot


@dataclass
class SupervisorCallbacks:
    """Observation and action callbacks exchanged between Supervisor and
    the training framework.

    Observation side (framework → supervisor):
        *metrics_getter*   returns a MetricSnapshot for the current cycle.
        *frames_getter*    returns list of (label, png_bytes) tuples; takes
                           optional ``overlay`` keyword argument.
        *params_getter*    returns {param_name: current_value} for all
                           known parameters.
        *evaluation_getter* optionally returns auxiliary evaluation signals
                           such as actor-critic value statistics.

    Action side (supervisor → framework):
        *param_setter*        (name, value) -> None; mutates a live parameter.
        *known_params_getter* () -> list[str]; returns currently active
                              parameter names (cached once at init).
    """

    metrics_getter: Callable[[], MetricSnapshot]
    frames_getter: Callable[..., list[tuple[str, bytes]]]
    params_getter: Callable[[], dict[str, float]]
    param_setter: Callable[[str, float], None]
    known_params_getter: Callable[[], list[str]]
    evaluation_getter: Callable[[], dict[str, Any]] | None = None
