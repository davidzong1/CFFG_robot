"""Configuration dataclasses for the supervisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SupervisorConfig:
    """Runtime configuration for the supervisor daemon."""

    # Cycle pacing
    interval_min: float = 15.0
    cooldown_iters: int = 200
    warmup_iters: int = 200

    # Patch guardrails
    max_patch_fields: int = 3
    max_rel_change: float = 0.30
    max_consecutive_rollbacks: int = 2

    # Early stopping: the supervisor scores the model each cycle and can
    # halt training when the policy is good enough.
    early_stopping: bool = True       # master enable/disable
    pass_score: float = 75.0          # score threshold (0-100)
    min_training_iters: int = 1000    # min iters before early stop is allowed

    # Data shipped to the LLM
    metric_window: int = 200  # last-N TB points considered
    metric_downsample: int = 50  # points kept after downsampling
    clips_per_cycle: int = 2
    video_frames_per_clip: int = 6

    # LLM provider
    provider: str = "anthropic"  # "anthropic" | "openai" | "openrouter" | "stub"
    model: str = "claude-opus-4-7"
    max_tokens: int = 2048
    temperature: float = 0.3
    api_base: str | None = None   # base URL for openai / openrouter / any compatible API
    api_key: str | None = None   # inline key (takes precedence over api_key_env)
    api_key_env: str = "ANTHROPIC_API_KEY"
    # Extra HTTP headers injected into every LLM request (useful for OpenRouter,
    # self-hosted gateways, etc.).  Example:
    #   extra_headers:
    #     HTTP-Referer: "https://my.repo"
    #     X-Title: "My Bot"
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Schema file
    schema_path: str | None = None

    # Training-objective JSON; overrides the built-in default if set.
    objective_path: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path | None) -> "SupervisorConfig":
        """Load a YAML config; missing path → defaults."""
        if path is None:
            return SupervisorConfig()
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in SupervisorConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = SupervisorConfig(**kwargs)
        cfg.extra = extra
        return cfg


@dataclass
class RewardBound:
    """Per-term constraint loaded from the schema YAML."""

    min: float
    max: float
    default: float
    description: str = ""

    def clamp(self, value: float) -> float:
        return max(self.min, min(self.max, value))

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max
