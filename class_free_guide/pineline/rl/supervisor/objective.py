"""Training-objective spec loaded from a JSON file at startup.

The JSON tells the LLM *what* to optimise for ("walk stably", "walk fast",
"stay energy-efficient", ...). It is passed verbatim to the diagnose and
propose prompts so every patch decision is biased toward that goal.

Schema (all keys optional except ``name``)::

    {
        "name":        "stable_walk",
        "summary":     "Stable upright trot that tracks commanded velocity.",
        "priorities":  ["upright body", "clean foot contacts", "tracking accuracy"],
        "avoid":       ["high body roll/pitch", "foot slipping"],
        "notes":       "free-form extra context for the LLM"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingObjective:
    name: str = "default"
    summary: str = (
        "Track the commanded velocity stably with an upright body and clean "
        "foot contacts."
    )
    priorities: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    notes: str = ""

    @staticmethod
    def load(path: str | Path | None) -> "TrainingObjective":
        """Load an objective from a JSON file. ``None`` → built-in default."""
        if path is None:
            return TrainingObjective()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"objective file not found: {p}")
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"objective JSON must be an object, got {type(data)}")
        return TrainingObjective(
            name=str(data.get("name", "custom")),
            summary=str(data.get("summary", "")),
            priorities=list(data.get("priorities", [])),
            avoid=list(data.get("avoid", [])),
            notes=str(data.get("notes", "")),
        )

    def to_prompt_block(self) -> str:
        """Render as a compact text block injected into the LLM prompt."""
        lines = [f"# Training objective: {self.name}"]
        if self.summary:
            lines.append(self.summary)
        if self.priorities:
            lines.append("Priorities (in order):")
            for i, p in enumerate(self.priorities, 1):
                lines.append(f"  {i}. {p}")
        if self.avoid:
            lines.append("Avoid:")
            for a in self.avoid:
                lines.append(f"  - {a}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)
