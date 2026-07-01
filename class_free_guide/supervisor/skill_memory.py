"""Persistent markdown skill memory for supervisor knowledge distillation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


DEFAULT_SKILL_PATH = "class_free_guide/supervisor/skill/SKILL.md"


def default_skill_template() -> str:
    return """---
name: supervisor-rl-knowledge
description: Persistent distilled knowledge used by the RL reward supervisor across training runs.
---

# Supervisor RL Knowledge

## Purpose
Capture reusable lessons from previous Unitree-Go2 FPO reward-supervision cycles so future supervisor analyses can start from accumulated training experience.

## Stable Patterns
- No distilled patterns yet.

## Reward Tuning Heuristics
- No distilled heuristics yet.

## Failure Modes
- No distilled failure modes yet.

## Evidence Log
- No training-cycle evidence has been distilled yet.

## Open Questions
- Which metrics most reliably predict stable Go2 locomotion for this task?
"""


class SkillMemory:
    """Owns the supervisor SKILL.md lifecycle and update gate."""

    def __init__(
        self,
        path: str | Path = DEFAULT_SKILL_PATH,
        *,
        enabled: bool = True,
        min_update_iter: int = 1500,
        max_chars: int = 12000,
    ):
        self.path = Path(path)
        self.enabled = enabled
        self.min_update_iter = int(min_update_iter)
        self.max_chars = int(max_chars)
        if self.enabled:
            self.ensure_exists()

    def ensure_exists(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._atomic_write(default_skill_template())

    def read(self) -> str:
        if not self.enabled:
            return ""
        self.ensure_exists()
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return ""
        return text[: self.max_chars]

    def should_update(self, current_iter: int) -> bool:
        return self.enabled and int(current_iter) >= self.min_update_iter

    def write(self, text: str) -> None:
        if not self.enabled:
            return
        cleaned = self._clean_markdown(text)
        if not cleaned:
            return
        if not cleaned.lstrip().startswith("---"):
            header = (
                "---\n"
                "name: supervisor-rl-knowledge\n"
                "description: Persistent distilled knowledge used by the RL reward supervisor across training runs.\n"
                "---\n\n"
            )
            cleaned = header + cleaned
        self._atomic_write(cleaned)

    def update_with_llm(
        self,
        llm: Any,
        *,
        current_iter: int,
        current_skill: str,
        cycle: dict[str, Any],
    ) -> str | None:
        if not self.should_update(current_iter):
            return None
        distill = getattr(llm, "distill_skill", None)
        if distill is None:
            return None
        new_skill = distill(current_skill=current_skill, cycle=cycle)
        if not isinstance(new_skill, str) or not new_skill.strip():
            return None
        self.write(new_skill)
        return self.read()

    def _atomic_write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = text.rstrip() + "\n"
        tmp = self.path.with_suffix(f".md.tmp.{os.getpid()}.{int(time.time() * 1000)}")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _clean_markdown(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned
