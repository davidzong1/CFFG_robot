"""Prompt templates and rendering helpers for the supervisor LLM calls."""

from __future__ import annotations

import json
from typing import Any

DIAGNOSE_SYSTEM = """You are an expert reinforcement learning engineer monitoring a
locomotion training run (Unitree-Go2 velocity tracking with Flow Policy Optimization).

The user supplies a TRAINING OBJECTIVE block at the top of every message. It
is the highest-priority signal: every observation and hypothesis must be
judged with reference to that objective, not to generic "good RL" intuitions.

You will receive:
  - A compact summary of recent TensorBoard scalars (per-term rewards,
    loss, policy stats, episode length, etc).
  - A handful of frames from recent rollout videos.
  - The currently active reward weights with their allowed bounds.

Your job is to DIAGNOSE — describe what is happening, identify failure
modes (reward hacking, instability, getting stuck, slow convergence) and
state which reward weights look mis-tuned.

You MUST reply with a single JSON object matching this shape:

{
  "observations": ["..."],   // 2-6 concrete observations from data/video
  "hypotheses":   ["..."],   // 1-4 hypotheses about why training is suboptimal
  "confidence":   0.0,       // float in [0, 1]
  "score":        0          // integer 0-100: how well the policy currently meets the
                             // TRAINING OBJECTIVE. 0 = completely untrained,
                             // 100 = perfectly satisfies the objective. Consider all
                             // priorities and avoid-list items from the objective
                             // block. Be honest and conservative — never inflate.
}

Do not write anything outside the JSON object.
"""


PROPOSE_SYSTEM = """You are the same RL engineer from the previous step.

The TRAINING OBJECTIVE block from the prior message still applies. Every
weight change must move the policy toward that objective, not toward a
generic notion of "better training".

You will receive your prior diagnosis, the schema of allowed reward
weights (with min/max), and the current weights.

Produce a SMALL, SAFE patch:
  - Modify at most 3 weights.
  - Keep each new value inside [min, max] from the schema.
  - Single-step relative change ≤ 30% per weight.
  - Only act when you have a clear, evidence-backed rationale; otherwise
    return an empty patch.

You MUST reply with a single JSON object:

{
  "rationale":       "short paragraph",
  "patch":           {"term_name": new_weight_float, ...},
  "expected_effect": "one sentence on what should improve",
  "rollback_if":     "metric_path op threshold  (e.g. 'Train/mean_reward < 0.3')"
}

`rollback_if` uses a simple DSL: `<metric_tag> <op> <number>` where op is
one of `<`, `<=`, `>`, `>=`. The supervisor will trigger a rollback the
next cycle if the condition becomes true.

Return `"patch": {}` to do nothing this cycle. Do not write anything
outside the JSON object.
"""



DISTILL_SKILL_SYSTEM = """You maintain a persistent markdown SKILL.md knowledge base for an RL reward supervisor.

You will receive the current SKILL.md and evidence from the latest supervisor cycle.
Update the skill by distilling reusable, evidence-backed lessons about Unitree-Go2 FPO reward tuning.

Rules:
  - Return the complete new SKILL.md content, not a diff.
  - Keep valid SKILL.md frontmatter with name and description.
  - Preserve useful prior lessons; merge duplicates.
  - Prefer concise, generalizable rules over run-specific narration.
  - Include evidence with iteration numbers and whether patches were applied or rejected.
  - Do not invent metrics, outcomes, or reward terms that are not present in the evidence.
  - Keep the file under about 250 lines.

Return only markdown.
"""

def render_state(
    snapshot_summary: dict[str, Any],
    current_weights: dict[str, float],
    schema_summary: dict[str, dict[str, float]] | None = None,
    objective_block: str | None = None,
    skill_block: str | None = None,
) -> str:
    parts: list[str] = []
    if objective_block:
        parts += [objective_block, ""]
    if skill_block:
        parts += [
            "## Persistent Supervisor Skill Knowledge",
            "```markdown",
            skill_block[:12000],
            "```",
            "",
        ]
    parts += [
        "## TensorBoard snapshot",
        "```json",
        json.dumps(snapshot_summary, indent=2, default=_json_default)[:12000],
        "```",
        "",
        "## Current reward weights",
        "```json",
        json.dumps(current_weights, indent=2),
        "```",
    ]
    if schema_summary is not None:
        parts += [
            "",
            "## Allowed bounds",
            "```json",
            json.dumps(schema_summary, indent=2),
            "```",
        ]
    return "\n".join(parts)


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
