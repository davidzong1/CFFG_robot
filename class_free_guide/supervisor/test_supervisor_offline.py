"""Offline smoke test for the supervisor: no GPU, no network, no LLM call.

Validates that one supervisor cycle:
  - reads a hand-crafted metric snapshot
  - asks a fake LLM client for a patch
  - writes a versioned yaml under ``log_dir/reward_cfg/``
  - mutates the live "reward manager" weights
  - records the action in ``audit.jsonl``

Run with::

    conda run -n cffg pytest class_free_guide/pineline/rl/supervisor/test_supervisor_offline.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from class_free_guide.supervisor.supervisor import (
    Supervisor,
    SupervisorCallbacks,
    SupervisorConfig,
)
from class_free_guide.supervisor.llm_client import LLMClient
from class_free_guide.supervisor.metric_collector import (
    MetricSnapshot,
    ScalarSeries,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTermCfg:
    def __init__(self, weight: float):
        self.weight = weight


class FakeRewardManager:
    def __init__(self, weights: dict[str, float]):
        self._terms = {k: FakeTermCfg(v) for k, v in weights.items()}

    @property
    def active_terms(self):
        return list(self._terms.keys())

    def get_term_cfg(self, name: str) -> FakeTermCfg:
        return self._terms[name]


class FakeEnv:
    def __init__(self, weights: dict[str, float]):
        self.reward_manager = FakeRewardManager(weights)
        self.common_step_counter = 0

    @property
    def unwrapped(self):
        return self


class ScriptedClient(LLMClient):
    """Returns a hard-coded diagnose + proposal; records what it was called with."""

    def __init__(self, patch: dict[str, float]):
        self._patch = patch
        self.last_objective_block: str | None = None
        self.last_skill_block: str | None = None

    def diagnose(self, snapshot_summary, current_weights, frames, objective_block=None, skill_block=None):
        self.last_objective_block = objective_block
        self.last_skill_block = skill_block
        return {
            "observations": ["test"],
            "hypotheses": ["test"],
            "confidence": 0.9,
        }

    def propose(self, diagnose_result, schema_summary, current_weights, objective_block=None, skill_block=None):
        self.last_objective_block = objective_block
        self.last_skill_block = skill_block
        return {
            "rationale": "scripted test patch",
            "patch": self._patch,
            "expected_effect": "n/a",
            "rollback_if": "",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_snapshot() -> MetricSnapshot:
    s = ScalarSeries(name="Train/mean_reward")
    s.steps = [10, 20, 30]
    s.values = [0.1, 0.2, 0.3]
    return MetricSnapshot(step=30, series={"Train/mean_reward": s})


def _make_supervisor(tmp_path: Path, patch: dict[str, float]):
    """Factory using from_rl_env (backward-compat RL path)."""
    weights = {
        "track_linear_velocity": 1.0,
        "action_rate_l2": -0.05,
        "foot_slip": -0.25,
    }
    env = FakeEnv(weights)
    cfg = SupervisorConfig(
        interval_min=60.0,
        cooldown_iters=0,
        warmup_iters=0,
        max_patch_fields=3,
        max_rel_change=0.30,
        clips_per_cycle=0,
        video_frames_per_clip=0,
        provider="stub",
    )
    sup = Supervisor.from_rl_env(
        env=env,
        writer=None,
        log_dir=tmp_path,
        config=cfg,
        llm=ScriptedClient(patch),
        iter_getter=lambda: 100,
    )
    return sup, env


def _make_supervisor_for_early_stop(
    tmp_path: Path,
    patch: dict[str, float],
    score: int,
    *,
    early_stopping: bool = True,
    pass_score: float = 75.0,
    min_training_iters: int = 50,
    current_iter: int = 100,
):
    """Factory for early-stop test supervisors with a score-bearing LLM client."""
    weights = {
        "track_linear_velocity": 1.0,
        "action_rate_l2": -0.05,
        "foot_slip": -0.25,
    }
    env = FakeEnv(weights)

    class ScoredClient(ScriptedClient):
        def diagnose(self, *args, **kwargs):
            result = super().diagnose(*args, **kwargs)
            result["score"] = score
            return result

    cfg = SupervisorConfig(
        interval_min=60.0,
        cooldown_iters=0,
        warmup_iters=0,
        max_patch_fields=3,
        max_rel_change=0.30,
        clips_per_cycle=0,
        video_frames_per_clip=0,
        provider="stub",
        early_stopping=early_stopping,
        pass_score=pass_score,
        min_training_iters=min_training_iters,
    )
    stopping_event = threading.Event()
    sup = Supervisor.from_rl_env(
        env=env,
        writer=None,
        log_dir=tmp_path,
        config=cfg,
        llm=ScoredClient(patch),
        iter_getter=lambda: current_iter,
        stopping_event=stopping_event,
    )
    return sup, env, stopping_event


# ---------------------------------------------------------------------------
# Tests — RL backward-compat (via from_rl_env)
# ---------------------------------------------------------------------------


def test_one_cycle_applies_patch(tmp_path, monkeypatch):
    patch = {"action_rate_l2": -0.04}  # +20% relative change, within bounds
    sup, env = _make_supervisor(tmp_path, patch)

    # Bypass disk-based collector / video-sampler via callback monkeypatch.
    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])

    # Write v00 baseline (normally done by start()).
    sup.patcher.write_initial_snapshot(sup._current_weights())

    sup._run_cycle()

    # Weight mutated on the fake reward manager.
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(-0.04)
    # v01.yaml written on disk and contains the new value.
    v1 = tmp_path / "reward_cfg" / "v01.yaml"
    assert v1.exists(), "expected v01.yaml after one applied patch"
    import yaml

    data = yaml.safe_load(v1.read_text())
    assert data["weights"]["action_rate_l2"] == pytest.approx(-0.04)
    # All known terms persisted (not just the diff).
    assert {"track_linear_velocity", "foot_slip"} <= set(data["weights"].keys())

    # Audit log records the apply.
    audit = (tmp_path / "supervisor" / "audit.jsonl").read_text().strip().split("\n\n")
    kinds = [json.loads(block)["kind"] for block in audit]
    assert "applied" in kinds


def test_out_of_bounds_patch_rejected(tmp_path, monkeypatch):
    # action_rate_l2 bounds are [-0.5, 0.0]; relative change limit also 30%.
    # -0.20 is a 300% change from -0.05 → guardrail rejects it.
    patch = {"action_rate_l2": -0.20}
    sup, env = _make_supervisor(tmp_path, patch)

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())

    sup._run_cycle()

    # Weight unchanged.
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(-0.05)
    # No v01.yaml.
    assert not (tmp_path / "reward_cfg" / "v01.yaml").exists()
    # audit has a rejected entry.
    audit = (tmp_path / "supervisor" / "audit.jsonl").read_text().strip().split("\n\n")
    assert any(json.loads(block)["kind"] == "rejected" for block in audit)


def test_killswitch_pauses_cycle(tmp_path, monkeypatch):
    patch = {"action_rate_l2": -0.04}
    sup, env = _make_supervisor(tmp_path, patch)

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())

    (tmp_path / "supervisor").mkdir(exist_ok=True)
    (tmp_path / "supervisor" / "PAUSE").touch()

    sup._run_cycle()

    # No apply, no v01.yaml.
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(-0.05)
    assert not (tmp_path / "reward_cfg" / "v01.yaml").exists()
    audit = (tmp_path / "supervisor" / "audit.jsonl").read_text().strip().split("\n\n")
    assert any(json.loads(block)["kind"] == "paused" for block in audit)


def test_skill_memory_created_and_passed_to_llm(tmp_path, monkeypatch):
    patch = {"action_rate_l2": -0.04}
    sup, _env = _make_supervisor(tmp_path, patch)

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    skill_path = Path("class_free_guide/supervisor/skill/SKILL.md")
    assert skill_path.exists()
    assert "Supervisor RL Knowledge" in skill_path.read_text()
    assert isinstance(sup.llm.last_skill_block, str)
    assert "Supervisor RL Knowledge" in sup.llm.last_skill_block


def test_skill_memory_not_updated_before_min_iter(tmp_path, monkeypatch):
    class DistillingClient(ScriptedClient):
        def distill_skill(self, current_skill, cycle):
            return current_skill + "\n## Should Not Appear\n"

    weights = {"track_linear_velocity": 1.0, "action_rate_l2": -0.05, "foot_slip": -0.25}
    env = FakeEnv(weights)
    skill_path = tmp_path / "skill" / "SKILL.md"
    cfg = SupervisorConfig(
        interval_min=60.0,
        cooldown_iters=0,
        warmup_iters=0,
        max_patch_fields=3,
        max_rel_change=0.30,
        provider="stub",
        skill_memory_path=str(skill_path),
        skill_memory_min_update_iter=1500,
    )
    sup = Supervisor.from_rl_env(
        env=env,
        writer=None,
        log_dir=tmp_path,
        config=cfg,
        llm=DistillingClient({"action_rate_l2": -0.04}),
        iter_getter=lambda: 1499,
    )
    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    before = skill_path.read_text()
    sup._run_cycle()

    assert skill_path.read_text() == before


def test_skill_memory_updated_at_min_iter(tmp_path, monkeypatch):
    class DistillingClient(ScriptedClient):
        def distill_skill(self, current_skill, cycle):
            assert cycle["iter"] == 1500
            assert cycle["outcome"]["kind"] == "applied"
            return current_skill + "\n## Distilled Lesson\n- Iter 1500: action_rate_l2 patch was applied.\n"

    weights = {"track_linear_velocity": 1.0, "action_rate_l2": -0.05, "foot_slip": -0.25}
    env = FakeEnv(weights)
    skill_path = tmp_path / "skill" / "SKILL.md"
    cfg = SupervisorConfig(
        interval_min=60.0,
        cooldown_iters=0,
        warmup_iters=0,
        max_patch_fields=3,
        max_rel_change=0.30,
        provider="stub",
        skill_memory_path=str(skill_path),
        skill_memory_min_update_iter=1500,
    )
    sup = Supervisor.from_rl_env(
        env=env,
        writer=None,
        log_dir=tmp_path,
        config=cfg,
        llm=DistillingClient({"action_rate_l2": -0.04}),
        iter_getter=lambda: 1500,
    )
    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    text = skill_path.read_text()
    assert "Distilled Lesson" in text
    audit = (tmp_path / "supervisor" / "audit.jsonl").read_text().strip().split("\n\n")
    assert any(json.loads(block)["kind"] == "skill_distilled" for block in audit)


def test_objective_loaded_and_passed_to_llm(tmp_path, monkeypatch):
    """A JSON objective on disk must reach the LLM call and the audit log."""
    obj_path = tmp_path / "obj.json"
    obj_path.write_text(
        json.dumps(
            {
                "name": "stable_walk",
                "summary": "Stable upright trot.",
                "priorities": ["upright body", "clean foot contacts"],
                "avoid": ["body roll spikes"],
                "notes": "stability over speed",
            }
        )
    )

    patch = {"action_rate_l2": -0.04}
    weights = {
        "track_linear_velocity": 1.0,
        "action_rate_l2": -0.05,
        "foot_slip": -0.25,
    }
    env = FakeEnv(weights)
    cfg = SupervisorConfig(
        interval_min=60.0,
        cooldown_iters=0,
        warmup_iters=0,
        clips_per_cycle=0,
        video_frames_per_clip=0,
        provider="stub",
        objective_path=str(obj_path),
    )
    client = ScriptedClient(patch)
    sup = Supervisor.from_rl_env(
        env=env,
        writer=None,
        log_dir=tmp_path,
        config=cfg,
        llm=client,
        iter_getter=lambda: 100,
    )

    # Objective loaded into the supervisor.
    assert sup.objective.name == "stable_walk"
    assert "upright body" in sup.objective.priorities

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    # The objective block was rendered and passed to the LLM client.
    assert client.last_objective_block is not None
    assert "stable_walk" in client.last_objective_block
    assert "upright body" in client.last_objective_block
    assert "body roll spikes" in client.last_objective_block


# ---------------------------------------------------------------------------
# OpenRouter client tests
# ---------------------------------------------------------------------------


def test_openrouter_requires_api_base(monkeypatch):
    """provider=openrouter without api_base must raise with a helpful message."""
    from class_free_guide.supervisor.llm_client import build_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    cfg = SupervisorConfig(provider="openrouter", api_base=None)
    with pytest.raises(RuntimeError, match="requires api_base"):
        build_client(cfg)


def test_openrouter_reads_base_url_from_config(monkeypatch):
    """api_base in config must become the OpenAI client's base_url."""
    from class_free_guide.supervisor.llm_client import OpenRouterClient, build_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    cfg = SupervisorConfig(
        provider="openrouter",
        api_base="https://openrouter.ai/api/v1",
    )
    client = build_client(cfg)
    assert isinstance(client, OpenRouterClient)
    assert "openrouter.ai" in str(client._client.base_url)


def test_openrouter_supports_any_router(monkeypatch):
    """Any OpenAI-compatible base_url must work (Groq, Fireworks, vLLM...)."""
    from class_free_guide.supervisor.llm_client import OpenRouterClient, build_client

    monkeypatch.setenv("GROQ_API_KEY", "sk-groq-test")
    cfg = SupervisorConfig(
        provider="openrouter",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    )
    client = build_client(cfg)
    assert isinstance(client, OpenRouterClient)
    assert "groq.com" in str(client._client.base_url)


def test_openrouter_extra_headers_forwarded(monkeypatch):
    """extra_headers dict must reach the OpenAI client."""
    from class_free_guide.supervisor.llm_client import OpenRouterClient, build_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    cfg = SupervisorConfig(
        provider="openrouter",
        api_base="https://openrouter.ai/api/v1",
        extra_headers={
            "HTTP-Referer": "https://example.com/repo",
            "X-Title": "Go2 Supervisor",
        },
    )
    client = build_client(cfg)
    assert isinstance(client, OpenRouterClient)

    headers = getattr(client._client, "default_headers", {}) or {}
    assert headers.get("HTTP-Referer") == "https://example.com/repo"
    assert headers.get("X-Title") == "Go2 Supervisor"


def test_openrouter_missing_key_raises(monkeypatch):
    """OpenRouter must complain when no API key env var is set."""
    from class_free_guide.supervisor.llm_client import build_client

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = SupervisorConfig(
        provider="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="NONEXISTENT_KEY",
    )
    with pytest.raises(RuntimeError, match="No API key found"):
        build_client(cfg)


def test_openai_thinking_level_sets_reasoning_effort(monkeypatch):
    """OpenAI-compatible calls should translate thinking_level to reasoning_effort."""
    from class_free_guide.supervisor.llm_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    client = OpenAIClient(SupervisorConfig(provider="openai", thinking_level="medium"))
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    assert client._call("system", [{"type": "text", "text": "state"}]) == '{"ok": true}'
    assert captured["reasoning_effort"] == "medium"


def test_openrouter_thinking_level_sets_reasoning_body(monkeypatch):
    """OpenRouter calls should translate thinking_level to extra_body.reasoning."""
    from class_free_guide.supervisor.llm_client import OpenRouterClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    client = OpenRouterClient(
        SupervisorConfig(
            provider="openrouter",
            api_base="https://openrouter.ai/api/v1",
            thinking_level="high",
        )
    )
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    assert client._call("system", [{"type": "text", "text": "state"}]) == '{"ok": true}'
    assert captured["extra_body"] == {
        "reasoning": {"effort": "high", "exclude": False}
    }


def test_unknown_thinking_level_raises(monkeypatch):
    """Invalid thinking_level values should fail before a network request."""
    from class_free_guide.supervisor.llm_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    with pytest.raises(ValueError, match="Unsupported thinking_level"):
        OpenAIClient(SupervisorConfig(provider="openai", thinking_level="extreme"))


# ---------------------------------------------------------------------------
# Early stopping tests (via from_rl_env)
# ---------------------------------------------------------------------------


def test_early_stop_triggers_when_score_high(tmp_path, monkeypatch):
    """When score >= pass_score and iter >= min_training_iters, event is set."""
    patch = {"action_rate_l2": -0.04}
    sup, _env, stopping_event = _make_supervisor_for_early_stop(
        tmp_path,
        patch,
        score=85,
        pass_score=75.0,
        min_training_iters=50,
        current_iter=100,
    )

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    assert stopping_event.is_set()
    assert (tmp_path / "supervisor" / "EARLY_STOP").exists()
    audit = (tmp_path / "supervisor" / "audit.jsonl").read_text().strip().split("\n\n")
    assert any(json.loads(block)["kind"] == "early_stop" for block in audit)


def test_early_stop_blocked_below_min_iters(tmp_path, monkeypatch):
    """Early stop does NOT trigger when current_iter < min_training_iters."""
    patch = {"action_rate_l2": -0.04}
    sup, _env, stopping_event = _make_supervisor_for_early_stop(
        tmp_path,
        patch,
        score=85,
        pass_score=75.0,
        min_training_iters=500,
        current_iter=100,
    )

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    assert not stopping_event.is_set()
    assert not (tmp_path / "supervisor" / "EARLY_STOP").exists()


def test_early_stop_blocked_by_low_score(tmp_path, monkeypatch):
    """Early stop does NOT trigger when score < pass_score."""
    patch = {"action_rate_l2": -0.04}
    sup, _env, stopping_event = _make_supervisor_for_early_stop(
        tmp_path,
        patch,
        score=30,
        pass_score=75.0,
        min_training_iters=50,
        current_iter=500,
    )

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    assert not stopping_event.is_set()
    assert not (tmp_path / "supervisor" / "EARLY_STOP").exists()


def test_early_stop_disabled_by_config(tmp_path, monkeypatch):
    """When early_stopping=False, no early stop even with high score."""
    patch = {"action_rate_l2": -0.04}
    sup, _env, stopping_event = _make_supervisor_for_early_stop(
        tmp_path,
        patch,
        score=90,
        early_stopping=False,
        pass_score=75.0,
        min_training_iters=50,
        current_iter=500,
    )

    monkeypatch.setattr(sup._callbacks, "metrics_getter", _seed_snapshot)
    monkeypatch.setattr(sup._callbacks, "frames_getter", lambda overlay=None: [])
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    assert not stopping_event.is_set()


# ---------------------------------------------------------------------------
# Framework-agnostic callback constructor test
# ---------------------------------------------------------------------------


def test_callback_constructor_without_env(tmp_path):
    """Supervisor works with raw callbacks — no env, no RL machinery."""
    params = {
        "track_linear_velocity": 1.0,
        "action_rate_l2": -0.05,
        "foot_slip": -0.25,
    }

    callbacks = SupervisorCallbacks(
        metrics_getter=_seed_snapshot,
        frames_getter=lambda overlay=None: [],
        params_getter=lambda: dict(params),
        param_setter=lambda n, v: params.__setitem__(n, v),
        known_params_getter=lambda: list(params.keys()),
    )

    cfg = SupervisorConfig(
        interval_min=60.0,
        cooldown_iters=0,
        warmup_iters=0,
        max_patch_fields=3,
        max_rel_change=0.30,
        provider="stub",
    )
    sup = Supervisor(
        log_dir=tmp_path,
        config=cfg,
        callbacks=callbacks,
        llm=ScriptedClient({"action_rate_l2": -0.04}),
        iter_getter=lambda: 100,
    )
    sup.patcher.write_initial_snapshot(sup._current_weights())
    sup._run_cycle()

    # Patch applied to the plain dict (no reward_manager involved).
    assert params["action_rate_l2"] == pytest.approx(-0.04)

    v1 = tmp_path / "reward_cfg" / "v01.yaml"
    assert v1.exists()

    audit = (tmp_path / "supervisor" / "audit.jsonl").read_text().strip().split("\n\n")
    assert any(json.loads(block)["kind"] == "applied" for block in audit)
