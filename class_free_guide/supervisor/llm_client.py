"""Pluggable LLM clients.

Two concrete backends:
  * ``ClaudeClient`` — uses the official ``anthropic`` SDK with prompt
    caching on the stable system prompt + image attachments.
  * ``OpenAIClient`` — talks to any OpenAI-compatible endpoint via
    ``openai`` SDK; uses JSON-mode response format.

For testing/offline use, ``StubClient`` returns canned responses.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from .config import SupervisorConfig
from .prompts import DIAGNOSE_SYSTEM, DISTILL_SKILL_SYSTEM, PROPOSE_SYSTEM, render_state


_THINKING_BUDGET: dict[str, int] = {
    "xhigh": 16000,
    "high": 8192,
    "medium": 4096,
    "small": 1024,
}

_OPENAI_REASONING_EFFORT: dict[str, str] = {
    "xhigh": "high",
    "high": "high",
    "medium": "medium",
    "small": "low",
}

_OPENROUTER_REASONING_EFFORT: dict[str, str] = {
    "xhigh": "xhigh",
    "high": "high",
    "medium": "medium",
    "small": "low",
}


def _thinking_level(cfg: SupervisorConfig) -> str | None:
    level = getattr(cfg, "thinking_level", None)
    if level is None:
        return None
    level = str(level).strip().lower()
    if not level:
        return None
    if level not in _THINKING_BUDGET:
        allowed = ", ".join(_THINKING_BUDGET)
        raise ValueError(f"Unsupported thinking_level={level!r}. Expected one of: {allowed}")
    return level


class LLMClient(ABC):
    """Two-call interface used by the supervisor."""

    @abstractmethod
    def diagnose(
        self,
        snapshot_summary: dict[str, Any],
        current_weights: dict[str, float],
        frames: list[tuple[str, bytes]],
        objective_block: str | None = None,
        skill_block: str | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def propose(
        self,
        diagnose_result: dict[str, Any],
        schema_summary: dict[str, dict[str, float]],
        current_weights: dict[str, float],
        patch_limits: dict[str, Any] | None = None,
        objective_block: str | None = None,
        skill_block: str | None = None,
    ) -> dict[str, Any]: ...

    def distill_skill(self, current_skill: str, cycle: dict[str, Any]) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support skill distillation")

    def test_connection(self, timeout: float = 30.0) -> bool:
        """Send a minimal request to verify the LLM is reachable.

        Returns True if the API responds within *timeout* seconds.
        The default implementation always returns True (used by StubClient).
        Subclasses SHOULD override this to perform an actual API call.
        """
        return True


def build_client(cfg: SupervisorConfig) -> LLMClient:
    provider = cfg.provider.lower()
    if provider == "anthropic":
        return ClaudeClient(cfg)
    if provider == "openai":
        return OpenAIClient(cfg)
    if provider == "openrouter":
        return OpenRouterClient(cfg)
    if provider == "stub":
        return StubClient(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


def _resolve_api_key(cfg: SupervisorConfig, *fallback_envs: str) -> str:
    """Resolve API key: cfg.api_key (inline) > cfg.api_key_env (env var) > fallbacks."""
    if cfg.api_key:
        return cfg.api_key
    for env_name in (cfg.api_key_env, *fallback_envs):
        key = os.environ.get(env_name)
        if key:
            return key
    tried = ["cfg.api_key"] + [cfg.api_key_env] + list(fallback_envs)
    raise RuntimeError(f"No API key found. Tried: {', '.join(tried)}")


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------


class ClaudeClient(LLMClient):
    def __init__(self, cfg: SupervisorConfig):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "anthropic SDK not installed; pip install anthropic"
            ) from e
        api_key = _resolve_api_key(cfg)
        self._client = anthropic.Anthropic(api_key=api_key)
        self.cfg = cfg

        # ---- validate thinking_level configuration ---------------------------
        thinking_level = _thinking_level(cfg)
        if thinking_level:
            budget = _THINKING_BUDGET[thinking_level]
            # 1. max_tokens must exceed the thinking budget (Anthropic hard requirement).
            if cfg.max_tokens <= budget:
                raise ValueError(
                    f"thinking_level={thinking_level} requires budget_tokens={budget}, "
                    f"but max_tokens={cfg.max_tokens}. "
                    f"Set max_tokens > {budget} in supervisor.yaml "
                    f"(recommended: max_tokens={budget + 4096} or higher)."
                )
            # 2. Check that the installed anthropic SDK supports extended thinking.
            min_version = (0, 39, 0)
            current = tuple(int(x) for x in anthropic.__version__.split(".")[:3])
            if current < min_version:
                raise RuntimeError(
                    f"Anthropic SDK {anthropic.__version__} does not support extended thinking. "
                    f"Upgrade to >=0.39.0: pip install 'anthropic>=0.39.0'"
                )
            # 3. Log the thinking configuration so it shows up in training output.
            logging.getLogger(__name__).info(
                "[supervisor] extended thinking enabled: level=%s budget=%d max_tokens=%d sdk=%s",
                thinking_level,
                budget,
                cfg.max_tokens,
                anthropic.__version__,
            )

    def _call(self, system: str, user_blocks: list[dict[str, Any]]) -> str:
        # ``cache_control`` on the system prompt — it is stable across cycles
        # and large enough to benefit from caching.
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": user_blocks}],
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )

        thinking_level = _thinking_level(self.cfg)
        if thinking_level:
            budget = _THINKING_BUDGET[thinking_level]
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["temperature"] = 1.0  # API requires temperature=1 when extended thinking is enabled
        else:
            kwargs["temperature"] = self.cfg.temperature

        resp = self._client.messages.create(**kwargs)
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    def diagnose(
        self,
        snapshot_summary: dict[str, Any],
        current_weights: dict[str, float],
        frames: list[tuple[str, bytes]],
        objective_block: str | None = None,
        skill_block: str | None = None,
    ) -> dict[str, Any]:
        user: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": render_state(
                    snapshot_summary,
                    current_weights,
                    objective_block=objective_block,
                    skill_block=skill_block,
                ),
            }
        ]
        if not self.cfg.only_text:
            for label, png in frames:
                user.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(png).decode("ascii"),
                        },
                    }
                )
                user.append({"type": "text", "text": f"(frame: {label})"})
        text = self._call(DIAGNOSE_SYSTEM, user)
        return _parse_json(text)

    def propose(
        self,
        diagnose_result: dict[str, Any],
        schema_summary: dict[str, dict[str, float]],
        current_weights: dict[str, float],
        patch_limits: dict[str, Any] | None = None,
        objective_block: str | None = None,
        skill_block: str | None = None,
    ) -> dict[str, Any]:
        body = (
            "## Prior diagnosis\n```json\n"
            + json.dumps(diagnose_result, indent=2)
            + "\n```\n\n"
            + render_state(
                {},
                current_weights,
                schema_summary=schema_summary,
                patch_limits=patch_limits,
                objective_block=objective_block,
                skill_block=skill_block,
            )
        )
        text = self._call(PROPOSE_SYSTEM, [{"type": "text", "text": body}])
        return _parse_json(text)

    def distill_skill(self, current_skill: str, cycle: dict[str, Any]) -> str:
        body = _render_skill_distill_body(
            current_skill,
            cycle,
            max_chars=self.cfg.skill_memory_max_chars,
        )
        return self._call(DISTILL_SKILL_SYSTEM, [{"type": "text", "text": body}])

    def test_connection(self, timeout: float = 30.0) -> bool:
        """Send a minimal request to verify the Anthropic API is reachable.

        Returns True on success, False on any error (timeout, auth, network, etc.).
        The caller is responsible for deciding how to handle the failure.
        """
        log = logging.getLogger(__name__)
        try:
            self._client.messages.create(
                model=self.cfg.model,
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": "Reply with: OK"}],
                timeout=timeout,
            )
            return True
        except Exception as e:
            log.error("[supervisor] LLM connectivity check failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    def __init__(self, cfg: SupervisorConfig):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai SDK not installed; pip install openai") from e
        api_key = _resolve_api_key(cfg, "OPENAI_API_KEY")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if cfg.api_base:
            kwargs["base_url"] = cfg.api_base
        if cfg.extra_headers:
            kwargs["default_headers"] = dict(cfg.extra_headers)
        self._client = openai.OpenAI(**kwargs)
        self.cfg = cfg
        self._reasoning_effort = self._build_reasoning_effort(cfg)

    def _build_reasoning_effort(self, cfg: SupervisorConfig) -> str | None:
        level = _thinking_level(cfg)
        if level is None:
            return None
        effort = _OPENAI_REASONING_EFFORT[level]
        logging.getLogger(__name__).info(
            "[supervisor] reasoning effort enabled: provider=%s level=%s effort=%s",
            cfg.provider,
            level,
            effort,
        )
        return effort

    def _call(self, system: str, user_content: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def diagnose(
        self,
        snapshot_summary: dict[str, Any],
        current_weights: dict[str, float],
        frames: list[tuple[str, bytes]],
        objective_block: str | None = None,
        skill_block: str | None = None,
    ) -> dict[str, Any]:
        user: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": render_state(
                    snapshot_summary,
                    current_weights,
                    objective_block=objective_block,
                    skill_block=skill_block,
                ),
            }
        ]
        if not self.cfg.only_text:
            for label, png in frames:
                user.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(png).decode("ascii"),
                        },
                    }
                )
                user.append({"type": "text", "text": f"(frame: {label})"})
        try:
            return _parse_json(self._call(DIAGNOSE_SYSTEM, user))
        except Exception as exc:
            err_msg = str(exc).lower()
            # Detect APIs that reject multimodal image_url blocks.
            # Only fatal when only_text is off (user asked for vision).
            if not self.cfg.only_text and "image_url" in err_msg and ("unknown variant" in err_msg or "expected" in err_msg):
                raise RuntimeError(
                    f"Supervisor requires a vision-capable model, but "
                    f"'{self.cfg.model}' does not accept image inputs. "
                    f"The API returned: {exc}. "
                    f"Switch to a multimodal model (e.g. gpt-4o, claude-opus-4-7, "
                    f"or any model that supports image_url content blocks) "
                    f"and restart training."
                ) from exc
            raise

    def propose(
        self,
        diagnose_result: dict[str, Any],
        schema_summary: dict[str, dict[str, float]],
        current_weights: dict[str, float],
        patch_limits: dict[str, Any] | None = None,
        objective_block: str | None = None,
        skill_block: str | None = None,
    ) -> dict[str, Any]:
        body = (
            "## Prior diagnosis\n```json\n"
            + json.dumps(diagnose_result, indent=2)
            + "\n```\n\n"
            + render_state(
                {},
                current_weights,
                schema_summary=schema_summary,
                patch_limits=patch_limits,
                objective_block=objective_block,
                skill_block=skill_block,
            )
        )
        return _parse_json(self._call(PROPOSE_SYSTEM, [{"type": "text", "text": body}]))

    def distill_skill(self, current_skill: str, cycle: dict[str, Any]) -> str:
        body = _render_skill_distill_body(
            current_skill,
            cycle,
            max_chars=self.cfg.skill_memory_max_chars,
        )
        return self._call(DISTILL_SKILL_SYSTEM, [{"type": "text", "text": body}])

    def test_connection(self, timeout: float = 30.0) -> bool:
        """Send a minimal request to verify the API is reachable.

        When ``only_text`` is False (default) the request includes a tiny PNG
        to confirm the model supports vision.  A non-vision model causes a
        ``RuntimeError`` so training can abort before wasting GPU hours.

        When ``only_text`` is True the request is plain text — any model works.

        Returns True on success.
        Raises RuntimeError when the API rejects ``image_url`` blocks (non-
        vision model, only_text=False).
        Returns False on other errors (timeout, auth, network, etc.).
        """
        log = logging.getLogger(__name__)
        if self.cfg.only_text:
            # Plain-text probe — any model suffices.
            try:
                self._client.chat.completions.create(
                    model=self.cfg.model,
                    max_tokens=10,
                    temperature=0,
                    messages=[{"role": "user", "content": "Reply with: OK"}],
                    timeout=timeout,
                )
                return True
            except Exception as e:
                log.error("[supervisor] LLM connectivity check failed: %s", e)
                return False

        # Multimodal probe — must accept image_url blocks.
        # 1×1 red pixel PNG (valid, minimal, ~68 bytes).
        DUMMY_PNG_B64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        try:
            self._client.chat.completions.create(
                model=self.cfg.model,
                max_tokens=10,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Reply with: OK"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + DUMMY_PNG_B64},
                            },
                        ],
                    }
                ],
                timeout=timeout,
            )
            return True
        except Exception as e:
            err_msg = str(e).lower()
            if "image_url" in err_msg and (
                "unknown variant" in err_msg or "expected" in err_msg
            ):
                raise RuntimeError(
                    f"Supervisor requires a vision-capable model, but "
                    f"'{self.cfg.model}' does not accept image inputs. "
                    f"The API returned: {e}. "
                    f"Switch to a multimodal model (e.g. gpt-4o, claude-opus-4-7, "
                    f"or any model that supports image_url content blocks) "
                    f"and restart training."
                ) from e
            log.error("[supervisor] LLM connectivity check failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# OpenRouter / any OpenAI-compatible router
# ---------------------------------------------------------------------------


class OpenRouterClient(OpenAIClient):
    """Generic OpenAI-compatible router client.

    Works with **any** provider that speaks the OpenAI ``/v1/chat/completions``
    protocol — OpenRouter, DeepInfra, Together, Fireworks, Groq, self-hosted
    vLLM, etc.

    Configure via ``supervisor.yaml``:

    .. code-block:: yaml

        provider: openrouter
        api_base: https://openrouter.ai/api/v1    # REQUIRED — no hard-coded default
        api_key_env: OPENROUTER_API_KEY
        model: anthropic/claude-opus-4-7
        extra_headers:
          HTTP-Referer: "https://github.com/your/repo"
          X-Title: "My Supervisor"

    All of ``OpenAIClient``'s message formatting, image encoding, and
    ``response_format=json_object`` are reused unchanged.
    """

    def __init__(self, cfg: SupervisorConfig):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai SDK not installed; pip install openai") from e

        # ---- api_key -------------------------------------------------------
        api_key = _resolve_api_key(cfg, "OPENROUTER_API_KEY", "OPENAI_API_KEY")

        # ---- base_url -------------------------------------------------------
        if not cfg.api_base:
            raise RuntimeError(
                "provider=openrouter requires api_base to be set in supervisor.yaml. "
                "Examples:\n"
                "  OpenRouter:   api_base: https://openrouter.ai/api/v1\n"
                "  DeepInfra:    api_base: https://api.deepinfra.com/v1/openai\n"
                "  Together:     api_base: https://api.together.xyz/v1\n"
                "  Groq:         api_base: https://api.groq.com/openai/v1\n"
                "  vLLM (local): api_base: http://localhost:8000/v1"
            )
        base_url = cfg.api_base

        # ---- extra_headers --------------------------------------------------
        default_headers = dict(cfg.extra_headers) if cfg.extra_headers else {}

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        self.cfg = cfg
        self._reasoning_effort = self._build_reasoning_effort(cfg)

    def _build_reasoning_effort(self, cfg: SupervisorConfig) -> str | None:
        level = _thinking_level(cfg)
        if level is None:
            return None
        effort = _OPENROUTER_REASONING_EFFORT[level]
        logging.getLogger(__name__).info(
            "[supervisor] reasoning enabled: provider=%s level=%s effort=%s",
            cfg.provider,
            level,
            effort,
        )
        return effort

    def _call(self, system: str, user_content: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        if self._reasoning_effort is not None:
            kwargs["extra_body"] = {
                "reasoning": {
                    "effort": self._reasoning_effort,
                    "exclude": False,
                }
            }
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Stub — useful for tests and dry runs
# ---------------------------------------------------------------------------


class StubClient(LLMClient):
    """Returns deterministic empty patches; logs are still written."""

    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg

    def diagnose(self, snapshot_summary, current_weights, frames, objective_block=None, skill_block=None):
        return {
            "observations": ["stub: no analysis performed"],
            "hypotheses": [],
            "confidence": 0.0,
            "score": 0,
        }

    def propose(
        self,
        diagnose_result,
        schema_summary,
        current_weights,
        patch_limits=None,
        objective_block=None,
        skill_block=None,
    ):
        return {
            "rationale": "stub: no patch",
            "patch": {},
            "expected_effect": "",
            "rollback_if": "",
        }

    def distill_skill(self, current_skill: str, cycle: dict[str, Any]) -> str:
        return current_skill


def _render_skill_distill_body(
    current_skill: str,
    cycle: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    """Render the user message for persistent SKILL.md distillation."""
    max_chars = max(1000, int(max_chars))
    cycle_json = json.dumps(cycle, indent=2, default=str)
    body = (
        "## Current SKILL.md\n"
        "```markdown\n"
        f"{current_skill}\n"
        "```\n\n"
        "## Latest supervisor cycle evidence\n"
        "```json\n"
        f"{cycle_json}\n"
        "```\n"
    )
    if len(body) <= max_chars:
        return body

    header_budget = 600
    skill_budget = max(200, int((max_chars - header_budget) * 0.45))
    cycle_budget = max(200, max_chars - header_budget - skill_budget)
    body = (
        "## Current SKILL.md\n"
        "```markdown\n"
        f"{_clip_text(current_skill, skill_budget)}\n"
        "```\n\n"
        "## Latest supervisor cycle evidence\n"
        "```json\n"
        f"{_clip_text(cycle_json, cycle_budget)}\n"
        "```\n"
    )
    return body[:max_chars]


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    if max_chars <= len(marker) + 20:
        return text[:max_chars]
    keep_head = (max_chars - len(marker)) // 2
    keep_tail = max_chars - len(marker) - keep_head
    return text[:keep_head] + marker + text[-keep_tail:]


# ---------------------------------------------------------------------------
# JSON parsing (defensive)
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict[str, Any]:
    """Extract the largest JSON object from the response text."""
    text = text.strip()
    if not text:
        return {}
    # Fast path: the entire reply is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: scan for the first balanced {...} block.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return {}
