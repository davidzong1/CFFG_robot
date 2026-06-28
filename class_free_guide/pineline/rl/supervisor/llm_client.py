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
import os
from abc import ABC, abstractmethod
from typing import Any

from .config import SupervisorConfig
from .prompts import DIAGNOSE_SYSTEM, PROPOSE_SYSTEM, render_state


class LLMClient(ABC):
    """Two-call interface used by the supervisor."""

    @abstractmethod
    def diagnose(
        self,
        snapshot_summary: dict[str, Any],
        current_weights: dict[str, float],
        frames: list[tuple[str, bytes]],
        objective_block: str | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def propose(
        self,
        diagnose_result: dict[str, Any],
        schema_summary: dict[str, dict[str, float]],
        current_weights: dict[str, float],
        objective_block: str | None = None,
    ) -> dict[str, Any]: ...


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
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {cfg.api_key_env} is not set"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.cfg = cfg

    def _call(self, system: str, user_blocks: list[dict[str, Any]]) -> str:
        # ``cache_control`` on the system prompt — it is stable across cycles
        # and large enough to benefit from caching.
        resp = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_blocks}],
        )
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
    ) -> dict[str, Any]:
        user: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": render_state(
                    snapshot_summary, current_weights, objective_block=objective_block
                ),
            }
        ]
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
        objective_block: str | None = None,
    ) -> dict[str, Any]:
        body = (
            "## Prior diagnosis\n```json\n"
            + json.dumps(diagnose_result, indent=2)
            + "\n```\n\n"
            + render_state(
                {},
                current_weights,
                schema_summary=schema_summary,
                objective_block=objective_block,
            )
        )
        text = self._call(PROPOSE_SYSTEM, [{"type": "text", "text": body}])
        return _parse_json(text)


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    def __init__(self, cfg: SupervisorConfig):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai SDK not installed; pip install openai") from e
        api_key = os.environ.get(cfg.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"environment variable {cfg.api_key_env} (or OPENAI_API_KEY) is not set"
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if cfg.api_base:
            kwargs["base_url"] = cfg.api_base
        if cfg.extra_headers:
            kwargs["default_headers"] = dict(cfg.extra_headers)
        self._client = openai.OpenAI(**kwargs)
        self.cfg = cfg

    def _call(self, system: str, user_content: list[dict[str, Any]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content or ""

    def diagnose(
        self,
        snapshot_summary: dict[str, Any],
        current_weights: dict[str, float],
        frames: list[tuple[str, bytes]],
        objective_block: str | None = None,
    ) -> dict[str, Any]:
        user: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": render_state(
                    snapshot_summary, current_weights, objective_block=objective_block
                ),
            }
        ]
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
        return _parse_json(self._call(DIAGNOSE_SYSTEM, user))

    def propose(
        self,
        diagnose_result: dict[str, Any],
        schema_summary: dict[str, dict[str, float]],
        current_weights: dict[str, float],
        objective_block: str | None = None,
    ) -> dict[str, Any]:
        body = (
            "## Prior diagnosis\n```json\n"
            + json.dumps(diagnose_result, indent=2)
            + "\n```\n\n"
            + render_state(
                {},
                current_weights,
                schema_summary=schema_summary,
                objective_block=objective_block,
            )
        )
        return _parse_json(self._call(PROPOSE_SYSTEM, [{"type": "text", "text": body}]))


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
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            # Try a few common fallbacks so the user doesn't need to set
            # api_key_env explicitly for every provider.
            for fallback in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
                api_key = os.environ.get(fallback)
                if api_key:
                    break
        if not api_key:
            raise RuntimeError(
                f"No API key found. Tried env vars: {cfg.api_key_env}, "
                f"OPENROUTER_API_KEY, OPENAI_API_KEY. "
                f"Set one of them or change api_key_env in your supervisor yaml."
            )

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


# ---------------------------------------------------------------------------
# Stub — useful for tests and dry runs
# ---------------------------------------------------------------------------


class StubClient(LLMClient):
    """Returns deterministic empty patches; logs are still written."""

    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg

    def diagnose(self, snapshot_summary, current_weights, frames, objective_block=None):
        return {
            "observations": ["stub: no analysis performed"],
            "hypotheses": [],
            "confidence": 0.0,
        }

    def propose(self, diagnose_result, schema_summary, current_weights, objective_block=None):
        return {
            "rationale": "stub: no patch",
            "patch": {},
            "expected_effect": "",
            "rollback_if": "",
        }


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
