"""Training-objective spec loaded from a JSON file at startup.

The JSON tells the LLM *what* to optimise for ("walk stably", "walk fast",
"stay energy-efficient", ...). It is passed verbatim to the diagnose and
propose prompts so every patch decision is biased toward that goal.

Fields are **dynamic** — any key in the JSON becomes an accessible attribute.
Adding a new key to a ``.json`` objective file does not require code changes.

Example JSON::

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
import logging
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)


def _objectives_dir() -> Path:
    """Path to the built-in objectives directory."""
    return Path(__file__).parent / "objectives"


# ---------------------------------------------------------------------------
# Rendering hints for to_prompt_block().
# Keys listed here get structured formatting; anything else gets a sensible
# default (lists become bullet items, scalars become "key: value" lines).
# ---------------------------------------------------------------------------
_TITLE_KEY = "name"
_TEXT_KEY = "summary"  # rendered as a standalone text paragraph after the title
_NUMBERED_KEYS = frozenset({"priorities"})
_BULLET_KEYS = frozenset({"avoid"})
_FIELD_KEYS = frozenset({"notes", "reference_frame"})


class TrainingObjective:
    """A training objective whose fields are driven entirely by the JSON file.

    Every top-level key in the JSON becomes an attribute on the instance.
    The canonical defaults (applied when a key is missing) are:

    - ``name``: ``"default"``
    - ``summary``: ``""``
    """

    __slots__ = ("_data",)

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = {"name": "default", "summary": ""}
        self._data.update(kwargs)

    # -- attribute access ----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails (i.e. key not in instance
        # __dict__, and since we use __slots__ that means it's not a slot).
        # _data is a slot so this only fires for JSON keys.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} has no key {name!r}"
            ) from None

    # -- dict-like interface -------------------------------------------------

    def keys(self) -> Any:  # returns dict_keys, but Any keeps the API loose
        return self._data.keys()

    def items(self) -> Any:
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"TrainingObjective({inner})"

    # -- I/O ----------------------------------------------------------------

    @staticmethod
    def list_available() -> list[dict[str, str]]:
        """Scan the built-in objectives directory and return a list of
        ``{"name": ..., "summary": ..., "file": ...}`` dicts for every
        ``.json`` file found.  Results are sorted by name."""
        objs: list[dict[str, str]] = []
        obj_dir = _objectives_dir()
        if not obj_dir.exists():
            return objs
        for p in sorted(obj_dir.glob("*.json")):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
            except Exception:
                log.warning("Failed to read objective file: %s", p)
                continue
            if not isinstance(data, dict):
                log.warning("Objective file is not a JSON object: %s", p)
                continue
            name = p.stem
            json_name = str(data.get("name", ""))
            if json_name and json_name != name:
                log.warning(
                    "Objective file name %r doesn't match content name %r; using filename",
                    name,
                    json_name,
                )
            objs.append(
                {
                    "name": name,
                    "summary": str(data.get("summary", "")),
                    "file": str(p),
                }
            )
        return objs

    @staticmethod
    def load_by_name(name: str) -> "TrainingObjective":
        """Load an objective by its short name (e.g. ``"stable_walk"``),
        looking up ``objectives/{name}.json``."""
        p = _objectives_dir() / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"objective file not found: {p} (searched by name {name!r})"
            )
        return TrainingObjective.load(p)

    @staticmethod
    def load(path: str | Path | None) -> "TrainingObjective":
        """Load an objective from a JSON file.  Behaviour depends on *path*:

        * ``None`` → built-in default.
        * Contains ``/`` or ends with ``.json`` → treated as a file path.
        * Otherwise → treated as a short name, looked up in the built-in
          objectives directory.
        """
        if path is None:
            return TrainingObjective()

        path_str = str(path)
        # Heuristic: if it looks like a bare name, resolve via load_by_name.
        if "/" not in path_str and "\\" not in path_str and not path_str.endswith(".json"):
            return TrainingObjective.load_by_name(path_str)

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"objective file not found: {p}")
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"objective JSON must be an object, got {type(data)}")

        # Every JSON key becomes a dynamic member.
        return TrainingObjective(**data)

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the underlying data dict (useful for
        serialisation / audit logging without enumerating every key)."""
        return dict(self._data)

    # -- prompt rendering ----------------------------------------------------

    def to_prompt_block(self) -> str:
        """Render as a compact text block injected into the LLM prompt.

        Rendering is driven by the data, not by hardcoded members:
        *name* becomes the title line; *summary* becomes a standalone text
        paragraph; fields listed in ``_NUMBERED_KEYS`` are rendered as
        numbered lists; fields in ``_BULLET_KEYS`` as bullet lists; all
        other fields become ``"Key: value"`` lines (list values become
        one bullet per item).  Unknown keys added to a JSON file therefore
        appear in the prompt automatically.
        """
        data = self._data
        lines: list[str] = []

        # Title
        name = data.get(_TITLE_KEY, "")
        lines.append(f"# Training objective: {name}")

        # Summary (standalone text paragraph, no key label)
        summary = data.get(_TEXT_KEY, "")
        if summary:
            lines.append(str(summary))

        # Everything else, in insertion order.
        for key, value in data.items():
            if key in (_TITLE_KEY, _TEXT_KEY):
                continue  # already rendered above
            if not value:
                continue  # skip empty values

            if key in _NUMBERED_KEYS:
                lines.append(f"{key.capitalize()} (in order):")
                for i, item in enumerate(value, 1):
                    lines.append(f"  {i}. {item}")
            elif key in _BULLET_KEYS:
                lines.append(f"{key.capitalize()}:")
                for item in value:
                    lines.append(f"  - {item}")
            elif key in _FIELD_KEYS:
                lines.append(f"{key.capitalize()}: {value}")
            elif isinstance(value, list):
                # Generic list → bullet list
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)
