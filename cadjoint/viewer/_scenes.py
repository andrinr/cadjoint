"""Saved scene files: the ``/api/scenes`` endpoints and their storage rules.

Scenes are plain ``*.py`` programs in one directory under the server's
working directory.  Requests name a bare file only — every path separator,
traversal, hidden file, and non-``.py`` suffix is refused by
:func:`sanitize_scene_name` before anything touches the filesystem, and the
resolved path is checked to still sit inside the scenes directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cadjoint.viewer._limits import (
    MAX_SOURCE_BYTES,
    OVERSIZED_SOURCE_ERROR,
    exceeds_source_limit,
)

# Saved scenes live in one directory under the server's working directory.
# Requests supply bare ``*.py`` names only; anything resembling a path is
# rejected before it touches the filesystem.
SCENES_DIRNAME = "scenes"
MAX_SCENE_NAME_LENGTH = 128
_SCENE_STEM = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._ -]*")


def scenes_root() -> Path:
    """Directory holding saved scene files (created lazily on first save)."""
    return Path.cwd() / SCENES_DIRNAME


def sanitize_scene_name(name: Any) -> str | None:
    """Validate a scene file name, or return None if it is unacceptable.

    Accepts bare file names such as ``bracket.py``. Path separators, traversal
    (``../evil.py``), hidden files, and non-``.py`` suffixes are all refused.
    """
    if not isinstance(name, str) or not name.endswith(".py"):
        return None
    if len(name) > MAX_SCENE_NAME_LENGTH:
        return None
    if "/" in name or "\\" in name or "\x00" in name:
        return None
    stem = name[: -len(".py")]
    if not _SCENE_STEM.fullmatch(stem):
        return None
    return name


def _scene_path(name: str) -> Path | None:
    """Resolve a sanitized name inside the scenes directory, or None."""
    candidate = (scenes_root() / name).resolve()
    try:
        candidate.relative_to(scenes_root().resolve())
    except ValueError:
        return None
    return candidate


def list_scenes() -> dict[str, Any]:
    """List saved scene files as bare names, newest directory state wins."""
    root = scenes_root()
    if not root.is_dir():
        return {"ok": True, "files": []}
    names = sorted(
        path.name for path in root.glob("*.py") if path.is_file() and sanitize_scene_name(path.name)
    )
    return {"ok": True, "files": names}


def load_scene(request: dict[str, Any]) -> dict[str, Any]:
    """Read one saved scene file: ``{"name"}`` → ``{"source"}``."""
    name = sanitize_scene_name(request.get("name"))
    if name is None:
        return {"ok": False, "error": "Scene `name` must be a bare `*.py` file name."}
    path = _scene_path(name)
    if path is None or not path.is_file():
        return {"ok": False, "error": f"No saved scene named {name!r}."}
    if path.stat().st_size > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"{name!r} is larger than the source limit."}
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"ok": False, "error": f"Could not read {name!r}."}
    return {"ok": True, "name": name, "source": source}


def save_scene(request: dict[str, Any]) -> dict[str, Any]:
    """Write one scene file: ``{"name", "source"}`` → ``{"name"}``."""
    name = sanitize_scene_name(request.get("name"))
    if name is None:
        return {"ok": False, "error": "Scene `name` must be a bare `*.py` file name."}
    source = request.get("source")
    if not isinstance(source, str):
        return {"ok": False, "error": "The save request must contain a string `source` field."}
    if exceeds_source_limit(source):
        return {"ok": False, "error": OVERSIZED_SOURCE_ERROR}
    path = _scene_path(name)
    if path is None:
        return {"ok": False, "error": "Scene `name` must stay inside the scenes directory."}
    try:
        scenes_root().mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    except OSError as error:
        return {"ok": False, "error": f"Could not save {name!r}: {error}."}
    return {"ok": True, "name": name}
