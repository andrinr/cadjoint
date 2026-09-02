"""Saved scene files: the ``/api/scenes`` endpoints and their storage rules.

Scenes are plain ``*.py`` programs in one directory under the server's
working directory.  Requests name a bare file only — every path separator,
traversal, hidden file, and non-``.py`` suffix is refused by
:func:`sanitize_scene_name` before anything touches the filesystem, and the
resolved path is checked to still sit inside the scenes directory.

The listing carries a **description** of each scene as well as its name, so
a browser can show what a file contains before anyone opens it: the first
paragraph of its module docstring, and counts of the declarations that
matter — named and free design parameters, studies, meshes, optimizations,
and the materials it defines.  All of that is read with :mod:`ast`
(:func:`summarize_scene`), never by executing the file.  A scene browser
that ran every program in the directory to describe it would be a scene
browser that runs arbitrary code on a directory listing, which is not a
trade this server makes anywhere else either.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from datetime import datetime, timezone
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
#: Environment override for where saved scenes live.
#:
#: The default is ``./scenes`` under the server's working directory, which is
#: the repository's own scene directory when the playground is started from a
#: checkout.  That is right for a person and wrong for a test: a Playwright run
#: that exercises "Save As" would write a file into the repository and leave it
#: there.  Pointing this at a temporary directory seeded with copies of the
#: shipped scenes gives an automated run the same listing to browse and nowhere
#: to leak into.
SCENES_DIR_ENV = "CADJOINT_SCENES_DIR"
MAX_SCENE_NAME_LENGTH = 128
_SCENE_STEM = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._ -]*")


def scenes_root() -> Path:
    """Directory holding saved scene files (created lazily on first save).

    Read from the environment on every call rather than cached, so a test can
    point one server at a temporary directory without the import order of this
    module deciding the answer.
    """
    override = os.environ.get(SCENES_DIR_ENV)
    if override:
        return Path(override).expanduser()
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


#: Calls that declare a design parameter; a `name=` makes it a *named* one.
#:
#: All three constructors count, and they are counted wherever they appear —
#: not only in a top-level assignment.  Most of a real scene's design freedom
#: is *inside* a sketch: the starter declares three top-level scalars and
#: nineteen free ``Vector2`` sketch points, so a counter that looked only at
#: ``Scalar`` at module level reported 5 named and 3 free for a program whose
#: optimizer drives nineteen.
PARAMETER_CALLS = ("Scalar", "Vector", "Vector2")
#: Calls that declare a finite-element study.
STUDY_CALLS = ("ThermalStudy", "ElasticStudy")
#: Everything else counted, by the call that declares it.
MESH_CALLS = ("SimMesh",)
OPTIMIZATION_CALLS = ("Optimization",)
MATERIAL_CALLS = ("Material",)

#: Longest summary line kept; a docstring paragraph can run for pages.
MAX_SUMMARY_CHARS = 280


def _called_name(node: ast.Call) -> str | None:
    """The bare name of what a call calls, attribute-qualified or not."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    """One keyword argument's value, or None when it was not passed."""
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _string_keyword(node: ast.Call, name: str) -> str | None:
    """A keyword argument's value when it is a plain string literal."""
    value = _keyword(node, name)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _is_true(value: ast.expr | None) -> bool:
    """Whether an argument is the literal ``True``."""
    return isinstance(value, ast.Constant) and value.value is True


def docstring_summary(module: ast.Module) -> str:
    """The first paragraph of a module docstring, as one line.

    Args:
        module: The parsed scene program.

    Returns:
        The paragraph with its line breaks collapsed to single spaces, cut to
        :data:`MAX_SUMMARY_CHARS`; an empty string when the module has no
        docstring.
    """
    docstring = ast.get_docstring(module)
    if not docstring:
        return ""
    paragraph = docstring.strip().split("\n\n", 1)[0]
    summary = " ".join(paragraph.split())
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return summary


def summarize_scene(source: str) -> dict[str, Any]:
    """Describe a scene program without running a line of it.

    One :mod:`ast` pass over the whole file: the module docstring's first
    paragraph, and a count of each kind of declaration the playground cares
    about.  Parameters are counted only when they carry a ``name=`` — an
    unnamed ``Scalar(0.07)`` inside a primitive is a literal, not a design
    freedom — and a parameter is *free* when it also carries ``free=True``,
    which is exactly the set an optimization may move.

    A file that does not parse is described as far as it can be: the counts
    come back zero and ``error`` carries the syntax error, so a broken scene
    is still listed rather than silently missing from the browser.

    Args:
        source: The program text.

    Returns:
        ``{"summary", "counts", "materials", "error"}`` — see the module
        docstring for what the counts mean.
    """
    counts = {
        "parameters": 0,
        "free": 0,
        "studies": 0,
        "meshes": 0,
        "optimizations": 0,
        "materials": 0,
    }
    materials: list[str] = []
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        return {"summary": "", "counts": counts, "materials": [], "error": str(error)}

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        called = _called_name(node)
        if called is None:
            continue
        if called in PARAMETER_CALLS:
            if _string_keyword(node, "name") is not None:
                counts["parameters"] += 1
                if _is_true(_keyword(node, "free")):
                    counts["free"] += 1
        elif called in STUDY_CALLS:
            counts["studies"] += 1
        elif called in MESH_CALLS:
            counts["meshes"] += 1
        elif called in OPTIMIZATION_CALLS:
            counts["optimizations"] += 1
        elif called in MATERIAL_CALLS:
            counts["materials"] += 1
            name = _string_keyword(node, "name")
            if name and name not in materials:
                materials.append(name)

    return {
        "summary": docstring_summary(module),
        "counts": counts,
        "materials": materials,
        "error": None,
    }


def describe_scene(path: Path) -> dict[str, Any]:
    """One listing entry: identity, size, modification time, and a summary.

    ``source_hash`` is the same sha256 the job registry stamps on a request,
    so a client can cache anything derived from a scene — a rendered
    thumbnail, say — against the exact bytes it was derived from, and notice
    the moment the file changes underneath it.

    Args:
        path: The scene file, already inside the scenes directory.

    Returns:
        A JSON-safe entry, or one carrying ``error`` when the file could not
        be read at all.
    """
    entry: dict[str, Any] = {
        "name": path.name,
        "path": f"{SCENES_DIRNAME}/{path.name}",
        "bytes": 0,
        "modified": None,
        "source_hash": None,
        "summary": "",
        "counts": {
            "parameters": 0,
            "free": 0,
            "studies": 0,
            "meshes": 0,
            "optimizations": 0,
            "materials": 0,
        },
        "materials": [],
        "error": None,
    }
    try:
        stat = path.stat()
        entry["bytes"] = stat.st_size
        entry["modified"] = (
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        entry["error"] = f"Could not read {path.name!r}: {error}"
        return entry

    entry["source_hash"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    entry.update(summarize_scene(source))
    return entry


def list_scenes() -> dict[str, Any]:
    """List saved scene files, by name and with a description of each.

    ``files`` is the original contract — bare names, sorted — and is still
    what the Open dialog reads.  ``scenes`` is the same set described:
    :func:`describe_scene` per file, in the same order, for the browser that
    shows what is in them.  Describing a scene is a stat, a read and one
    :mod:`ast` parse; nothing here executes a scene program.

    Returns:
        ``{"ok", "files", "scenes"}``; both lists are empty when no scenes
        directory exists yet.
    """
    root = scenes_root()
    if not root.is_dir():
        return {"ok": True, "files": [], "scenes": []}
    paths = sorted(
        (path for path in root.glob("*.py") if path.is_file() and sanitize_scene_name(path.name)),
        key=lambda path: path.name,
    )
    return {
        "ok": True,
        "files": [path.name for path in paths],
        "scenes": [describe_scene(path) for path in paths],
    }


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
