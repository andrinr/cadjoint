"""Editor intelligence for the playground: lint, completion, and signature help.

Three endpoints that read the editor's Python without ever running it —
``/api/lint``, ``/api/complete``, ``/api/signature``.  They are the only
playground endpoints answered in the server process rather than a worker
subprocess, because a keystroke cannot afford a fresh interpreter: linting
shells out to the ``ruff`` binary (about 6 ms) and completion drives an
in-process, warm :mod:`jedi` state (about 8-17 ms once primed).

Both analysers are static.  ``ruff`` parses; ``jedi.Script`` infers from
source text.  Neither imports or executes the program under analysis, which
is what makes it safe to run these on every keystroke while ``/compile``
still needs its disposable child process.

Coordinates
-----------
Every line/column in this module's requests and responses uses the same
convention, which is also CodeMirror's and jedi's:

- ``line`` is **1-based** (the first line of the document is line 1)
- ``column`` is **0-based** (the position before the first character is 0)

Ruff reports 1-based columns, so its columns are shifted by one on the way
out.  Client-side, a diagnostic becomes a CodeMirror range with
``doc.line(from_line).from + from_col``.

Runtime errors
--------------
A failed ``/compile`` is the most valuable diagnostic there is: it names the
line that actually blew up.  :func:`record_compile` takes the compile
worker's result, and when its traceback names a frame in the user's program
that failure is remembered and folded into the next lint of the *same*
source text.  Editing anything invalidates it, so a stale red squiggle can
never outlive the code that caused it.

Degradation
-----------
Neither analyser is a hard dependency of the package.  A missing ``ruff``
binary or a missing ``jedi`` install answers ``{"ok": false, "error": ...}``
and the editor simply goes quiet; no analyser exception is ever allowed to
reach the HTTP layer.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from cadjoint.viewer._limits import OVERSIZED_SOURCE_ERROR, exceeds_source_limit

__all__ = [
    "COMPLETION_INFO_LIMIT",
    "LINT_TIMEOUT_SECONDS",
    "MAX_COMPLETIONS",
    "RUFF_IGNORE",
    "RUFF_SELECT",
    "complete_source",
    "lint_source",
    "record_compile",
    "signature_source",
    "warm_up",
]

# ── budgets ─────────────────────────────────────────────────────────────────

LINT_TIMEOUT_SECONDS = 5.0
"""Wall clock a single ruff invocation may take before it is abandoned."""

MAX_COMPLETIONS = 200
"""Completions returned per request; the popup shows a handful of them."""

COMPLETION_INFO_LIMIT = 20
"""How many completions get a docstring.

Resolving a docstring costs jedi a fresh inference (roughly 1 ms warm, more
when cold), so only the head of the list — everything the popup can show
without scrolling — pays for one.
"""

MAX_INFO_CHARS = 800
"""Docstrings are truncated here; the popup is a hint, not the manual."""

# ── ruff ────────────────────────────────────────────────────────────────────

RUFF_SELECT = "E,W,F,B,C4,UP,SIM,I"
"""Rules a scene is linted against.

Deliberately not the repository's own ``[tool.ruff]`` configuration: the
editor lints a standalone program, and ruff runs ``--isolated`` so the
answer never depends on which directory the server was started in.
"""

RUFF_IGNORE = "E501,E402,B008"
"""Rules that are noise in a scene.

``E501`` (line length) belongs to a formatter, ``E402`` (import not at top)
fires while a scene is being reorganised, and ``B008`` (call in an argument
default) is idiomatic in this API.
"""

_SYNTAX_CODE = "invalid-syntax"

_INFO_PREFIXES = ("W", "I0", "UP", "C4", "SIM", "E1", "E2", "E3")

_APPLICABILITIES = {"safe", "unsafe", "display"}

# ── jedi ────────────────────────────────────────────────────────────────────

SCENE_FILENAME = "scene.py"
"""Name jedi and ruff see for the buffer, so both report the same file."""

# jedi's inference caches live in module-global state and are not documented
# as thread-safe; the playground serves requests on a thread each.  One lock
# around every jedi call keeps the warm state consistent, and at 8-17 ms a
# call it never becomes the bottleneck.
_JEDI_LOCK = threading.Lock()

_WARMED = threading.Event()

# The last failing ``/compile``, as ``(source, line, message, code)``.  Read
# and written under its own lock so a lint racing a compile sees one or the
# other, never a half-updated record.
_RUNTIME_LOCK = threading.Lock()
_LAST_RUNTIME: tuple[str, int, str, str] | None = None

_PLAYGROUND_FRAME = re.compile(r'File "<cadjoint-playground>", line (\d+)')


def _error(message: str) -> dict[str, Any]:
    """One rejected request, in the shape every playground endpoint answers with."""
    return {"ok": False, "error": message}


def _checked_source(request: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Validate the ``source`` field shared by all three endpoints.

    Args:
        request: The decoded JSON request body.

    Returns:
        ``(error, source)`` — ``error`` is None exactly when the source is
            acceptable, and the same size budget the rest of the API agrees
            on applies here.
    """
    source = request.get("source")
    if not isinstance(source, str):
        return _error("The request must contain a string `source` field."), ""
    if exceeds_source_limit(source):
        return _error(OVERSIZED_SOURCE_ERROR), ""
    return None, source


def _checked_position(
    request: dict[str, Any], source: str
) -> tuple[dict[str, Any] | None, int, int]:
    """Validate and clamp a caret position against the source it points into.

    A caret one keystroke ahead of the source the client last sent is normal
    while typing, so an out-of-range position is clamped to the end of the
    document rather than refused.

    Args:
        request: The decoded JSON request body.
        source: The program the position points into.

    Returns:
        ``(error, line, column)`` — ``line`` is 1-based and ``column``
            0-based, both clamped inside ``source``.
    """
    line = request.get("line")
    column = request.get("column")
    if not isinstance(line, int) or isinstance(line, bool):
        return _error("The request needs an integer `line` (1-based)."), 0, 0
    if not isinstance(column, int) or isinstance(column, bool):
        return _error("The request needs an integer `column` (0-based)."), 0, 0
    if line < 1 or column < 0:
        return _error("`line` is 1-based and `column` is 0-based."), 0, 0

    lines = source.split("\n")
    line = min(line, len(lines))
    column = min(column, len(lines[line - 1]))
    return None, line, column


# ── lint ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _ruff_binary() -> str | None:
    """Locate the ruff executable, preferring the one installed beside us."""
    try:
        from ruff.__main__ import find_ruff_bin
    except ImportError:
        pass
    else:
        try:
            return find_ruff_bin()
        except FileNotFoundError:
            pass
    return shutil.which("ruff")


def _severity(code: str) -> str:
    """Map a ruff rule code onto the three severities CodeMirror draws.

    Args:
        code: A ruff rule code such as ``F401``, or ``invalid-syntax``.

    Returns:
        One of ``"error"``, ``"warning"``, ``"info"`` — a program that
            cannot run is an error, a probable bug is a warning, and a style
            or import-order remark is info.
    """
    if code == _SYNTAX_CODE or code.startswith(("E9", "F82")):
        return "error"
    if code.startswith(_INFO_PREFIXES):
        return "info"
    return "warning"


def _column(position: Any, lines: list[str]) -> tuple[int, int]:
    """Convert one ruff ``{"row", "column"}`` position to (line, 0-based column).

    Args:
        position: A ruff location object, with 1-based row and column.
        lines: The linted source, split on newlines, for clamping.

    Returns:
        ``(line, column)`` with a 1-based line and a 0-based column, clamped
            inside the document.
    """
    if not isinstance(position, dict):
        return 1, 0
    row = position.get("row")
    col = position.get("column")
    row = row if isinstance(row, int) and row >= 1 else 1
    col = col if isinstance(col, int) and col >= 1 else 1
    row = min(row, len(lines))
    return row, min(col - 1, len(lines[row - 1]))


def _fix_payload(fix: Any, lines: list[str]) -> dict[str, Any] | None:
    """Translate ruff's autofix into the edit list the client can offer.

    Args:
        fix: The ``fix`` member of one ruff diagnostic, or None.
        lines: The linted source, split on newlines.

    Returns:
        ``{"message", "applicability", "edits"}`` in this module's coordinate
            convention, or None when the rule has no fix.
    """
    if not isinstance(fix, dict):
        return None
    edits = []
    for edit in fix.get("edits") or ():
        if not isinstance(edit, dict):
            continue
        from_line, from_col = _column(edit.get("location"), lines)
        to_line, to_col = _column(edit.get("end_location"), lines)
        content = edit.get("content")
        edits.append(
            {
                "from_line": from_line,
                "from_col": from_col,
                "to_line": to_line,
                "to_col": to_col,
                "content": content if isinstance(content, str) else "",
            }
        )
    if not edits:
        return None
    applicability = fix.get("applicability")
    message = fix.get("message")
    return {
        "message": message if isinstance(message, str) else "Apply fix",
        "applicability": applicability if applicability in _APPLICABILITIES else "unsafe",
        "edits": edits,
    }


def _ruff_diagnostics(source: str, lines: list[str]) -> tuple[str | None, list[dict[str, Any]]]:
    """Run ruff over one program and translate its JSON report.

    Args:
        source: The program to lint.
        lines: ``source`` split on newlines.

    Returns:
        ``(error, diagnostics)`` — ``error`` is a message when ruff could not
            be run at all, in which case the diagnostics are empty.
    """
    binary = _ruff_binary()
    if binary is None:
        return "Linting needs the `ruff` executable; install cadjoint[editor].", []

    command = [
        binary,
        "check",
        "--output-format",
        "json",
        "--stdin-filename",
        SCENE_FILENAME,
        "--isolated",
        "--no-cache",
        "--select",
        RUFF_SELECT,
        "--ignore",
        RUFF_IGNORE,
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=source.encode("utf-8"),
            capture_output=True,
            timeout=LINT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Linting exceeded the {LINT_TIMEOUT_SECONDS:g}-second timeout.", []
    except OSError as failure:
        return f"Could not run `ruff`: {failure}.", []

    try:
        report = json.loads(completed.stdout or b"[]")
    except (json.JSONDecodeError, UnicodeDecodeError):
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        return f"Unreadable ruff report: {detail[-500:]}", []
    if not isinstance(report, list):
        return "Unreadable ruff report.", []

    diagnostics = []
    for item in report:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        code = code if isinstance(code, str) else _SYNTAX_CODE
        from_line, from_col = _column(item.get("location"), lines)
        to_line, to_col = _column(item.get("end_location"), lines)
        message = item.get("message")
        url = item.get("url")
        diagnostics.append(
            {
                "from_line": from_line,
                "from_col": from_col,
                "to_line": to_line,
                "to_col": to_col,
                "severity": _severity(code),
                "message": message if isinstance(message, str) else "",
                "code": code,
                "source": "ruff",
                "url": url if isinstance(url, str) else None,
                "fix": _fix_payload(item.get("fix"), lines),
            }
        )
    return None, diagnostics


def record_compile(source: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Remember (or forget) the last runtime failure of a program.

    Wrapped around ``/compile`` and ``/api/mesh`` by the routing table so the
    linter can surface a traceback that names a line of the user's scene —
    the one diagnostic no static analyser can produce.  The result is
    returned unchanged; this is a tap on the wire, not a filter.

    Args:
        source: The program that was run, as the request carried it.
        result: The worker's response.

    Returns:
        ``result``, untouched, so this can wrap an endpoint inline.
    """
    global _LAST_RUNTIME

    record = None
    if isinstance(source, str) and not result.get("ok"):
        detail = result.get("error")
        if isinstance(detail, str):
            frames = _PLAYGROUND_FRAME.findall(detail)
            if frames:
                message = next(
                    (line.strip() for line in reversed(detail.splitlines()) if line.strip()),
                    "Execution failed.",
                )
                code = message.split(":", 1)[0].strip() if ":" in message else "RuntimeError"
                record = (source, int(frames[-1]), message, code)
    with _RUNTIME_LOCK:
        _LAST_RUNTIME = record
    return result


def _runtime_diagnostic(source: str, lines: list[str]) -> dict[str, Any] | None:
    """The remembered runtime failure, if it still belongs to this text.

    Args:
        source: The program being linted.
        lines: ``source`` split on newlines.

    Returns:
        One diagnostic marking the line the traceback blamed, or None when
            nothing failed or the buffer has changed since it did.
    """
    with _RUNTIME_LOCK:
        record = _LAST_RUNTIME
    if record is None:
        return None
    recorded, line, message, code = record
    if recorded != source or not 1 <= line <= len(lines):
        return None
    text = lines[line - 1]
    start = len(text) - len(text.lstrip())
    return {
        "from_line": line,
        "from_col": min(start, max(len(text) - 1, 0)),
        "to_line": line,
        "to_col": len(text),
        "severity": "error",
        "message": message,
        "code": code,
        "source": "runtime",
        "url": None,
        "fix": None,
    }


def lint_source(request: dict[str, Any]) -> dict[str, Any]:
    """Lint one program: ``{"source"}`` → ``{"diagnostics"}``.

    Diagnostics come from ruff plus, when the same text failed its last
    ``/compile``, the traceback's own line.  Lines are 1-based and columns
    0-based; see the module docstring.

    Args:
        request: ``{"source": str}``.

    Returns:
        ``{"ok": True, "diagnostics": [...], "runtime": bool}`` — ``runtime``
            says whether a remembered traceback contributed one of them.
    """
    failure, source = _checked_source(request)
    if failure is not None:
        return failure

    lines = source.split("\n")
    try:
        error, diagnostics = _ruff_diagnostics(source, lines)
    except Exception as failure:  # noqa: BLE001 - an analyser must never 500 the server
        return _error(f"Linting failed: {failure!r}")
    if error is not None:
        return _error(error)

    runtime = _runtime_diagnostic(source, lines)
    if runtime is not None:
        diagnostics.insert(0, runtime)
    return {"ok": True, "diagnostics": diagnostics, "runtime": runtime is not None}


# ── jedi ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _project() -> Any:
    """The jedi project the editor's scenes are analysed inside.

    Pinned to this interpreter's ``sys.path`` so jedi resolves the *installed*
    ``cadjoint`` — completions therefore know the real signatures of
    ``SketchPlane``, ``extrude``, ``ThermalStudy`` and the rest instead of
    guessing.  ``load_unsafe_extensions`` stays off: jedi may inspect a
    compiled dependency, but only ever in its own helper process.
    """
    import jedi

    root = Path.cwd()
    return jedi.Project(
        str(root),
        sys_path=[entry for entry in sys.path if entry],
        added_sys_path=[str(root)],
        smart_sys_path=False,
        load_unsafe_extensions=False,
    )


def _script(source: str) -> Any:
    """A jedi Script over one scene, inside the warm project."""
    import jedi

    return jedi.Script(code=source, path=str(Path.cwd() / SCENE_FILENAME), project=_project())


def _completion_type(kind: str) -> str:
    """Map a jedi completion type onto a CodeMirror completion type."""
    return {
        "module": "namespace",
        "class": "class",
        "instance": "variable",
        "function": "function",
        "param": "property",
        "path": "text",
        "keyword": "keyword",
        "property": "property",
        "statement": "variable",
    }.get(kind, "variable")


def _interesting(name: str, prefix_length: int, source_line: str, column: int) -> bool:
    """Whether a completion is worth showing.

    Dunders and private names bury the API the popup is meant to teach, so
    they appear only once the caret has actually typed a leading underscore.

    Args:
        name: The completion label.
        prefix_length: How many characters of it the user already typed.
        source_line: The line the caret sits on.
        column: The caret's 0-based column.

    Returns:
        True when the completion should be offered.
    """
    if not name.startswith("_"):
        return True
    typed = source_line[column - prefix_length : column] if prefix_length else ""
    return typed.startswith("_")


def complete_source(request: dict[str, Any]) -> dict[str, Any]:
    """Complete at a caret: ``{"source", "line", "column"}`` → ``{"completions"}``.

    Args:
        request: ``{"source": str, "line": int, "column": int}`` with a
            1-based line and 0-based column.

    Returns:
        ``{"ok": True, "from_line", "from_column", "completions", "truncated"}``
            — ``from_column`` is where the already-typed prefix starts, so the
            client replaces ``from_column``..``column`` with a completion's
            ``apply`` text.
    """
    failure, source = _checked_source(request)
    if failure is not None:
        return failure
    failure, line, column = _checked_position(request, source)
    if failure is not None:
        return failure

    try:
        import jedi  # noqa: F401 - presence check before the analyser runs
    except ImportError:
        return _error("Completion needs `jedi`; install cadjoint[editor].")

    source_line = source.split("\n")[line - 1]
    try:
        with _JEDI_LOCK:
            raw = _script(source).complete(line, column)
            prefix_length = raw[0].get_completion_prefix_length() if raw else 0
            kept = [
                item for item in raw if _interesting(item.name, prefix_length, source_line, column)
            ]
            kept = _keyword_arguments_first(kept)
            truncated = len(kept) > MAX_COMPLETIONS
            kept = kept[:MAX_COMPLETIONS]
            completions = [
                _completion_payload(item, index < COMPLETION_INFO_LIMIT)
                for index, item in enumerate(kept)
            ]
    except Exception as failure:  # noqa: BLE001 - an analyser must never 500 the server
        return _error(f"Completion failed: {failure!r}")

    return {
        "ok": True,
        "from_line": line,
        "from_column": max(column - prefix_length, 0),
        "completions": completions,
        "truncated": truncated,
    }


def _keyword_arguments_first(items: list[Any]) -> list[Any]:
    """Hoist the enclosing call's keyword arguments to the head of the list.

    Jedi does offer ``origin=`` and ``normal=`` inside ``SketchPlane(`` — but
    in one alphabetical run with every builtin, so at the moment the popup is
    most useful they sit a hundred rows down.  The arguments of the call the
    caret is actually inside are the answer to the question being asked, so
    they go first (and so they are inside the head of the list that pays for
    an ``info``); everything else keeps jedi's order.

    Args:
        items: The completions jedi produced, in its own order.

    Returns:
        The same completions, stably partitioned with the keyword arguments
            in front.
    """
    keywords = [item for item in items if item.type == "param"]
    if not keywords:
        return items
    return keywords + [item for item in items if item.type != "param"]


def _completion_payload(item: Any, with_info: bool) -> dict[str, Any]:
    """One jedi completion in the shape CodeMirror's ``Completion`` wants.

    Args:
        item: A ``jedi.api.classes.Completion``.
        with_info: Whether to pay for its docstring.

    Returns:
        ``{"label", "type", "detail", "info", "apply"}`` — ``info`` is the
            signature-and-docstring the popup teaches from, present only for
            the head of the list.
    """
    info = None
    if with_info:
        try:
            # A keyword argument has no docstring of its own; its description
            # ("param origin=(0.0, 0.0, 0.0)") carries the default instead,
            # which is exactly what the popup should show for one.
            info = (item.description if item.type == "param" else item.docstring(raw=False)) or None
        except Exception:  # noqa: BLE001 - one unresolvable name must not lose the list
            info = None
        if info is not None and len(info) > MAX_INFO_CHARS:
            info = info[:MAX_INFO_CHARS].rstrip() + "\n…"
    return {
        "label": item.name,
        "type": _completion_type(item.type),
        "detail": item.type,
        "info": info,
        "apply": item.name,
    }


def signature_source(request: dict[str, Any]) -> dict[str, Any]:
    """Describe the call the caret sits inside: ``{"source", "line", "column"}``.

    Args:
        request: ``{"source": str, "line": int, "column": int}`` with a
            1-based line and 0-based column.

    Returns:
        ``{"ok": True, "signatures": [...]}`` — each signature carries its
            rendered ``label``, its ``parameters``, and ``active_parameter``:
            the index of the argument being typed, or None between calls.
    """
    failure, source = _checked_source(request)
    if failure is not None:
        return failure
    failure, line, column = _checked_position(request, source)
    if failure is not None:
        return failure

    try:
        import jedi  # noqa: F401 - presence check before the analyser runs
    except ImportError:
        return _error("Signature help needs `jedi`; install cadjoint[editor].")

    try:
        with _JEDI_LOCK:
            signatures = [
                _signature_payload(item) for item in _script(source).get_signatures(line, column)
            ]
    except Exception as failure:  # noqa: BLE001 - an analyser must never 500 the server
        return _error(f"Signature help failed: {failure!r}")

    return {"ok": True, "signatures": signatures}


def _signature_payload(item: Any) -> dict[str, Any]:
    """One jedi signature in the shape a signature-help tooltip wants."""
    try:
        documentation = item.docstring(raw=True) or None
    except Exception:  # noqa: BLE001 - a signature is useful without its prose
        documentation = None
    if documentation is not None and len(documentation) > MAX_INFO_CHARS:
        documentation = documentation[:MAX_INFO_CHARS].rstrip() + "\n…"
    parameters = []
    for param in item.params:
        try:
            label = param.to_string()
        except Exception:  # noqa: BLE001 - fall back to the bare name
            label = param.name
        parameters.append({"name": param.name, "label": label})
    return {
        "name": item.name,
        "label": item.to_string(),
        "active_parameter": item.index,
        "parameters": parameters,
        "documentation": documentation,
    }


def warm_up() -> None:
    """Pay jedi's first-call cost off the request path, once per process.

    The first completion in a process costs a few hundred milliseconds while
    jedi parses the standard library and ``cadjoint``; every later one costs
    single-digit milliseconds.  Servers call this on a daemon thread at
    startup so the editor's first popup is already warm.
    """
    if _WARMED.is_set():
        return
    _WARMED.set()

    def prime() -> None:
        from cadjoint.viewer._example_scene import EXAMPLE_SOURCE

        try:
            with _JEDI_LOCK:
                _script(EXAMPLE_SOURCE + "\nSk").complete(len(EXAMPLE_SOURCE.split("\n")) + 1, 2)
        except Exception:  # noqa: BLE001 - a cold cache is the only cost of failing
            pass

    threading.Thread(target=prime, name="cadjoint-intelligence-warmup", daemon=True).start()
