"""Where a plugin runs: a name for one ``Tesseract.from_*`` call.

A :class:`PluginSpec` is configuration, not a client.  Each transport is
exactly one tesseract-core constructor, and ``options`` is forwarded to it
verbatim — this module opens no sockets, encodes nothing, and knows nothing
about HTTP:

===========  ============================================================
transport    the call it makes
===========  ============================================================
``local``    ``Tesseract.from_tesseract_api(api_path, **options)``
``container`` ``Tesseract.from_image(image, **options)`` then ``serve()``
``remote``   ``Tesseract.from_url(url, **options)``
===========  ============================================================

The Kubernetes case is ``remote`` pointed at a Service; ``tesseract serve``
is what puts a Tesseract behind such an address (see ``docs/plugins.qmd``).

``${VAR}`` in ``api_path``, ``image`` and ``url`` expands from the
environment when the spec is built, so a namespace or a host stays out of
the file that is checked in.

This module is also the one place that decides *where the runtime writes*.
Every Tesseract call opens a ``run_<uuid>/logs/`` scratch directory under
the runtime's output path, which defaults to the process's working
directory for a served Tesseract and to an unmanaged ``mkdtemp`` in-process
— so an unconfigured checkout accumulates hundreds of ``run_*`` directories.
:func:`runtime_scratch` gives it one per-process directory outside the
checkout instead, removed at exit; see that function for the served case.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

#: The transports a spec can name, and the constructor each resolves to.
TRANSPORTS = ("local", "container", "remote")

#: Which field each transport requires.
_TARGET = {"local": "api_path", "container": "image", "remote": "url"}

_TESSERACT_EXTRA_MESSAGE = (
    "tesseract-core / tesseract-jax are not installed. "
    "Install the 'tesseract' extra: pip install cadjoint[tesseract]."
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Environment variable overriding where the runtime's scratch lives.
SCRATCH_ENV = "CADJOINT_TESSERACT_RUNS"
#: The environment variable a *served* Tesseract reads for the same thing.
SERVED_OUTPUT_ENV = "TESSERACT_OUTPUT_PATH"

_scratch: Path | None = None
_scratch_lock = threading.Lock()


def scratch_root() -> Path:
    """The directory this machine keeps Tesseract run scratch under.

    ``$CADJOINT_TESSERACT_RUNS`` overrides it; otherwise it follows the XDG
    cache location beside :func:`cadjoint.cache.cache_directory`'s own
    subdirectory, i.e. ``~/.cache/cadjoint/tesseract-runs``.

    Returns:
        The root directory (not created).
    """
    override = os.environ.get(SCRATCH_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "cadjoint" / "tesseract-runs"


def runtime_scratch() -> Path:
    """One scratch directory for this process, removed when it exits.

    The Tesseract runtime writes a ``run_<uuid>/logs/`` directory per
    endpoint call and never cleans it up.  In-process that lands in an
    unmanaged ``mkdtemp``; **served**, it lands in the server's working
    directory — which is why an unconfigured ``tesseract-runtime serve``
    started from a checkout fills it with ``run_*``.  This function is the
    one answer to both: :meth:`PluginSpec.open` passes it to
    ``Tesseract.from_tesseract_api`` as ``output_path``, and a server should
    be started with ``TESSERACT_OUTPUT_PATH`` (:data:`SERVED_OUTPUT_ENV`)
    set to it or to any other path outside the repository.

    Created on first use and registered for removal with :mod:`atexit`, so
    a long optimization's thousands of run directories do not outlive the
    process that made them.

    Returns:
        The per-process scratch directory.
    """
    global _scratch
    with _scratch_lock:
        if _scratch is None:
            root = scratch_root()
            root.mkdir(parents=True, exist_ok=True)
            path = Path(tempfile.mkdtemp(prefix=f"pid{os.getpid()}-", dir=root))
            atexit.register(shutil.rmtree, path, True)
            _scratch = path
        return _scratch


class PluginConfigError(ValueError):
    """A plugin spec is malformed or names a transport it cannot serve."""


def expand(text: str) -> str:
    """Expand ``${VAR}`` references from the environment.

    Args:
        text: The raw value from a config file or entry point.

    Returns:
        The expanded string.

    Raises:
        PluginConfigError: If a referenced variable is unset — substituting
            an empty value would fail much later, inside a connection error.
    """

    def substitute(match: re.Match) -> str:
        try:
            return os.environ[match.group(1)]
        except KeyError:
            raise PluginConfigError(
                f"plugin config references ${{{match.group(1)}}}, which is not set"
            ) from None

    return _ENV_PATTERN.sub(substitute, text)


@dataclass(frozen=True)
class PluginSpec:
    """Which ``Tesseract.from_*`` call stands behind one plugin name.

    Attributes:
        name: The registry key, e.g. ``"thermal_jaxfem"``.
        kind: The slot it fills, e.g. ``"thermal_solver"``.
        transport: One of :data:`TRANSPORTS`.
        api_path: ``local`` — the package's ``tesseract_api.py``.
        image: ``container`` — the Docker image reference.
        url: ``remote`` — the base URL of a served Tesseract.
        options: Keyword arguments passed straight to the constructor
            (``timeout``, ``environment``, ``volumes``, ``num_workers``, …
            — whatever the installed tesseract-core accepts).
        version: The version the package's ``tesseract_config.yaml``
            declares.  Advisory; the authority is :meth:`Plugin.probe`.
        schema_hash: The expected ``sha256:...`` of the served schema.  When
            set, :meth:`Plugin.probe` refuses an instance that does not
            match — the staleness fence for a long-lived remote.
    """

    name: str
    kind: str
    transport: str = "local"
    api_path: Path | None = None
    image: str | None = None
    url: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    version: str | None = None
    schema_hash: str | None = None

    def __post_init__(self) -> None:
        if self.transport not in TRANSPORTS:
            raise PluginConfigError(
                f"plugin {self.name!r}: transport must be one of {', '.join(TRANSPORTS)} "
                f"(got {self.transport!r})."
            )
        required = _TARGET[self.transport]
        if getattr(self, required) is None:
            raise PluginConfigError(
                f"plugin {self.name!r}: transport {self.transport!r} needs {required!r}."
            )

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> PluginSpec:
        """Build a spec from a ``plugins.toml`` table or an entry point.

        Args:
            name: The registry key (the table name).
            data: The table's contents.

        Returns:
            The spec, with ``${VAR}`` references expanded.

        Raises:
            PluginConfigError: On an unknown key, a missing ``kind``, or a
                transport whose target is absent.
        """
        known = {"kind", "transport", *_TARGET.values(), "options", "version", "schema_hash"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise PluginConfigError(
                f"plugin {name!r}: unknown key(s) {unknown}; known keys are {sorted(known)}."
            )
        if not data.get("kind"):
            raise PluginConfigError(f"plugin {name!r}: 'kind' is required.")
        text = {
            key: expand(str(data[key])) for key in ("api_path", "image", "url") if data.get(key)
        }
        return cls(
            name=name,
            kind=str(data["kind"]),
            transport=str(data.get("transport", "local")),
            api_path=Path(text["api_path"]).expanduser() if "api_path" in text else None,
            image=text.get("image"),
            url=text.get("url"),
            options=dict(data.get("options") or {}),
            version=str(data["version"]) if data.get("version") else None,
            schema_hash=str(data["schema_hash"]) if data.get("schema_hash") else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        """The spec as a ``plugins.toml`` table."""
        table: dict[str, Any] = {"kind": self.kind, "transport": self.transport}
        for key in (*_TARGET.values(), "version", "schema_hash"):
            value = getattr(self, key)
            if value is not None:
                table[key] = str(value) if isinstance(value, Path) else value
        if self.options:
            table["options"] = dict(self.options)
        return table

    def at(self, **changes: Any) -> PluginSpec:
        """A copy of this spec with ``changes`` applied (e.g. a new ``url``)."""
        return replace(self, **changes)

    def open(self) -> tuple[Any, bool]:
        """Make the ``Tesseract.from_*`` call this spec names.

        Returns:
            ``(tesseract, spawned)`` — the ``tesseract_core.Tesseract`` and
            whether this call started a container the caller must tear down.

        Raises:
            ImportError: Without the ``tesseract`` extra.
            PluginConfigError: If a ``local`` spec's ``api_path`` is absent.
        """
        try:
            from tesseract_core import Tesseract
        except ImportError as error:  # pragma: no cover - needs an extra-less env
            raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error

        options = dict(self.options)
        if self.transport == "local":
            path = Path(self.api_path)  # type: ignore[arg-type]
            if not path.is_file():
                raise PluginConfigError(
                    f"plugin {self.name!r}: api_path {str(path)!r} does not exist."
                )
            # Keep the runtime's per-call run_<uuid>/ scratch out of the
            # working directory (and out of an unmanaged mkdtemp).
            options.setdefault("output_path", runtime_scratch())
            return Tesseract.from_tesseract_api(str(path), **options), False
        if self.transport == "container":
            tesseract = Tesseract.from_image(self.image, **options)
            tesseract.serve()
            return tesseract, True
        return Tesseract.from_url(self.url, **options), False
