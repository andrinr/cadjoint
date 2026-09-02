"""Which Tesseract plays which cadjoint kind, and where it runs.

This is the part that is genuinely cadjoint's: a name -> spec table, a
kind -> name default, and three places a spec can come from, later ones
winning:

1. **Built-in packages** — the Tesseract packages in this repository
   (:data:`BUILTIN_PACKAGES`), as ``local`` specs.
2. **Entry points** — the ``cadjoint.plugins`` group.  A distribution
   registers a component by exposing a :class:`~cadjoint.plugins.PluginSpec`,
   a mapping in the ``plugins.toml`` table form, or a callable returning
   either.
3. **The config file** — ``plugins.toml``, found at ``$CADJOINT_PLUGINS``
   (a file, or a directory holding one) or else
   ``$XDG_CONFIG_HOME/cadjoint/plugins.toml`` (default
   ``~/.config/cadjoint/plugins.toml``).  A table whose name matches an
   existing plugin **replaces** its spec — that is how the thermal solver
   moves from this process to a cluster URL without a line of code
   changing.

The file has two tables::

    [defaults]                       # which plugin fills each slot
    elastic_solver = "elastic_calculix"

    [plugins.thermal_jaxfem]         # where a plugin runs
    kind = "thermal_solver"
    transport = "remote"
    url = "http://thermal.cadjoint.svc.cluster.local:8000"
    schema_hash = "sha256:2a1f..."

Callers ask by name (:func:`get_plugin`) for one specific component, or by
kind (:func:`plugin_for_kind`) for whatever currently fills a slot — which
is what :mod:`cadjoint.fem.tesseracts.chain` and
:mod:`cadjoint.fem.backends` do, so neither imports a ``tesseract_api``
module.  Instances are cached, so a ``local`` plugin imports its API module
once and a ``remote`` one holds a single warm session.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cadjoint.plugins.plugin import Plugin, TesseractPlugin
from cadjoint.plugins.spec import PluginConfigError, PluginSpec

#: The entry-point group third-party distributions register under.
ENTRY_POINT_GROUP = "cadjoint.plugins"
#: Environment variable naming a config file (or a directory holding one).
CONFIG_ENV = "CADJOINT_PLUGINS"
#: The file name looked for in a config directory.
CONFIG_NAME = "plugins.toml"

#: The slots cadjoint itself resolves by kind.  A plugin may declare any
#: kind; these are the ones the pipeline asks for.
KINDS = ("mesher", "tetfill", "thermal_solver", "elastic_solver", "qef", "flow_solver")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESSERACTS = Path(__file__).resolve().parents[1] / "fem" / "tesseracts"

#: name -> (kind, package directory) for the packages shipped in this repo.
#: The ``native`` QEF core lives outside the Python package, so it registers
#: only when the checkout it belongs to is present.
BUILTIN_PACKAGES: dict[str, tuple[str, Path]] = {
    "mesher": ("mesher", _TESSERACTS / "mesher"),
    "tetfill": ("tetfill", _TESSERACTS / "tetfill"),
    "thermal_jaxfem": ("thermal_solver", _TESSERACTS / "thermal_jaxfem"),
    "elastic_jaxfem": ("elastic_solver", _TESSERACTS / "elastic_jaxfem"),
    "elastic_calculix": ("elastic_solver", _TESSERACTS / "elastic_calculix"),
    "flow_brinkman": ("flow_solver", _TESSERACTS / "flow_brinkman"),
    "qef_native": ("qef", _REPO_ROOT / "native"),
}

#: Which plugin fills each kind unless the config says otherwise.  The two
#: elastic solvers share a kind, so this is where that choice is made.
BUILTIN_DEFAULTS: dict[str, str] = {
    "mesher": "mesher",
    "tetfill": "tetfill",
    "thermal_solver": "thermal_jaxfem",
    "elastic_solver": "elastic_jaxfem",
    "flow_solver": "flow_brinkman",
    "qef": "qef_native",
}

_VERSION_LINE = re.compile(r"^version:\s*['\"]?([^'\"\s]+)['\"]?\s*$", re.MULTILINE)


# ── the config file ─────────────────────────────────────────────────────


def config_path() -> Path | None:
    """The ``plugins.toml`` in effect, or ``None`` when there is none.

    Returns:
        The resolved path.

    Raises:
        PluginConfigError: If ``$CADJOINT_PLUGINS`` names something
            unreadable — silently falling back to the built-ins would hide
            a deployment bug.
    """
    override = os.environ.get(CONFIG_ENV)
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            path = path / CONFIG_NAME
        if not path.is_file():
            raise PluginConfigError(
                f"${CONFIG_ENV} points at {override!r}, which is not a readable {CONFIG_NAME} file."
            )
        return path
    base = os.environ.get("XDG_CONFIG_HOME")
    home = Path(base).expanduser() if base else Path.home() / ".config"
    path = home / "cadjoint" / CONFIG_NAME
    return path if path.is_file() else None


def parse_config(document: Mapping[str, Any], *, source: str) -> tuple[dict, dict]:
    """Split a parsed config document into specs and per-kind defaults.

    Args:
        document: The parsed TOML mapping.
        source: What to name in error messages (a path, usually).

    Returns:
        ``(specs, defaults)`` — ``{name: PluginSpec}`` and
        ``{kind: plugin name}``.

    Raises:
        PluginConfigError: On an unknown top-level table or a malformed
            plugin table.
    """
    unknown = sorted(set(document) - {"plugins", "defaults"})
    if unknown:
        raise PluginConfigError(
            f"{source}: unknown top-level table(s) {unknown}; expected [plugins] and [defaults]."
        )
    specs = {}
    for name, table in (document.get("plugins") or {}).items():
        if not isinstance(table, Mapping):
            raise PluginConfigError(f"{source}: [plugins.{name}] must be a table.")
        specs[str(name)] = PluginSpec.from_mapping(str(name), table)
    defaults = {str(kind): str(name) for kind, name in (document.get("defaults") or {}).items()}
    return specs, defaults


def load_config(path: Path | None = None) -> tuple[dict, dict, Path | None]:
    """Load the plugins config in effect.

    Args:
        path: An explicit file to read; ``None`` resolves via
            :func:`config_path`.

    Returns:
        ``(specs, defaults, path)``; the first two are empty and ``path`` is
        ``None`` when no config file exists.

    Raises:
        PluginConfigError: Before Python 3.11 without ``tomli`` installed.
    """
    resolved = path if path is not None else config_path()
    if resolved is None:
        return {}, {}, None
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as error:  # pragma: no cover
            raise PluginConfigError(
                f"reading {resolved} needs a TOML parser: Python 3.11+ or `pip install tomli`."
            ) from error
    with resolved.open("rb") as handle:
        document = tomllib.load(handle)
    specs, defaults = parse_config(document, source=str(resolved))
    return specs, defaults, resolved


# ── discovery ───────────────────────────────────────────────────────────


def _declared_version(package: Path) -> str | None:
    """The ``version`` a package's ``tesseract_config.yaml`` declares.

    Read with a line regex rather than a YAML parser: this runs at import
    time for every built-in and cadjoint's core must not gain a PyYAML
    dependency for it.  An unreadable file yields ``None`` — the version is
    advisory, and :meth:`Plugin.probe` is the authority.

    Args:
        package: The Tesseract package directory.

    Returns:
        The declared version, or ``None``.
    """
    try:
        text = (package / "tesseract_config.yaml").read_text(encoding="utf-8")
    except OSError:
        return None
    match = _VERSION_LINE.search(text)
    return match.group(1) if match else None


def builtin_specs() -> dict[str, PluginSpec]:
    """Local specs for every Tesseract package present in this checkout."""
    specs = {}
    for name, (kind, package) in BUILTIN_PACKAGES.items():
        api = package / "tesseract_api.py"
        if api.is_file():
            specs[name] = PluginSpec(
                name=name,
                kind=kind,
                transport="local",
                api_path=api,
                version=_declared_version(package),
            )
    return specs


def _coerce(name: str, value: Any) -> PluginSpec:
    """Turn an entry point's object into a :class:`PluginSpec`."""
    if callable(value) and not isinstance(value, PluginSpec):
        value = value()
    if isinstance(value, PluginSpec):
        return value if value.name == name else value.at(name=name)
    if isinstance(value, Mapping):
        return PluginSpec.from_mapping(name, value)
    raise PluginConfigError(
        f"entry point {name!r} in group {ENTRY_POINT_GROUP!r} resolved to "
        f"{type(value).__name__}; expected a PluginSpec, a mapping, or a callable "
        "returning one."
    )


def entry_point_specs() -> dict[str, PluginSpec]:
    """Specs contributed by installed distributions.

    A broken entry point raises a :class:`PluginConfigError` naming it
    rather than being skipped: a plugin that silently fails to register is
    the hardest kind of deployment bug to see.
    """
    from importlib.metadata import entry_points

    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - importlib.metadata before 3.10
        found = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[union-attr]
    specs = {}
    for entry in found:
        try:
            specs[entry.name] = _coerce(entry.name, entry.load())
        except PluginConfigError:
            raise
        except Exception as error:
            raise PluginConfigError(
                f"entry point {entry.name!r} in group {ENTRY_POINT_GROUP!r} failed to load: {error}"
            ) from error
    return specs


# ── the registry ────────────────────────────────────────────────────────


class PluginRegistry:
    """The plugins this process knows about, and the instances it has open.

    Args:
        specs: Name -> spec.
        defaults: Kind -> plugin name.
        source: Where the config came from, for error messages.
    """

    def __init__(
        self,
        specs: Mapping[str, PluginSpec] | None = None,
        defaults: Mapping[str, str] | None = None,
        *,
        source: Path | None = None,
    ):
        self._specs: dict[str, PluginSpec] = dict(specs or {})
        self._defaults: dict[str, str] = dict(defaults or {})
        self._instances: dict[str, Plugin] = {}
        self._lock = threading.Lock()
        self.source = source

    def names(self) -> list[str]:
        """Registered plugin names, sorted."""
        return sorted(self._specs)

    def kinds(self) -> dict[str, list[str]]:
        """Registered plugin names grouped by kind."""
        grouped: dict[str, list[str]] = {}
        for name, spec in sorted(self._specs.items()):
            grouped.setdefault(spec.kind, []).append(name)
        return grouped

    def spec(self, name: str) -> PluginSpec:
        """The spec registered under ``name``.

        Raises:
            KeyError: If nothing is registered under that name.
        """
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"Unknown plugin {name!r}; registered: {self.names()}"
                + (f" (config: {self.source})" if self.source else "")
            ) from None

    def default_for(self, kind: str) -> str:
        """The plugin name filling ``kind``.

        Raises:
            KeyError: If no plugin declares that kind, if several do with no
                default set, or if the default names something unregistered.
        """
        name = self._defaults.get(kind)
        if name is None:
            candidates = self.kinds().get(kind) or []
            if len(candidates) == 1:
                name = candidates[0]
            elif not candidates:
                raise KeyError(
                    f"No plugin registered for kind {kind!r}; registered kinds: "
                    f"{sorted(self.kinds())}"
                )
            else:
                raise KeyError(
                    f"Kind {kind!r} is filled by several plugins ({candidates}) and no "
                    "default is set; add a [defaults] entry to plugins.toml."
                )
        if name not in self._specs:
            raise KeyError(
                f"Kind {kind!r} defaults to plugin {name!r}, which is not registered; "
                f"registered: {self.names()}"
            )
        return name

    def register(self, spec: PluginSpec, *, default: bool | None = None) -> None:
        """Add or replace a spec, closing any open instance under that name.

        Args:
            spec: The spec to register.
            default: Make it the default for its kind (``None``: only when
                nothing else fills that kind yet).
        """
        with self._lock:
            existing = self._instances.pop(spec.name, None)
            self._specs[spec.name] = spec
            if default or (default is None and spec.kind not in self._defaults):
                self._defaults[spec.kind] = spec.name
        if existing is not None:
            existing.close()  # type: ignore[attr-defined]

    def set_default(self, kind: str, name: str) -> None:
        """Point ``kind`` at the plugin registered as ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        self.spec(name)
        self._defaults[kind] = name

    def plugin(self, name: str) -> Plugin:
        """The (cached) plugin instance registered under ``name``."""
        with self._lock:
            instance = self._instances.get(name)
            if instance is None:
                instance = TesseractPlugin(self.spec(name))
                self._instances[name] = instance
            return instance

    def for_kind(self, kind: str) -> Plugin:
        """The (cached) plugin instance currently filling ``kind``."""
        return self.plugin(self.default_for(kind))

    def close(self) -> None:
        """Close every open instance (tearing down spawned containers)."""
        with self._lock:
            instances = list(self._instances.values())
            self._instances.clear()
        for instance in instances:
            instance.close()  # type: ignore[attr-defined]


def build_registry(*, config: Path | None = None, use_entry_points: bool = True) -> PluginRegistry:
    """Assemble a registry from built-ins, entry points and the config file.

    Args:
        config: An explicit ``plugins.toml``; ``None`` resolves it the usual
            way.
        use_entry_points: Include installed ``cadjoint.plugins`` entry
            points (tests turn this off to stay hermetic).

    Returns:
        The assembled registry.
    """
    specs = builtin_specs()
    if use_entry_points:
        specs.update(entry_point_specs())
    file_specs, defaults, source = load_config(config)
    specs.update(file_specs)
    resolved = dict(BUILTIN_DEFAULTS)
    # A configured plugin that is the only one of its kind becomes the
    # default for it, so adding a solver needs no [defaults] entry.
    for name, spec in file_specs.items():
        resolved.setdefault(spec.kind, name)
    resolved.update(defaults)
    return PluginRegistry(specs, {k: v for k, v in resolved.items() if v in specs}, source=source)


_REGISTRY: PluginRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def registry() -> PluginRegistry:
    """The process-wide registry, assembled on first use."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = build_registry()
        return _REGISTRY


def set_registry(new: PluginRegistry | None) -> PluginRegistry | None:
    """Install ``new`` as the process-wide registry, returning the old one.

    ``None`` clears it, so the next :func:`registry` call rebuilds from the
    environment.  Tests use this to point cadjoint at a served plugin
    without touching the user's config.
    """
    global _REGISTRY
    with _REGISTRY_LOCK:
        previous, _REGISTRY = _REGISTRY, new
        return previous


def get_plugin(name: str) -> Plugin:
    """The plugin registered under ``name`` in the process-wide registry."""
    return registry().plugin(name)


def plugin_for_kind(kind: str) -> Plugin:
    """The plugin currently filling ``kind`` in the process-wide registry."""
    return registry().for_kind(kind)


def available_plugins() -> list[str]:
    """Names registered in the process-wide registry."""
    return registry().names()


def register_plugin(spec: PluginSpec, *, default: bool | None = None) -> None:
    """Register ``spec`` in the process-wide registry."""
    registry().register(spec, default=default)
