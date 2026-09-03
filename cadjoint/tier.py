"""The two tiers: is ``diff-brep`` installed, and what to say when it is not.

cadjoint is the public tier: dual contouring, the lattice feature
classifier, faceted STEP/OBJ/STL, TetGen and Gmsh tet meshes, every solver.
``diff-brep`` is the private tier: everything that solves against the patch
fields with a derivative — the derived B-rep, the projection kernel and
its implicit-function adjoint, analytic STEP, the drag inverse problem,
and the map from design parameters to the node positions of a Gmsh mesh.
It fills five plugin kinds (:data:`KINDS`) through the ``python`` transport
of :mod:`cadjoint.plugins`; ``import diff_brep`` is never written in this
tree.

This module is the **one place** the public tier spells out what is
missing.  Everything else asks :func:`available`, :func:`require` or
:func:`component` and either degrades — lattice edges, a faceted STEP, a
frozen Gmsh mesh — or raises :class:`TierUnavailable` with the sentence a
public user should read.  The viewer shows :func:`status`.

Without the private tier the app compiles, meshes (TetGen or Gmsh),
solves, inspects, exports VTK and faceted STEP, and draws lattice feature
edges.  What it cannot do is written in :func:`message`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from cadjoint.enums import PluginKind
from cadjoint.plugins.contracts import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "EDGES_LATTICE_NOTE",
    "GEOMETRY_FROZEN_NOTE",
    "KINDS",
    "PROVIDER",
    "KindStatus",
    "TierStatus",
    "TierUnavailable",
    "absent",
    "available",
    "component",
    "message",
    "report",
    "require",
    "status",
]

#: The distribution that fills the private kinds.
PROVIDER = "diff-brep"

#: The plugin kinds the private tier provides, in the memo's order.
KINDS: tuple[str, ...] = (
    PluginKind.NODE_MAP.value,
    PluginKind.FEATURE_EDGES.value,
    PluginKind.BREP.value,
    PluginKind.STEP_EXPORT.value,
    PluginKind.DRAG.value,
)

#: What the viewer's title block prints under the mesh line without the tier.
EDGES_LATTICE_NOTE = "EDGES LATTICE · DIFF-BREP NOT INSTALLED"
#: What the Meshes window shows for a Gmsh mesh without the tier.
GEOMETRY_FROZEN_NOTE = "GEOMETRY FROZEN · DIFF-BREP NOT INSTALLED"

#: The one sentence per kind, written for a public user.
_MESSAGES: dict[str, str] = {
    PluginKind.NODE_MAP.value: (
        "node positions of a Gmsh mesh do not follow the design without diff-brep: "
        "the mesh is frozen geometry. Use mesher='tetgen' for a differentiable mesh, "
        "or install diff-brep."
    ),
    PluginKind.FEATURE_EDGES.value: (
        "feature edges are read off the lattice without diff-brep; the exact edge "
        "curves of the derived B-rep are the private tier's."
    ),
    PluginKind.BREP.value: (
        "the derived B-rep is the private tier's; without diff-brep there is no B-rep "
        "object, only the dual-contour mesh."
    ),
    PluginKind.STEP_EXPORT.value: (
        "STEP export is faceted without diff-brep; analytic planes and cylinders from "
        "the derived B-rep are the private tier's."
    ),
    PluginKind.DRAG.value: (
        "dragging a B-rep handle solves the design through the derived B-rep, which "
        "is the private tier's; install diff-brep."
    ),
}


class TierUnavailable(RuntimeError):
    """A private-tier kind was needed and is not filled (or not compatible).

    Attributes:
        kind: The plugin kind asked for.
        reason: Why it is unavailable — not registered, failed to import,
            or built for another contract version.
    """

    def __init__(self, kind: str, reason: str):
        self.kind = str(kind)
        self.reason = reason
        super().__init__(message(kind, reason))


@dataclass(frozen=True)
class KindStatus:
    """One private kind's standing in this process.

    Attributes:
        kind: The plugin kind.
        available: Filled by a plugin that imports and speaks this
            contract version.
        plugin: The registry name filling it, or ``None``.
        version: What the component reports, or ``None``.
        contract_version: The contract version it declares, or ``None``.
        compatible: Whether that equals :data:`CONTRACT_VERSION` (a
            component declaring none is taken at its word).
        reason: Why it is unavailable, or ``None``.
    """

    kind: str
    available: bool
    plugin: str | None = None
    version: str | None = None
    contract_version: int | None = None
    compatible: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class TierStatus:
    """Every private kind's standing, and the one-word summary.

    Attributes:
        kinds: Per kind, its :class:`KindStatus`.
        installed: Every kind is available.
    """

    kinds: dict[str, KindStatus]
    installed: bool

    def __getitem__(self, kind: str) -> KindStatus:
        return self.kinds[str(kind)]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready: ``{"installed": bool, "provider": ..., "kinds": {...}}``."""
        return {
            "installed": self.installed,
            "provider": PROVIDER,
            "contract_version": CONTRACT_VERSION,
            "kinds": {kind: asdict(entry) for kind, entry in self.kinds.items()},
        }

    def flags(self) -> dict[str, bool]:
        """``{kind: available}`` — the shape the compile payload carries."""
        return {kind: entry.available for kind, entry in self.kinds.items()}


def message(kind: str, reason: str | None = None) -> str:
    """The refusal text for ``kind`` — the one wording the app and library use.

    Args:
        kind: A private kind.
        reason: Appended when the tier is present but unusable (an import
            failure, a contract mismatch), so the user sees why.
    """
    text = _MESSAGES.get(str(kind), f"kind {kind!r} is the private tier's; install {PROVIDER}.")
    if reason and reason != _NOT_REGISTERED:
        return f"{text} ({reason})"
    return text


_NOT_REGISTERED = "not registered"


def _kind_status(kind: str, registry: Any) -> KindStatus:
    """Probe one kind against ``registry`` without raising."""
    from cadjoint.plugins.plugin import PluginMismatch

    try:
        name = registry.default_for(kind)
    except KeyError:
        return KindStatus(kind=kind, available=False, reason=_NOT_REGISTERED)
    try:
        plugin = registry.plugin(name)
        probe = plugin.probe(strict=False)
        declared = getattr(plugin, "contract_version", None)
    except PluginMismatch as error:
        return KindStatus(kind=kind, available=False, plugin=name, reason=str(error))
    except Exception as error:  # noqa: BLE001 - a status report never raises
        return KindStatus(
            kind=kind, available=False, plugin=name, reason=f"{type(error).__name__}: {error}"
        )
    compatible = declared is None or int(declared) == CONTRACT_VERSION
    reason = None
    if not compatible:
        reason = (
            f"{PROVIDER} is installed but built for plugin contract version {declared}; "
            f"this cadjoint speaks version {CONTRACT_VERSION}"
        )
    return KindStatus(
        kind=kind,
        available=compatible and probe.status == "ok",
        plugin=name,
        version=probe.version,
        contract_version=declared,
        compatible=compatible,
        reason=reason,
    )


def status(registry: Any = None) -> TierStatus:
    """Per kind: filled by whom, at what version, and whether it is usable.

    Args:
        registry: A :class:`~cadjoint.plugins.PluginRegistry`; the
            process-wide one by default.

    Returns:
        The :class:`TierStatus`.  Never raises: a kind whose provider fails
        to import reports the failure as its ``reason``.
    """
    if registry is None:
        from cadjoint.plugins import registry as process_registry

        registry = process_registry()
    kinds = {kind: _kind_status(kind, registry) for kind in KINDS}
    return TierStatus(kinds=kinds, installed=all(entry.available for entry in kinds.values()))


def available(kind: str) -> bool:
    """Whether ``kind`` is filled and usable in this process."""
    from cadjoint.plugins import registry

    return _kind_status(str(kind), registry()).available


def require(kind: str) -> Any:
    """The plugin filling ``kind``, or :class:`TierUnavailable`.

    Also checks the component's ``contract_version`` against
    :data:`CONTRACT_VERSION`, so a stale private build is refused with a
    sentence rather than failing inside a trace.

    Returns:
        The :class:`~cadjoint.plugins.Plugin` (a
        :class:`~cadjoint.plugins.PythonPlugin` for the private kinds).

    Raises:
        TierUnavailable: When the kind is not registered, its provider
            cannot be imported, or it was built for another contract.
    """
    from cadjoint.plugins import registry

    current = registry()
    entry = _kind_status(str(kind), current)
    if not entry.available:
        raise TierUnavailable(kind, entry.reason or "unavailable")
    return current.plugin(entry.plugin)


def component(kind: str, *, default: Any = None) -> Any:
    """The imported object filling ``kind``, or ``default`` when it is absent.

    The degrade-gracefully form of :func:`require`: the overlay and the
    exporter ask for the object and fall back when they get ``None``.
    """
    try:
        return require(kind).component
    except TierUnavailable:
        return default


def report() -> dict[str, Any]:
    """Versions and tier status, JSON-ready — what a public bug report carries."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for distribution in ("cadjoint", PROVIDER, "jax", "numpy", "gmsh", "tetgen", "tesseract-core"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return {"versions": versions, "tier": status().as_dict()}


@contextlib.contextmanager
def absent(kinds: Any = KINDS) -> Iterator[None]:
    """Run a block with the private kinds unregistered process-wide.

    Installs a copy of the current registry without ``kinds`` and restores
    the original afterwards.  What the degradation tests use to exercise
    the public tier's behaviour with the in-tree providers still present.

    Args:
        kinds: The kinds to hide; all five by default.
    """
    from cadjoint.plugins import registry, set_registry

    reduced = registry().without(kinds)
    previous = set_registry(reduced)
    try:
        yield
    finally:
        set_registry(previous)
        reduced.close()
