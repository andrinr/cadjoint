"""Plugins: which component plays which cadjoint role, and where it runs.

A **plugin** is a component cadjoint calls but does not implement — a
mesher, a tet filler, a solver, the private tier's node map — reached
through ``apply`` and ``vjp``.  Tesseract is how the coarse ones are
*implemented*; it is not what a plugin *is*.  This package is the thin
layer that makes that distinction usable: the caller
(:mod:`cadjoint.fem.tesseracts.chain`, :mod:`cadjoint.fem.backends`,
:mod:`cadjoint.optimize`, the viewer) asks for "whatever fills the
``thermal_solver`` slot" and never learns whether the answer runs in this
process, in a container, or on a cluster.

**One contract, two transports.**  A component called once per optimizer
step or once per job is Tesseract grain: a served round trip costs ~10 ms,
1–2 % of a step.  A component called inside a trace or once per compile —
a Newton kernel under ``vmap``, a curve tracer, the viewer's overlay — is
not, so it crosses as a Python object in this process
(:class:`PythonPlugin`, the ``python`` transport) against a typed Protocol
in :mod:`cadjoint.plugins.contracts`.  The private tier ``diff-brep`` fills
its five kinds that way; :mod:`cadjoint.tier` is where its absence is
reported.

**What this package does not do.**  It does not encode arrays, speak HTTP,
declare schemas, or implement a JAX primitive.  Each Tesseract transport is
one ``Tesseract.from_*`` call; ``inputs``/``outputs``/differentiability
come from the schema ``tesseract-runtime`` publishes; the operations are
the ``Tesseract`` object's methods; :meth:`Plugin.as_jax` is
``tesseract_jax.apply_tesseract`` bound to a client; :meth:`Plugin.probe`
calls the runtime's ``/health`` and hashes the schema it already returns.
What is cadjoint's is the *kind* vocabulary, the contracts, the registry,
the config, and one piece of housekeeping the runtime leaves open: where
its per-call ``run_<uuid>/`` scratch lands (:func:`runtime_scratch`).

* :class:`Plugin` — the interface, with :class:`TesseractPlugin` and
  :class:`PythonPlugin` the implementations.
* :class:`PluginSpec` — *where* it runs: ``local`` (a Tesseract in this
  process), ``container`` (Docker image), ``remote`` (a URL — a
  ``tesseract serve`` process, or a Kubernetes Service), ``python`` (an
  importable object, no Tesseract runtime).
* :class:`PluginRegistry` — discovery from the built-ins, from
  ``cadjoint.plugins`` entry points, and from ``plugins.toml``.

Moving a solver onto a cluster::

    # ~/.config/cadjoint/plugins.toml
    [plugins.thermal_jaxfem]
    kind = "thermal_solver"
    transport = "remote"
    url = "http://thermal.cadjoint.svc.cluster.local:8000"
    schema_hash = "sha256:2a1f..."

Nothing in cadjoint changes; the next ``gradient_path="tesseract-dc"`` run
sends its solves over HTTP, and :meth:`Plugin.probe` refuses up front if the
deployed component's schema no longer matches.  See ``docs/plugins.qmd``.
"""

from __future__ import annotations

from cadjoint.plugins.contracts import (
    CONTRACT_VERSION,
    KIND_CONTRACTS,
    BRepExtractor,
    Differentiable,
    Drag,
    DragOutcome,
    EdgeSet,
    FeatureEdges,
    NodeMap,
    OwnedNodes,
    StepExporter,
)
from cadjoint.plugins.plugin import (
    Capabilities,
    Plugin,
    PluginMismatch,
    PluginProbe,
    PythonPlugin,
    TesseractPlugin,
)
from cadjoint.plugins.registry import (
    BUILTIN_DEFAULTS,
    BUILTIN_PACKAGES,
    CONFIG_ENV,
    CONFIG_NAME,
    ENTRY_POINT_GROUP,
    KINDS,
    PluginRegistry,
    available_plugins,
    build_registry,
    builtin_specs,
    config_path,
    entry_point_specs,
    get_plugin,
    load_config,
    parse_config,
    plugin_for_kind,
    register_plugin,
    registry,
    set_registry,
)
from cadjoint.plugins.spec import (
    SCRATCH_ENV,
    SERVED_OUTPUT_ENV,
    TRANSPORTS,
    PluginConfigError,
    PluginSpec,
    runtime_scratch,
    scratch_root,
)

__all__ = [
    "BUILTIN_DEFAULTS",
    "BUILTIN_PACKAGES",
    "CONFIG_ENV",
    "CONFIG_NAME",
    "CONTRACT_VERSION",
    "ENTRY_POINT_GROUP",
    "KINDS",
    "KIND_CONTRACTS",
    "SCRATCH_ENV",
    "SERVED_OUTPUT_ENV",
    "TRANSPORTS",
    "BRepExtractor",
    "Capabilities",
    "Differentiable",
    "Drag",
    "DragOutcome",
    "EdgeSet",
    "FeatureEdges",
    "NodeMap",
    "OwnedNodes",
    "Plugin",
    "PluginConfigError",
    "PluginMismatch",
    "PluginProbe",
    "PluginRegistry",
    "PluginSpec",
    "PythonPlugin",
    "StepExporter",
    "TesseractPlugin",
    "available_plugins",
    "build_registry",
    "builtin_specs",
    "config_path",
    "entry_point_specs",
    "get_plugin",
    "load_config",
    "parse_config",
    "plugin_for_kind",
    "register_plugin",
    "registry",
    "runtime_scratch",
    "scratch_root",
    "set_registry",
]
