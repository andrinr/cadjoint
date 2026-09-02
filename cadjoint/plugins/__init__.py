"""Plugins: which Tesseract plays which cadjoint role, and where it runs.

A **plugin** is a component cadjoint calls but does not implement — a
mesher, a tet filler, a solver — reached through ``apply`` and ``vjp``.
Tesseract is how every plugin shipped today is *implemented*; it is not
what a plugin *is*.  This package is the thin layer that makes that
distinction usable: the caller
(:mod:`cadjoint.fem.tesseracts.chain`, :mod:`cadjoint.fem.backends`,
:mod:`cadjoint.optimize`) asks for "whatever fills the ``thermal_solver``
slot" and never learns whether the answer runs in this process, in a
container, or on a cluster.

**What this package does not do.**  It does not encode arrays, speak HTTP,
declare schemas, or implement a JAX primitive.  Each transport is one
``Tesseract.from_*`` call; ``inputs``/``outputs``/differentiability come
from the schema ``tesseract-runtime`` publishes; the operations are the
``Tesseract`` object's methods; :meth:`Plugin.as_jax` is
``tesseract_jax.apply_tesseract`` bound to a client; :meth:`Plugin.probe`
calls the runtime's ``/health`` and hashes the schema it already returns.
What is cadjoint's is the *kind* vocabulary, the registry, the config, and
one piece of housekeeping the runtime leaves open: where its per-call
``run_<uuid>/`` scratch lands (:func:`runtime_scratch`).

* :class:`Plugin` — the interface, with :class:`TesseractPlugin` the
  implementation.
* :class:`PluginSpec` — *where* it runs: ``local`` (in-process),
  ``container`` (Docker image), ``remote`` (a URL — a ``tesseract serve``
  process, or a Kubernetes Service).
* :class:`PluginRegistry` — discovery from the built-in packages, from
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

from cadjoint.plugins.plugin import (
    Capabilities,
    Plugin,
    PluginMismatch,
    PluginProbe,
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
    "ENTRY_POINT_GROUP",
    "KINDS",
    "SCRATCH_ENV",
    "SERVED_OUTPUT_ENV",
    "TRANSPORTS",
    "Capabilities",
    "Plugin",
    "PluginConfigError",
    "PluginMismatch",
    "PluginProbe",
    "PluginRegistry",
    "PluginSpec",
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
