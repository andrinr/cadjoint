"""What a plugin *is*: a Tesseract, plus the cadjoint kind it fills.

A plugin is a component cadjoint calls but does not implement — a mesher, a
tet filler, a solver — reached through the two operations everything
composes on, ``apply`` and ``vjp``, and declaring what it takes and returns.

**Everything here delegates.**  The operations are the ``Tesseract``
object's own methods; the JAX bridge is ``tesseract_jax.apply_tesseract``
and its ``custom_vjp``; the declaration is the schema ``tesseract-runtime``
generates and publishes (including which inputs carry ``Differentiable``);
health is the runtime's ``/health``.  This module adds a *name*, a *kind*,
and a readiness check — nothing a Tesseract already does.
:meth:`TesseractPlugin.__getattr__` forwards everything else (``jacobian``,
``jacobian_vector_product``, ``abstract_eval``, ``test``, ``server_logs``)
straight to the client, so nothing has to be re-exposed here as it is added
upstream.

:class:`Plugin` is the protocol and :class:`TesseractPlugin` the one
implementation, because Tesseract is how plugins are implemented today, not
what a plugin is.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from cadjoint.plugins.spec import PluginSpec

_TESSERACT_EXTRA_MESSAGE = (
    "tesseract-core / tesseract-jax are not installed. "
    "Install the 'tesseract' extra: pip install cadjoint[tesseract]."
)

#: The runtime's own schema sections: the concrete wire models, and the
#: abstract ones carrying ``differentiable_arrays`` and ``field_order``.
_SCHEMA_KEYS = ("Apply_InputSchema", "Apply_OutputSchema", "ApplyInputSchema", "ApplyOutputSchema")


class PluginMismatch(RuntimeError):
    """The running component is not the one the configuration expected.

    Raised by :meth:`Plugin.probe` when the served schema hash (or version)
    differs from the spec's — the case a long-lived remote makes real: the
    cluster was redeployed, and the arrays about to be sent no longer mean
    what they did.
    """


@dataclass(frozen=True)
class Capabilities:
    """What a plugin can do, read off what it declares.

    Nothing here is hand-maintained.  The first three fields are the
    runtime's own declarations; ``features`` names them against cadjoint's
    array ABI so a caller can ask a question instead of matching strings.

    Attributes:
        differentiable_inputs: Inputs declared ``Differentiable[...]``.
        differentiable_outputs: Outputs declared ``Differentiable[...]``.
        endpoints: The endpoints the instance serves.
        features: Flags — ``differentiable``; ``vjp``/``jvp``/``jacobian``/
            ``abstract_eval`` per served endpoint; ``per_element_properties``
            for a floating-point ``cell_*`` input (a heterogeneous material
            field); ``body_force``; ``frozen_topology`` for a
            ``cell_template`` input.
    """

    differentiable_inputs: frozenset[str]
    differentiable_outputs: frozenset[str]
    endpoints: frozenset[str]
    features: frozenset[str]

    def supports(self, feature: str) -> bool:
        """Whether ``feature`` is present."""
        return feature in self.features


@dataclass(frozen=True)
class PluginProbe:
    """A readiness report for one plugin instance.

    Attributes:
        name: The registry key probed.
        kind: The slot it fills.
        transport: ``local``/``container``/``remote``.
        status: The runtime's own health status, ``"ok"`` when healthy.
        version: What the instance reports, falling back to the spec's
            declared version (``tesseract-runtime`` says ``"unknown"``
            unless ``TESSERACT_VERSION`` is set).
        schema_hash: ``sha256:`` of the served schema.
        endpoints: The endpoints it serves.
    """

    name: str
    kind: str
    transport: str
    status: str
    version: str
    schema_hash: str
    endpoints: frozenset[str]


@runtime_checkable
class Plugin(Protocol):
    """The interface every cadjoint plugin satisfies."""

    name: str
    kind: str

    @property
    def inputs(self) -> Mapping[str, Any]:
        """The declared inputs, as the runtime publishes them."""
        ...

    @property
    def outputs(self) -> Mapping[str, Any]:
        """The declared outputs, as the runtime publishes them."""
        ...

    @property
    def capabilities(self) -> Capabilities:
        """What this plugin declares it can do."""
        ...

    def apply(self, inputs: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Evaluate the component on concrete arrays."""
        ...

    def vjp(
        self,
        inputs: Mapping[str, Any],
        vjp_inputs: Any,
        vjp_outputs: Any,
        cotangent_vector: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Pull a cotangent back onto the requested inputs."""
        ...

    def as_jax(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """A JAX-traceable ``inputs -> outputs`` callable for this plugin."""
        ...

    def probe(self) -> PluginProbe:
        """Check the instance is up and is the one the spec expects."""
        ...


class TesseractPlugin:
    """A cadjoint kind, filled by a Tesseract over one of the transports.

    The client is opened on first use and kept warm for the life of the
    object: a ``local`` plugin imports its ``tesseract_api.py`` once, a
    ``container`` plugin spawns its container once (torn down by
    :meth:`close`), a ``remote`` plugin holds one session.

    Args:
        spec: Which ``Tesseract.from_*`` call stands behind this plugin.
    """

    def __init__(self, spec: PluginSpec):
        self.spec = spec
        self.name = spec.name
        self.kind = spec.kind
        self._tesseract: Any = None
        self._spawned = False
        self._schema: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"TesseractPlugin(name={self.name!r}, kind={self.kind!r}, "
            f"transport={self.spec.transport!r})"
        )

    # ── the client ──────────────────────────────────────────────────────

    @property
    def tesseract(self) -> Any:
        """The underlying ``tesseract_core.Tesseract`` (opened on first use)."""
        if self._tesseract is None:
            self._tesseract, self._spawned = self.spec.open()
        return self._tesseract

    @property
    def client(self) -> Any:
        """Alias of :attr:`tesseract` for callers that speak of the client."""
        return self.tesseract

    def __getattr__(self, name: str) -> Any:
        """Forward anything not defined here to the Tesseract itself.

        ``jacobian``, ``jacobian_vector_product``, ``abstract_eval``,
        ``health``, ``test``, ``server_logs``, ``openapi_schema`` and
        whatever tesseract-core adds next are reachable through a plugin
        without this module growing a wrapper per method.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.tesseract, name)

    def close(self) -> None:
        """Release the instance, tearing down a container this object spawned."""
        tesseract, spawned = self._tesseract, self._spawned
        self._tesseract = None
        self._spawned = False
        self._schema = None
        if tesseract is not None and spawned:
            tesseract.teardown()

    def __enter__(self) -> TesseractPlugin:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── the declaration (all of it the runtime's) ───────────────────────

    @property
    def schema(self) -> dict[str, Any]:
        """The runtime's own schema sections, fetched once and cached."""
        if self._schema is None:
            published = self.tesseract.openapi_schema["components"]["schemas"]
            self._schema = {key: published[key] for key in _SCHEMA_KEYS}
        return self._schema

    @property
    def inputs(self) -> Mapping[str, Any]:
        """Declared ``apply`` inputs — the runtime's own property models."""
        return self.schema["Apply_InputSchema"]["properties"]

    @property
    def outputs(self) -> Mapping[str, Any]:
        """Declared ``apply`` outputs — the runtime's own property models."""
        return self.schema["Apply_OutputSchema"]["properties"]

    @property
    def served_version(self) -> str:
        """The version the running instance advertises (``"unknown"`` if none)."""
        return str(self.tesseract.openapi_schema.get("info", {}).get("version", "unknown"))

    @property
    def version(self) -> str:
        """What the instance reports, else the version its package declares."""
        served = self.served_version
        return served if served not in ("", "unknown") else (self.spec.version or served)

    @property
    def capabilities(self) -> Capabilities:
        """See :class:`Capabilities`."""
        endpoints = frozenset(self.tesseract.available_endpoints)
        named = {
            "vector_jacobian_product": "vjp",
            "jacobian_vector_product": "jvp",
            "jacobian": "jacobian",
            "abstract_eval": "abstract_eval",
        }
        differentiable_inputs = frozenset(self.schema["ApplyInputSchema"]["differentiable_arrays"])
        features = {flag for endpoint, flag in named.items() if endpoint in endpoints}
        if differentiable_inputs:
            features.add("differentiable")
        if any(
            name.startswith("cell_") and "float" in str(model.get("title", ""))
            for name, model in self.inputs.items()
        ):
            features.add("per_element_properties")
        if "body_force" in self.inputs:
            features.add("body_force")
        if "cell_template" in self.inputs:
            features.add("frozen_topology")
        return Capabilities(
            differentiable_inputs=differentiable_inputs,
            differentiable_outputs=frozenset(
                self.schema["ApplyOutputSchema"]["differentiable_arrays"]
            ),
            endpoints=endpoints,
            features=frozenset(features),
        )

    def schema_hash(self) -> str:
        """``sha256:`` of the schema the runtime publishes, canonicalized.

        A hash of the runtime's own document, not of a cadjoint restatement
        of it — and measured identical across the ``local`` and ``remote``
        forms of one package (``tests/plugins/test_transport.py``), which is
        what makes it a usable staleness check for a cluster.
        """
        return (
            "sha256:"
            + hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode("utf-8")).hexdigest()
        )

    # ── the two operations (both the Tesseract's) ───────────────────────

    def apply(self, inputs: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """``Tesseract.apply``, callable with a mapping or with keywords.

        Args:
            inputs: The payload, or ``None`` to build it from ``kwargs``.
            **kwargs: Payload entries, when ``inputs`` is not given.

        Returns:
            The outputs the component declares.
        """
        return self.tesseract.apply({**(inputs or {}), **kwargs})

    def vjp(
        self,
        inputs: Mapping[str, Any],
        vjp_inputs: Any,
        vjp_outputs: Any,
        cotangent_vector: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``Tesseract.vector_jacobian_product``, with list-coerced names."""
        return self.tesseract.vector_jacobian_product(
            dict(inputs),
            vjp_inputs=list(vjp_inputs),
            vjp_outputs=list(vjp_outputs),
            cotangent_vector=dict(cotangent_vector),
        )

    def as_jax(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """``tesseract_jax.apply_tesseract`` bound to this plugin's client.

        That function *is* the JAX primitive — it carries the ``custom_vjp``
        dispatching to the component's ``vector_jacobian_product`` under
        ``jax.grad``.  Nothing is re-implemented here; this only supplies
        the client, so a caller composing a chain never handles one.

        Returns:
            ``inputs -> outputs``.

        Raises:
            ImportError: Without the ``tesseract`` extra.
        """
        try:
            from tesseract_jax import apply_tesseract
        except ImportError as error:
            raise ImportError(_TESSERACT_EXTRA_MESSAGE) from error

        return functools.partial(apply_tesseract, self.tesseract)

    # ── readiness ───────────────────────────────────────────────────────

    def probe(self, *, strict: bool = True) -> PluginProbe:
        """Check the instance is up and is the component the spec expects.

        Args:
            strict: Raise on a schema-hash or version mismatch (the
                default); ``False`` reports it and leaves the judgement to
                the caller.

        Returns:
            The :class:`PluginProbe`.

        Raises:
            PluginMismatch: When ``strict`` and the served schema hash — or
                the version, when both sides state one — differs from the
                spec's.
            RuntimeError: When the instance cannot be reached.
        """
        try:
            status = str(self.tesseract.health().get("status", "unknown"))
        except Exception as error:
            raise RuntimeError(
                f"plugin {self.name!r} ({self.spec.transport}) is unreachable: {error}"
            ) from error
        digest = self.schema_hash()
        probe = PluginProbe(
            name=self.name,
            kind=self.kind,
            transport=self.spec.transport,
            status=status,
            version=self.version,
            schema_hash=digest,
            endpoints=frozenset(self.tesseract.available_endpoints),
        )
        if not strict:
            return probe
        if self.spec.schema_hash and self.spec.schema_hash != digest:
            target = self.spec.url or self.spec.image or self.spec.api_path
            raise PluginMismatch(
                f"plugin {self.name!r} ({self.spec.transport}) serves schema {digest}, but the "
                f"configuration expects {self.spec.schema_hash}. The component behind {target} "
                "has changed its interface; update the spec's schema_hash once you have "
                "checked the new one."
            )
        served = self.served_version
        if self.spec.version and served not in ("", "unknown") and served != self.spec.version:
            raise PluginMismatch(
                f"plugin {self.name!r} serves version {served!r}, configuration expects "
                f"{self.spec.version!r}."
            )
        return probe
