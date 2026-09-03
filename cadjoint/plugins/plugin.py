"""What a plugin *is*: a component filling a cadjoint kind, over one transport.

A plugin is a component cadjoint calls but does not implement — a mesher, a
tet filler, a solver, the private tier's node map — reached through the two
operations everything composes on, ``apply`` and ``vjp``, and declaring
what it takes and returns.

**Everything here delegates.**  For a :class:`TesseractPlugin` the
operations are the ``Tesseract`` object's own methods; the JAX bridge is
``tesseract_jax.apply_tesseract`` and its ``custom_vjp``; the declaration is
the schema ``tesseract-runtime`` generates and publishes (including which
inputs carry ``Differentiable``); health is the runtime's ``/health``.  This
module adds a *name*, a *kind*, and a readiness check — nothing a Tesseract
already does.  :meth:`TesseractPlugin.__getattr__` forwards everything else
(``jacobian``, ``jacobian_vector_product``, ``abstract_eval``, ``test``,
``server_logs``) straight to the client, so nothing has to be re-exposed
here as it is added upstream.

For a :class:`PythonPlugin` the component is an object imported into this
process that satisfies one of the Protocols in
:mod:`cadjoint.plugins.contracts`: the operations are its own methods, the
JAX callable is the method itself (with whatever ``custom_vjp`` it
carries), and the declaration is read off the contract's annotations.  A
Tesseract round trip costs ~10 ms and cannot sit inside ``vmap`` or a
trace; a component called per compile or per traced step crosses the seam
this way instead.

:class:`Plugin` is the protocol; :class:`TesseractPlugin` and
:class:`PythonPlugin` are its two implementations — one contract, two
transports.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from cadjoint.plugins.contracts import (
    CONTRACT_VERSION,
    contract_for,
    contract_signature,
    payload_types,
    primary_method,
)
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
        transport: ``local``/``container``/``remote``/``python``.
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


class PythonPlugin:
    """A cadjoint kind, filled by an object imported into this process.

    The object is whatever ``spec.object`` (``"module:attribute"``) names,
    imported on first use and held for the life of the plugin.  It must
    satisfy the kind's Protocol in :mod:`cadjoint.plugins.contracts`; a
    kind with no contract (a third party's own) is accepted as is, and its
    ``apply`` calls the object itself.

    Args:
        spec: A ``transport="python"`` spec.
    """

    def __init__(self, spec: PluginSpec):
        self.spec = spec
        self.name = spec.name
        self.kind = spec.kind
        self._component: Any = None

    def __repr__(self) -> str:
        return f"PythonPlugin(name={self.name!r}, kind={self.kind!r}, object={self.spec.object!r})"

    # ── the component ───────────────────────────────────────────────────

    @property
    def component(self) -> Any:
        """The imported object (resolved on first use).

        Raises:
            PluginMismatch: If it does not satisfy the kind's Protocol.
        """
        if self._component is None:
            component, _spawned = self.spec.open()
            contract = contract_for(self.kind)
            if contract is not None and not isinstance(component, contract):
                missing = sorted(
                    name
                    for name in getattr(contract, "__protocol_attrs__", ())
                    if not hasattr(component, name)
                )
                raise PluginMismatch(
                    f"plugin {self.name!r} ({self.spec.object}) does not satisfy the "
                    f"{contract.__name__} contract for kind {self.kind!r}; missing {missing}."
                )
            self._component = component
        return self._component

    @property
    def method(self) -> Callable[..., Any]:
        """The bound method the kind's contract names (the object itself otherwise)."""
        name = primary_method(self.kind)
        component = self.component
        return getattr(component, name) if name else component

    def close(self) -> None:
        """Drop the reference; the next use imports again."""
        self._component = None

    def __enter__(self) -> PythonPlugin:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── the declaration (the contract's) ────────────────────────────────

    @property
    def inputs(self) -> Mapping[str, Any]:
        """The primary method's parameters, with their ``Differentiable`` tags."""
        return payload_types(self.kind)[0]

    @property
    def outputs(self) -> Mapping[str, Any]:
        """The primary method's result, under the key ``"result"``."""
        _inputs, output = payload_types(self.kind)
        return {"result": output} if output else {}

    @property
    def version(self) -> str:
        """The object's own ``version``, else the spec's, else ``"unknown"``."""
        declared = getattr(self.component, "version", None)
        return str(declared) if declared else (self.spec.version or "unknown")

    @property
    def contract_version(self) -> int | None:
        """The contract version the object declares (``None`` if it declares none)."""
        declared = getattr(self.component, "contract_version", None)
        return int(declared) if declared is not None else None

    @property
    def capabilities(self) -> Capabilities:
        """See :class:`Capabilities`; ``in_process`` is always among the features."""
        inputs, output = payload_types(self.kind)
        differentiable_inputs = frozenset(
            name for name, entry in inputs.items() if entry["differentiable"]
        )
        differentiable_outputs = frozenset(
            ["result"] if output and output["differentiable"] else []
        )
        endpoints = {"apply"}
        features = {"in_process"}
        if differentiable_inputs or differentiable_outputs:
            endpoints.add("vector_jacobian_product")
            features.update({"differentiable", "vjp"})
        return Capabilities(
            differentiable_inputs=differentiable_inputs,
            differentiable_outputs=differentiable_outputs,
            endpoints=frozenset(endpoints),
            features=frozenset(features),
        )

    def schema_hash(self) -> str:
        """``sha256:`` of the kind's contract signature (see :func:`contract_signature`)."""
        return contract_signature(self.kind)

    # ── the two operations (both the object's) ──────────────────────────

    def apply(self, inputs: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Call the contract's method with the payload as keywords.

        Returns:
            ``{"result": value}`` — one key, because a contract method
            returns one object.
        """
        return {"result": self.method(**{**(inputs or {}), **kwargs})}

    def vjp(
        self,
        inputs: Mapping[str, Any],
        vjp_inputs: Any,
        vjp_outputs: Any,
        cotangent_vector: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``jax.vjp`` of the contract's method with respect to the named inputs.

        The derivative is the object's own (its ``custom_vjp``); this only
        arranges the call.

        Raises:
            ValueError: For a kind whose contract carries no derivative, or
                an output other than ``"result"``.
        """
        import jax

        names = list(vjp_inputs)
        if not self.capabilities.supports("differentiable"):
            raise ValueError(f"plugin {self.name!r} (kind {self.kind!r}) carries no derivative.")
        if list(vjp_outputs) != ["result"]:
            raise ValueError(f"Only 'result' carries a vjp; requested: {sorted(vjp_outputs)}")
        payload = dict(inputs)

        def forward(*values: Any) -> Any:
            return self.method(**{**payload, **dict(zip(names, values))})

        _value, pull_back = jax.vjp(forward, *[payload[name] for name in names])
        return dict(zip(names, pull_back(cotangent_vector["result"])))

    def as_jax(self) -> Callable[..., Any]:
        """The object's own JAX callable — the contract's bound method.

        Unlike :meth:`TesseractPlugin.as_jax` nothing is wrapped: the
        method carries its own ``custom_vjp`` and is called with keywords
        inside the caller's trace.
        """
        return self.method

    # ── readiness ───────────────────────────────────────────────────────

    def probe(self, *, strict: bool = True) -> PluginProbe:
        """Import the object and check it speaks this cadjoint's contract.

        Args:
            strict: Raise on a contract-version or version mismatch (the
                default); ``False`` reports and leaves the judgement to the
                caller.

        Returns:
            The :class:`PluginProbe`; ``status`` is ``"ok"`` when the
            object imports and satisfies the Protocol.

        Raises:
            PluginMismatch: When ``strict`` and the object declares a
                contract version other than :data:`CONTRACT_VERSION`, or a
                version other than the spec's.
            RuntimeError: When the object cannot be imported.
        """
        try:
            component = self.component
        except PluginMismatch:
            raise
        except Exception as error:
            raise RuntimeError(
                f"plugin {self.name!r} (python) cannot be imported: {error}"
            ) from error
        probe = PluginProbe(
            name=self.name,
            kind=self.kind,
            transport=self.spec.transport,
            status="ok",
            version=self.version,
            schema_hash=self.schema_hash(),
            endpoints=self.capabilities.endpoints,
        )
        if not strict:
            return probe
        declared = getattr(component, "contract_version", None)
        if declared is not None and int(declared) != CONTRACT_VERSION:
            raise PluginMismatch(
                f"plugin {self.name!r} ({self.spec.object}) was built for plugin contract "
                f"version {declared}; this cadjoint speaks version {CONTRACT_VERSION}."
            )
        if (
            self.spec.version
            and self.version not in ("", "unknown")
            and self.version != self.spec.version
        ):
            raise PluginMismatch(
                f"plugin {self.name!r} reports version {self.version!r}, configuration expects "
                f"{self.spec.version!r}."
            )
        return probe
