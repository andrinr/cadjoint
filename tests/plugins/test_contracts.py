"""The seam itself: the five contracts, their payloads, and who fills them.

:mod:`cadjoint.plugins.contracts` is the module the private tier
(``diff-brep``) implements against, so what it promises has to be checked
from the public side alone — no ``import diff_brep`` anywhere, ever.  Three
claims are worth a test.

**Every kind has a contract, and a provider that satisfies it is bound.**
Nothing in this repository fills these five kinds, so the provider here is
``tests/plugins/stubs.py``, which computes nothing: if
:class:`~cadjoint.plugins.PythonPlugin` can import it, resolve the
contract's method and report its capabilities, then the seam is the
interface and not the implementation, and an installed diff-brep is bound
by exactly the same path.

**The payloads round-trip.**  :class:`~cadjoint.plugins.OwnedNodes` is the
one record that crosses in both directions — public
:func:`cadjoint.fem.gmsh.assign_ownership` produces it, the private node map
consumes it — so its invariants (arity counts the filled slots, blends are
exactly the unowned boundary nodes, the midside block matches
``edge_parents``) are checked by construction and its ``to_mapping`` is
checked to survive a rebuild.

**The refusal is one sentence in one place.**  :mod:`cadjoint.tier` is where
"not installed" is spelled; every degraded path quotes it rather than
writing its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint import tier
from cadjoint.enums import PluginKind, PluginTransport
from cadjoint.plugins import (
    CONTRACT_VERSION,
    EdgeSet,
    OwnedNodes,
    PluginSpec,
    PythonPlugin,
    contracts,
    plugin_for_kind,
    register_plugin,
)


def _owned(**overrides) -> OwnedNodes:
    """A four-corner, one-midside record: the smallest legal ``OwnedNodes``."""
    fields = {
        "seeds": np.array(
            [[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0, 0]], dtype=np.float64
        ),
        "patches": np.array(
            [[0, 1, 2], [0, 1, -1], [0, -1, -1], [-1, -1, -1], [0, 1, -1]], dtype=np.int32
        ),
        "arity": np.array([3, 2, 1, 0, 2], dtype=np.int8),
        "entity_dim": np.array([0, 1, 2, 3, 1], dtype=np.int8),
        "blend": np.zeros(5, dtype=bool),
        "midside": np.array([False, False, False, False, True]),
        "edge_parents": np.array([[0, 1]], dtype=np.int32),
        "cells": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "bar": 1e-3,
        "design": {"radius": np.array(0.25)},
    }
    fields.update(overrides)
    return OwnedNodes(**fields)


class TestEveryKindIsUnderContract:
    """The vocabulary: five kinds, five Protocols, one method each."""

    def test_the_tier_kinds_are_the_contracted_kinds(self):
        assert set(tier.KINDS) == set(contracts.KIND_CONTRACTS)
        assert set(tier.KINDS) == {
            PluginKind.NODE_MAP.value,
            PluginKind.FEATURE_EDGES.value,
            PluginKind.BREP.value,
            PluginKind.STEP_EXPORT.value,
            PluginKind.DRAG.value,
        }

    def test_every_kind_names_a_protocol_and_a_method(self):
        for kind in tier.KINDS:
            protocol = contracts.contract_for(kind)
            method = contracts.primary_method(kind)
            assert protocol is not None and method is not None, kind
            assert hasattr(protocol, method)

    def test_only_the_node_map_carries_a_derivative(self):
        """Edges, a B-rep, a STEP file and a drag result are not differentiated."""
        _inputs, output = contracts.payload_types(PluginKind.NODE_MAP.value)
        assert output["differentiable"]
        for kind in tier.KINDS:
            if kind == PluginKind.NODE_MAP.value:
                continue
            assert not contracts.payload_types(kind)[1]["differentiable"], kind

    def test_the_signature_hash_moves_with_the_contract_version(self, monkeypatch):
        """A stale private build is caught by the hash, not inside a trace."""
        before = contracts.contract_signature(PluginKind.NODE_MAP.value)
        monkeypatch.setattr(contracts, "CONTRACT_VERSION", CONTRACT_VERSION + 1)
        assert contracts.contract_signature(PluginKind.NODE_MAP.value) != before

    def test_a_kind_without_a_contract_is_not_one_of_ours(self):
        assert contracts.contract_for(PluginKind.TET_MESHER.value) is None
        assert contracts.payload_types(PluginKind.TET_MESHER.value) == ({}, {})


@pytest.mark.usefixtures("stub_tier")
class TestAProviderSatisfiesEveryContract:
    """The seam is the interface, not the implementation.

    Nothing in this repository fills these kinds — that is the whole point
    of the split — so the provider here is a stand-in that computes nothing.
    If the registry can import it, bind the contract's method and report its
    capabilities, then an installed ``diff-brep`` is bound the same way, and
    the only thing this repository has to get right is the interface.
    """

    @pytest.mark.parametrize("kind", tier.KINDS)
    def test_the_registered_object_satisfies_the_protocol(self, kind):
        plugin = plugin_for_kind(kind)
        assert isinstance(plugin, PythonPlugin)
        assert plugin.spec.transport == PluginTransport.PYTHON
        # ``component`` raises PluginMismatch when the Protocol is not met.
        assert isinstance(plugin.component, contracts.contract_for(kind))

    @pytest.mark.parametrize("kind", tier.KINDS)
    def test_the_contract_method_is_what_apply_and_as_jax_bind(self, kind):
        plugin = plugin_for_kind(kind)
        name = contracts.primary_method(kind)
        assert plugin.method.__name__ == name
        assert plugin.as_jax() == plugin.method
        assert plugin.as_jax().__self__ is plugin.component

    @pytest.mark.parametrize("kind", tier.KINDS)
    def test_the_object_declares_this_contract_version(self, kind):
        plugin = plugin_for_kind(kind)
        assert plugin.contract_version == CONTRACT_VERSION
        assert plugin.probe().status == "ok"
        assert plugin.probe().schema_hash == contracts.contract_signature(kind)

    def test_the_node_map_is_the_only_one_that_advertises_a_vjp(self):
        node_map = plugin_for_kind(PluginKind.NODE_MAP.value)
        assert node_map.capabilities.supports("differentiable")
        assert node_map.capabilities.supports("in_process")
        assert "vector_jacobian_product" in node_map.capabilities.endpoints
        edges = plugin_for_kind(PluginKind.FEATURE_EDGES.value)
        assert not edges.capabilities.supports("differentiable")
        with pytest.raises(ValueError, match="carries no derivative"):
            edges.vjp({}, ["scene"], ["result"], {"result": None})

    def test_the_registry_files_them_under_the_kinds_the_tier_asks_for(self, stub_tier):
        assert set(stub_tier) == set(tier.KINDS)
        for kind in tier.KINDS:
            assert plugin_for_kind(kind).spec.kind == kind

    def test_nothing_in_this_repository_fills_them_on_its_own(self):
        """Without a provider registered, all five kinds are simply absent.

        The registry ships ``local`` specs for the Tesseract packages in
        this checkout and nothing else; the private tier is discovered, never
        bundled (``research/two-tier.md`` §5 step 4).
        """
        from cadjoint.plugins import builtin_specs

        assert {spec.kind for spec in builtin_specs().values()}.isdisjoint(tier.KINDS)


class TestTheImportIsChecked:
    """A python spec that does not deliver says so, and does not crash later."""

    def test_an_object_missing_the_contract_is_refused_by_name(self):
        from cadjoint.plugins import PluginMismatch

        register_plugin(
            PluginSpec(
                name="not_a_node_map",
                kind=PluginKind.NODE_MAP.value,
                transport=PluginTransport.PYTHON,
                object="numpy:ndarray",
            ),
            default=True,
        )
        with pytest.raises(PluginMismatch, match="NodeMap contract"):
            assert plugin_for_kind(PluginKind.NODE_MAP.value).component

    def test_a_missing_module_is_a_reason_not_a_traceback(self):
        register_plugin(
            PluginSpec(
                name="absent_edges",
                kind=PluginKind.FEATURE_EDGES.value,
                transport=PluginTransport.PYTHON,
                object="cadjoint_no_such_module:EDGES",
            ),
            default=True,
        )
        entry = tier.status()[PluginKind.FEATURE_EDGES.value]
        assert not entry.available
        assert "cadjoint_no_such_module" in entry.reason
        assert tier.component(PluginKind.FEATURE_EDGES.value) is None

    def test_a_malformed_object_string_is_a_config_error(self):
        from cadjoint.plugins import PluginConfigError

        spec = PluginSpec(
            name="bad",
            kind=PluginKind.BREP.value,
            transport=PluginTransport.PYTHON,
            object="tests.plugins.stubs",
        )
        with pytest.raises(PluginConfigError, match="module:attribute"):
            assert PythonPlugin(spec).component

    def test_a_stale_build_reports_the_version_it_wants(self):
        """The sentence a user gets when diff-brep is older than cadjoint."""

        class _Stale:
            version = "0.1.0"
            contract_version = CONTRACT_VERSION + 1

            def extract(self, scene, grid, **options):  # pragma: no cover - never called
                return None

        import cadjoint.plugins.contracts as module

        module._STALE_FOR_TEST = _Stale()  # type: ignore[attr-defined]
        try:
            register_plugin(
                PluginSpec(
                    name="stale_brep",
                    kind=PluginKind.BREP.value,
                    transport=PluginTransport.PYTHON,
                    object="cadjoint.plugins.contracts:_STALE_FOR_TEST",
                ),
                default=True,
            )
            entry = tier.status()[PluginKind.BREP.value]
            assert not entry.compatible and not entry.available
            assert f"version {CONTRACT_VERSION + 1}" in entry.reason
            with pytest.raises(tier.TierUnavailable) as caught:
                tier.require(PluginKind.BREP.value)
            assert "built for plugin contract version" in str(caught.value)
        finally:
            del module._STALE_FOR_TEST  # type: ignore[attr-defined]


class TestTheOwnedNodesRecord:
    """The one payload that crosses in both directions."""

    def test_the_layout_properties_read_the_blocks(self):
        owned = _owned()
        assert owned.count == 5
        assert owned.num_corner == 4
        assert owned.num_surface == 3
        assert owned.order == 2
        assert owned.arity_counts() == {0: 1, 1: 1, 2: 2, 3: 1}
        assert owned.owned.tolist() == [True, True, True, False, True]

    def test_it_round_trips_through_a_plain_mapping(self):
        owned = _owned()
        rebuilt = OwnedNodes.from_mapping(owned.to_mapping())
        assert np.array_equal(rebuilt.seeds, owned.seeds)
        assert np.array_equal(rebuilt.patches, owned.patches)
        assert np.array_equal(rebuilt.edge_parents, owned.edge_parents)
        assert rebuilt.bar == owned.bar
        assert rebuilt.design_digest() == owned.design_digest()

    def test_the_design_digest_moves_with_the_design(self):
        assert _owned().design_digest() != _owned(design={"radius": np.array(0.30)}).design_digest()

    def test_arity_must_count_the_filled_slots(self):
        with pytest.raises(ValueError, match="arity must count"):
            _owned(arity=np.array([2, 2, 1, 0, 2], dtype=np.int8))

    def test_a_blend_is_an_unowned_boundary_node_and_nothing_else(self):
        with pytest.raises(ValueError, match="blend must mark"):
            _owned(blend=np.array([True, False, False, False, False]))
        with pytest.raises(ValueError, match="blend must mark"):
            # arity 0 but a volume node: mesh gauge, not a blend.
            _owned(blend=np.array([False, False, False, True, False]))

    def test_the_midside_block_must_match_edge_parents(self):
        with pytest.raises(ValueError, match="edge_parents"):
            _owned(edge_parents=np.zeros((2, 2), dtype=np.int32))

    def test_a_ragged_field_is_refused_by_shape(self):
        with pytest.raises(ValueError, match="seeds must be shaped"):
            _owned(seeds=np.zeros((5, 2)))
        with pytest.raises(ValueError, match="cells must be shaped"):
            _owned(cells=np.zeros((1, 5), dtype=np.int32))

    def test_a_short_column_is_refused_against_the_seeds(self):
        with pytest.raises(ValueError, match="rows but seeds has 5"):
            _owned(arity=np.array([3, 2, 1, 0], dtype=np.int8))


class TestTheEdgeSetPayload:
    """Display segments: what the ``feature_edges`` kind hands the overlay."""

    def _edges(self, **overrides) -> EdgeSet:
        fields = {
            "polylines": (
                np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]]),
                np.array([[0.0, 0, 1.0], [1.0, 0, 1.0]]),
            ),
            "closed": np.array([True, False]),
            "patches": np.array([[0, 1], [1, 2]], dtype=np.int32),
            "kind": ("traced", "sampled"),
            "residual": np.array([1e-9, 2e-9]),
            "vertices": np.array([[-1, -1], [0, 1]], dtype=np.int32),
        }
        fields.update(overrides)
        return EdgeSet(**fields)

    def test_chords_close_a_closed_curve_and_not_an_open_one(self):
        chords = self._edges().chords()
        # 3 chords around the closed triangle, 1 along the open segment.
        assert chords.shape == (4, 2, 3)
        assert chords[2, 1] == pytest.approx(chords[0, 0]), "the closed curve wraps"

    def test_it_round_trips_through_a_plain_mapping(self):
        edges = self._edges()
        rebuilt = EdgeSet.from_mapping(edges.to_mapping())
        assert rebuilt.count == edges.count
        assert rebuilt.kind == edges.kind
        for before, after in zip(edges.polylines, rebuilt.polylines):
            assert np.array_equal(before, after)

    def test_a_column_that_does_not_describe_every_curve_is_refused(self):
        with pytest.raises(ValueError, match="for 2 curves"):
            self._edges(kind=("traced",))

    def test_an_empty_set_still_yields_an_empty_chord_array(self):
        empty = EdgeSet(
            polylines=(),
            closed=np.zeros(0, bool),
            patches=np.zeros((0, 2), np.int32),
            kind=(),
            residual=np.zeros(0),
            vertices=np.zeros((0, 2), np.int32),
        )
        assert empty.chords().shape == (0, 2, 3)
