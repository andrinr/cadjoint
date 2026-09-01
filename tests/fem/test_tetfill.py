"""The tetfill tesseract and the DC gradient chain built on it.

``cadjoint/fem/tesseracts/tetfill`` wraps only TetGen — dual contouring
stays in JAX on the true SDF — so its VJP w.r.t. the surface vertices is
the transpose of a gather and must be exact, not approximate.  These tests
pin that:

- TetGen's ``-Y`` preservation holds bit-for-bit (the contract the whole
  VJP rests on), and the ``parents`` table encodes pass-through on
  preserved vertices, zero on Steiner nodes, half-and-half on TET10
  midsides;
- the VJP equals ``jax.vjp`` of the equivalent JAX gather to ~1e-15;
- the frozen fill (interior pinned) reproduces the TetGen fill exactly, so
  forward and derivative are consistent by construction;
- the chain ``freeze_study_chain_dc`` builds is a fixed point at the
  nominal design, solves identically to the in-process path, and its
  adjoint matches central finite differences of the same chain;
- ``Optimization(gradient_path="tesseract-dc")`` runs end to end.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tetgen")
pytest.importorskip("jax_fem")
pytest.importorskip("tesseract_core")
pytest.importorskip("tesseract_jax")

import jax
import jax.numpy as jnp

from cadjoint.fem.hexmesh import project_points
from cadjoint.fem.tesseracts.chain import _tesseract, freeze_study_chain_dc
from cadjoint.fem.tetmesh import tet10_from_tet4
from cadjoint.meshing import GridSpec, extract_mesh

_MIN_RATIO = np.float64(1.5)
_MIN_DIHEDRAL = np.float64(10.0)


def _sphere(point):
    """Sphere SDF (radius 0.6) on a single ``(3,)`` point."""
    return jnp.sqrt(jnp.sum(jnp.asarray(point) ** 2) + 1e-30) - 0.6


@pytest.fixture(scope="module")
def surface():
    """A small watertight DC surface of the sphere, projected onto its zero set."""
    grid = GridSpec(origin=(-1.0, -1.0, -1.0), spacing=(0.2, 0.2, 0.2), cells=(10, 10, 10))
    mesh = extract_mesh(_sphere, grid, sharp=False)
    raw = np.asarray(mesh.vertices, dtype=np.float64)
    points = np.asarray(
        project_points(_sphere, raw, 0.5 * float(np.linalg.norm(grid.spacing))), dtype=np.float64
    )
    return points, np.asarray(mesh.faces, dtype=np.int32)


def _payload(points, triangles, element, *, interior=None, template=None, node_ids=None):
    """One tetfill payload; ``interior``/``template`` switch on the frozen fill."""
    return {
        "points": points,
        "triangles": triangles,
        "element": np.int32(element),
        "min_ratio": _MIN_RATIO,
        "min_dihedral": _MIN_DIHEDRAL,
        "interior_points": np.zeros((0, 3)) if interior is None else interior,
        "node_ids": np.zeros(0, np.int32) if node_ids is None else node_ids,
        "cell_template": np.zeros((0, 4), np.int32) if template is None else template,
    }


@pytest.fixture(scope="module")
def tetfill():
    return _tesseract("tetfill")


@pytest.fixture(scope="module")
def filled(surface, tetfill):
    """``{element: (nodes, cells, parents, mask)}`` for TET4 and TET10."""
    points, triangles = surface
    out = {}
    for element in (0, 2):
        result = tetfill.apply(_payload(points, triangles, element))
        out[element] = tuple(
            np.asarray(result[key]) for key in ("nodes", "cells", "parents", "steiner_mask")
        )
    return out


class TestTetfillForward:
    def test_input_vertices_survive_verbatim(self, surface, filled):
        """The ``-Y`` contract, bit-for-bit — the VJP is exact only if it holds."""
        points, _triangles = surface
        for element in (0, 2):
            nodes = filled[element][0]
            assert np.array_equal(nodes[: len(points)], points)

    def test_steiner_mask_marks_everything_added(self, surface, filled):
        points, _triangles = surface
        for element in (0, 2):
            mask = filled[element][3]
            assert mask[: len(points)].sum() == 0
            assert mask[len(points) :].all()

    def test_tet10_appends_midsides_after_the_corners(self, surface, filled):
        points, _triangles = surface
        corner_nodes, corner_cells = filled[0][0], filled[0][1]
        nodes10, cells10 = filled[2][0], filled[2][1]
        assert corner_cells.shape[1] == 4
        assert cells10.shape[1] == 10
        assert np.array_equal(cells10[:, :4], corner_cells)
        assert np.array_equal(nodes10[: len(corner_nodes)], corner_nodes)
        assert len(nodes10) > len(corner_nodes) > len(points)

    def test_parents_table_encodes_the_three_cases(self, surface, filled):
        """Pass-through on preserved, none on Steiner, two corners per midside."""
        points, _triangles = surface
        count = len(points)
        corners = len(filled[0][0])
        for element in (0, 2):
            parents = filled[element][2]
            preserved = np.arange(count, dtype=np.int32)
            assert np.array_equal(parents[:count, 0], preserved)
            assert np.array_equal(parents[:count, 1], preserved)
            assert (parents[count:corners] == -1).all()
        midside_parents = filled[2][2][corners:]
        # Every midside lists at most two parents, and each is a preserved vertex.
        assert midside_parents.shape[1] == 2
        assert ((midside_parents == -1) | (midside_parents < count)).all()
        assert (midside_parents >= 0).any()

    def test_hex_element_code_is_rejected(self, surface, tetfill):
        points, triangles = surface
        with pytest.raises(Exception, match="element must be 0"):
            tetfill.apply(_payload(points, triangles, 1))


class TestTetfillVJP:
    @pytest.mark.parametrize("element", [0, 2])
    def test_vjp_equals_the_jax_gather_transpose(self, surface, filled, tetfill, element):
        """Mechanical check: the map is a gather, so the VJP must be exact."""
        points, triangles = surface
        nodes, cells, _parents, _mask = filled[element]
        count = len(points)
        corner_count = len(filled[0][0])
        interior = jnp.asarray(nodes[count:corner_count])
        edges = None
        if element == 2:
            _points10, _cells10, edges = tet10_from_tet4(nodes[:corner_count], cells[:, :4])
            edges = jnp.asarray(edges)

        def gather(surface_points):
            corner_block = jnp.concatenate([surface_points, interior], axis=0)
            if edges is None:
                return corner_block
            return jnp.concatenate([corner_block, corner_block[edges].mean(axis=1)], axis=0)

        primal, vjp_fn = jax.vjp(gather, jnp.asarray(points))
        assert np.abs(np.asarray(primal) - nodes).max() == 0.0
        cotangent = np.random.default_rng(0).normal(size=nodes.shape)
        (reference,) = vjp_fn(jnp.asarray(cotangent))
        produced = np.asarray(
            tetfill.vector_jacobian_product(
                _payload(points, triangles, element),
                ["points"],
                ["nodes"],
                {"nodes": cotangent},
            )["points"]
        )
        reference = np.asarray(reference)
        relative = np.abs(produced - reference).max() / np.abs(reference).max()
        assert relative < 1e-14

    def test_steiner_cotangents_are_dropped(self, surface, filled, tetfill):
        """A cotangent living only on added nodes pulls back to exactly zero."""
        points, triangles = surface
        nodes, _cells, _parents, mask = filled[0]
        cotangent = np.zeros_like(nodes)
        cotangent[mask.astype(bool)] = 1.0
        produced = np.asarray(
            tetfill.vector_jacobian_product(
                _payload(points, triangles, 0), ["points"], ["nodes"], {"nodes": cotangent}
            )["points"]
        )
        assert np.abs(produced).max() == 0.0

    def test_only_points_carries_a_vjp(self, surface, tetfill):
        """The schema itself pins the differentiable surface to ``points``."""
        points, triangles = surface
        cotangent = np.zeros((1, 3))
        with pytest.raises(Exception, match="Input should be 'points'"):
            tetfill.vector_jacobian_product(
                _payload(points, triangles, 0), ["triangles"], ["nodes"], {"nodes": cotangent}
            )


class TestFrozenFill:
    @pytest.mark.parametrize("element", [0, 2])
    def test_frozen_fill_reproduces_tetgen_exactly(self, surface, filled, tetfill, element):
        """Pinning the interior re-evaluates the same mesh — forward and VJP agree."""
        points, triangles = surface
        nodes, cells, parents, mask = filled[element]
        interior = filled[0][0][len(points) :]
        result = tetfill.apply(
            _payload(
                points,
                triangles,
                element,
                interior=interior,
                template=cells,
                node_ids=np.arange(len(nodes), dtype=np.int32),
            )
        )
        assert np.array_equal(np.asarray(result["nodes"]), nodes)
        assert np.array_equal(np.asarray(result["cells"]), cells)
        assert np.array_equal(np.asarray(result["parents"]), parents)
        assert np.array_equal(np.asarray(result["steiner_mask"]), mask)

    def test_topology_promise_is_enforced(self, surface, filled, tetfill):
        points, triangles = surface
        nodes = filled[0][0]
        with pytest.raises(Exception, match="Frozen-topology promise violated"):
            tetfill.apply(
                _payload(
                    points,
                    triangles,
                    0,
                    node_ids=np.arange(len(nodes) + 7, dtype=np.int32),
                )
            )

    def test_frozen_interior_needs_the_connectivity(self, surface, filled, tetfill):
        points, triangles = surface
        interior = filled[0][0][len(points) :]
        with pytest.raises(Exception, match="cell_template"):
            tetfill.apply(_payload(points, triangles, 0, interior=interior))


# ── the chain: JAX dual contouring -> tetfill -> solver ──────────────────────


def _bar_field(half_length):
    """Box SDF whose x half-length is the (traceable) design parameter."""
    from cadjoint.sdf.primitives.box import Box

    size = jnp.stack([jnp.asarray(half_length), jnp.asarray(0.35), jnp.asarray(0.35)])
    return lambda point: Box.sdf(jnp.asarray(point), size)


@pytest.fixture(scope="module")
def bar_chain():
    """A frozen DC chain on a conduction bar: flux in at -x, held at +x.

    The analytic solution ``T = (q/k)(L - x)`` makes the mean temperature
    equal to ``q L / k`` — linear in the design parameter, so the adjoint
    has a known target as well as a finite-difference one.
    """
    from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy

    declaration = SimMesh(
        name="tetfill-bar",
        resolution=(14, 7, 7),
        bounds=(-1.1, -0.6, -0.6),
        size=(2.2, 1.2, 1.2),
        method="tet4",
    )
    study = ThermalStudy(
        name="tetfill-bar-study",
        conductivity=1.0,
        bcs=[HeatFlux(Nodes.side("-x"), 2.0), Dirichlet(Nodes.side("+x"), 0.0)],
        mesh=declaration,
    )
    half0 = jnp.asarray(0.8)
    chain = freeze_study_chain_dc(study, declaration, _bar_field(half0))
    return study, chain, half0


class TestDCChain:
    def test_frozen_mesh_is_a_fixed_point_of_the_traced_surface(self, bar_chain):
        """Re-extracting at the freeze design reproduces the meshed boundary."""
        _study, chain, half0 = bar_chain
        surface = np.asarray(chain.dc_surface(_bar_field(half0)))
        boundary = np.asarray(chain.mesh.points)[: chain.mesh.num_surface]
        assert np.abs(surface - boundary).max() == 0.0
        assert chain.mesh.num_surface == chain.surface_points.shape[0]

    def test_solver_stage_matches_the_in_process_solve(self, bar_chain):
        _study, chain, half0 = bar_chain
        study, _chain, _half = bar_chain
        _points, packaged = chain._solve(chain.dc_surface(_bar_field(half0)))
        direct = study.solve(mesh=chain.mesh).temperature
        assert np.abs(np.asarray(packaged) - np.asarray(direct)).max() < 1e-9

    def test_adjoint_matches_central_differences(self, bar_chain):
        """The whole chain is smooth: pinning the interior removes topology jumps."""
        _study, chain, half0 = bar_chain

        def objective(half_length):
            return chain.metric_value(_bar_field(half_length), "mean")

        value, gradient = jax.value_and_grad(objective)(half0)
        eps = 2e-3
        finite = (float(objective(half0 + eps)) - float(objective(half0 - eps))) / (2 * eps)
        relative = abs(float(gradient) - finite) / abs(finite)
        print(f"\nbar chain: J={float(value):.6f} adjoint={float(gradient):+.6f} FD={finite:+.6f}")
        assert relative < 5e-3
        # Analytic target: mean T = q L / k = 2 * half_length, so dJ/dL = 2.
        assert 1.5 < float(gradient) < 2.5

    def test_hex_meshes_are_rejected(self):
        from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy

        declaration = SimMesh(name="tetfill-hex", resolution=(6, 4, 4), method="hex")
        study = ThermalStudy(
            name="tetfill-hex-study",
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 1.0), Dirichlet(Nodes.side("+x"), 0.0)],
            mesh=declaration,
        )
        with pytest.raises(ValueError, match="tet SimMesh"):
            freeze_study_chain_dc(study, declaration, _bar_field(jnp.asarray(0.8)))


class TestGradientPathSeam:
    def test_the_new_path_is_declared(self):
        from cadjoint.optimize import GRADIENT_PATHS, Optimization

        assert GRADIENT_PATHS == ("direct", "tesseract", "tesseract-dc")
        assert Optimization.__dataclass_fields__["gradient_path"].default == "direct"

    def test_unknown_paths_are_rejected(self):
        from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
        from cadjoint.optimize import Optimization

        study = ThermalStudy(
            name="tetfill-seam",
            resolution=(6, 4, 4),
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 1.0), Dirichlet(Nodes.side("+x"), 0.0)],
        )
        with pytest.raises(ValueError, match="gradient_path"):
            Optimization("o", study=study, metric="mean", gradient_path="tesseract-dual")

    def test_optimization_runs_on_the_dc_path(self):
        """Two seam steps on the bar, gradients through both tesseracts."""
        from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
        from cadjoint.geometry.parameters import Vector
        from cadjoint.optimize import Optimization
        from cadjoint.sdf.primitives.box import Box

        box = Box(Vector([0.8, 0.35, 0.35], free=True, name="size"))
        declaration = SimMesh(
            name="tetfill-bar-opt",
            resolution=(14, 7, 7),
            domain=box,
            bounds=(-1.1, -0.6, -0.6),
            size=(2.2, 1.2, 1.2),
            method="tet4",
        )
        study = ThermalStudy(
            name="tetfill-bar-opt-study",
            conductivity=1.0,
            bcs=[HeatFlux(Nodes.side("-x"), 2.0), Dirichlet(Nodes.side("+x"), 0.0)],
            mesh=declaration,
        )
        optimization = Optimization(
            "cool-bar-dc",
            study=study,
            metric="mean",
            gradient_path="tesseract-dc",
            remesh_every=0,
            steps=2,
            learning_rate=0.01,
        )
        run = optimization.run(2)
        assert len(run.history) == 2
        assert all(
            np.isfinite(record["objective"]) and np.isfinite(record["grad_norm"])
            for record in run.history
        )
        assert run.history[0]["grad_norm"] > 0.0
        # Shorter bar, cooler mean: the descent moved the objective down.
        assert run.history[-1]["objective"] < run.history[0]["objective"]
        assert run.result is not None
