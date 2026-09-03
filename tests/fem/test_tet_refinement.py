"""Automatic grid refinement when the DC surface is too coarse to tetrahedralize.

Tets are less forgiving than hexes on thin features.  The hex mesher only
decides in/out per cell, so a rib 1.3 cells thick still produces a legal
(if staircased) HEX8 mesh; TetGen needs the DC surface to be a valid
piecewise linear complex, and a wall thinner than about two cells makes
dual contouring fold that surface over itself.  The user-visible symptom
is ``RuntimeError: TetGen rejected the surface: The input surface mesh
contain self-intersections.`` for both ``sharp=True`` and the
``sharp=False`` fallback.

:func:`~cadjoint.fem.tetmesh.sdf_to_tet_mesh` now answers that by walking
a ladder of grids over the same box — declared, ``x1.5``, ``x2.25`` — and
records what it had to do on the returned mesh.  These tests pin the three
outcomes that matter: a scene that needs one refinement, a scene no rung
saves, and the starter that must still mesh on the first attempt.

The synthetic scene is a ribbed disc — a plate, a hub, and eight polar
gusset ribs joined with a smooth minimum — which is the end-cap's failure
shape in miniature: the fillet radius of the smooth union is comparable to
the rib thickness, so the DC surface folds where a rib meets the plate.
Rib thickness is stated in *cells* of the declared grid so the failure
tracks the resolution rather than a magic length.

Version note: the declared rung of both failing cases is rejected by the
*diagnostic* (exact segment-triangle geometry in numpy, deterministic for
the fixed seed), so those assertions do not depend on TetGen's internals.
The refined rungs of :class:`TestNoRungSaves` do depend on TetGen
rejecting them; they were measured against the tetgen wheel this repo
pins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tetgen")

import jax.numpy as jnp  # noqa: E402

from cadjoint.fem.tetmesh import refine_resolution, sdf_to_tet_mesh, tet10_mesh  # noqa: E402
from cadjoint.meshing import GridSpec  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]

#: Box the ribbed disc lives in; shared by every grid below so the ladder
#: only ever re-dices this volume.
_BOUNDS = (-1.05, -1.05, -0.12)
_SIZE = (2.1, 2.1, 0.86)


def _smooth_min(a, b, blend: float):
    """Polynomial smooth minimum (the blend the CAD booleans use)."""
    h = jnp.clip(0.5 + 0.5 * (b - a) / blend, 0.0, 1.0)
    return b * (1.0 - h) + a * h - blend * h * (1.0 - h)


def _ribbed_disc(rib_thickness: float, blend: float, ribs: int = 8):
    """A plate + hub + polar gusset ribs, joined with a smooth minimum.

    Args:
        rib_thickness: Full rib thickness in world units.
        blend: Smooth-union radius; comparable to ``rib_thickness`` is what
            folds the extracted surface.
        ribs: Number of gussets in the polar pattern.

    Returns:
        An SDF callable on ``(..., 3)`` points.
    """
    hub_radius, outer_radius, rib_top = 0.30, 0.90, 0.60
    half = 0.5 * rib_thickness

    def sdf(points):
        points = jnp.asarray(points)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        radius = jnp.sqrt(x * x + y * y + 1e-18)
        plate = jnp.maximum(radius - outer_radius, jnp.abs(z - 0.06) - 0.06)
        hub = jnp.maximum(radius - hub_radius, jnp.abs(z - 0.30) - 0.30)
        body = _smooth_min(plate, hub, blend)
        sector = 2.0 * jnp.pi / ribs
        angle = jnp.mod(jnp.arctan2(y, x) + sector / 2.0, sector) - sector / 2.0
        across = radius * jnp.sin(angle)
        along = radius * jnp.cos(angle)
        rib = jnp.maximum(
            jnp.maximum(jnp.abs(across) - half, along - outer_radius),
            jnp.maximum(hub_radius * 0.5 - along, jnp.abs(z - rib_top / 2.0) - rib_top / 2.0),
        )
        return _smooth_min(body, rib, blend)

    return sdf


def _grid(resolution: tuple[int, int, int]) -> GridSpec:
    return GridSpec.from_bounds(_BOUNDS, _SIZE, resolution)


def _scene(resolution: tuple[int, int, int], cells_thick: float, blend: float):
    """The ribbed disc whose ribs are ``cells_thick`` cells of ``resolution``."""
    grid = _grid(resolution)
    return _ribbed_disc(cells_thick * grid.spacing[0], blend), grid


class TestRefineResolution:
    def test_each_axis_rounds_up(self):
        assert refine_resolution((26, 26, 13), 1.5) == (39, 39, 20)
        assert refine_resolution((26, 26, 13), 2.25) == (59, 59, 30)

    def test_no_axis_collapses(self):
        assert refine_resolution((3, 1, 1), 0.1) == (1, 1, 1)


class TestRefinementRescues:
    """A rib one cell thick at the declared grid, two after one refinement."""

    _DECLARED = (14, 14, 7)

    @pytest.fixture(scope="class")
    def mesh(self):
        sdf, grid = _scene(self._DECLARED, cells_thick=1.0, blend=0.06)
        return sdf_to_tet_mesh(sdf, grid)

    def test_the_declared_grid_is_abandoned_for_a_finer_one(self, mesh):
        record = mesh.refinement
        assert record["declared"] == self._DECLARED
        assert record["used"] == refine_resolution(self._DECLARED, 1.5) == (21, 21, 11)
        assert record["factor"] == 1.5
        assert record["refined"] is True

    def test_the_declared_rung_is_skipped_on_the_diagnostic_not_on_tetgen(self, mesh):
        # Both placements at the declared resolution fold, and the sampled
        # diagnostic sees it — so TetGen is never asked (no "error" key).
        declared = [a for a in mesh.refinement["attempts"] if a["resolution"] == self._DECLARED]
        assert [a["sharp"] for a in declared] == [True, False]
        for attempt in declared:
            assert attempt["outcome"] == "self-intersecting"
            assert attempt["self_intersections"] > 0
            assert attempt["pairs_tested"] > 0
            assert "error" not in attempt

    def test_the_winning_rung_is_the_last_attempt_and_is_clean(self, mesh):
        last = mesh.refinement["attempts"][-1]
        assert last["outcome"] == "meshed"
        assert last["resolution"] == (21, 21, 11)
        assert last["sharp"] is True
        assert last["self_intersections"] == 0

    def test_the_mesh_is_a_valid_tet_mesh_on_the_refined_grid(self, mesh):
        from cadjoint.fem.quality import tet_volumes

        assert mesh.grid.cells == (21, 21, 11)
        assert np.allclose(mesh.grid.origin, _BOUNDS)
        assert mesh.num_cells > 0 and mesh.num_surface > 0
        assert mesh.cells.shape[1] == 4
        assert float(tet_volumes(mesh.points, mesh.cells).min()) > 0.0
        # -Y preservation still holds through the refinement wrapper.
        assert np.array_equal(np.unique(mesh.boundary_tris), np.arange(mesh.num_surface))

    def test_the_boundary_sits_on_the_zero_set(self, mesh):
        sdf, _ = _scene(self._DECLARED, cells_thick=1.0, blend=0.06)
        values = np.asarray(sdf(jnp.asarray(mesh.points[: mesh.num_surface])))
        assert np.abs(values).max() < 1e-6

    def test_the_record_survives_the_tet10_promotion(self, mesh):
        assert tet10_mesh(mesh).refinement is mesh.refinement

    def test_disabling_refinement_restores_the_old_failure(self):
        sdf, grid = _scene(self._DECLARED, cells_thick=1.0, blend=0.06)
        with pytest.raises(RuntimeError, match="TetGen rejected the surface"):
            sdf_to_tet_mesh(sdf, grid, max_refinements=0)


class TestNoRungSaves:
    """A scene the ladder cannot rescue: the error has to say so usefully."""

    _DECLARED = (12, 12, 6)

    @pytest.fixture(scope="class")
    def failure(self):
        sdf, grid = _scene(self._DECLARED, cells_thick=1.2, blend=0.06)
        with pytest.raises(RuntimeError) as caught:
            sdf_to_tet_mesh(sdf, grid)
        return str(caught.value)

    def test_the_message_keeps_the_original_prefix(self, failure):
        assert failure.startswith("TetGen rejected the surface: ")

    def test_the_message_names_the_declared_and_the_finest_resolution(self, failure):
        assert "declared (12, 12, 6)" in failure
        assert f"up to {refine_resolution(self._DECLARED, 2.25)}" in failure
        assert "(27, 27, 14)" in failure
        assert "x1.5 and x2.25" in failure

    def test_the_message_gives_the_thin_feature_heuristic_and_the_two_ways_out(self, failure):
        assert "thinner than two cells" in failure
        assert "raise the declared resolution" in failure
        assert "method='hex'" in failure


class TestStarterStillMeshesOnTheFirstAttempt:
    """Regression guard: the ladder must not perturb geometry that was fine.

    The starter's ``thermal_body`` at its declared (18, 13, 11) meshed
    before the ladder existed and must still mesh at the declared
    resolution, on the first (sharp) attempt, with the node and element
    counts it always had — 990 nodes / 3182 tets / 860 surface vertices
    for TET4, 6019 nodes for the TET10 promotion.
    """

    _DECLARED = (18, 13, 11)
    _SCENE = _REPO / "scenes" / "starter.py"

    @pytest.fixture(scope="class")
    def mesh(self):
        from cadjoint import extract_parameters, functionalize
        from cadjoint.fem import capture_sim_meshes, capture_studies

        namespace: dict = {}
        with capture_sim_meshes(), capture_studies():
            exec(compile(self._SCENE.read_text(), str(self._SCENE), "exec"), namespace, namespace)
        body = namespace["thermal_body"]
        declared = namespace["sink_mesh"]
        assert tuple(declared.resolution) == self._DECLARED
        free, fixed, _ = extract_parameters(body)
        sdf = functionalize(body)(free, fixed)
        return sdf_to_tet_mesh(sdf, declared.grid(sdf))

    def test_no_refinement_was_needed(self, mesh):
        record = mesh.refinement
        assert record["declared"] == record["used"] == self._DECLARED
        assert record["refined"] is False
        assert record["factor"] == 1.0

    def test_it_meshed_on_the_very_first_attempt(self, mesh):
        assert len(mesh.refinement["attempts"]) == 1
        only = mesh.refinement["attempts"][0]
        assert only["sharp"] is True
        assert only["outcome"] == "meshed"
        assert only["self_intersections"] == 0

    def test_the_node_and_element_counts_are_unchanged(self, mesh):
        assert (mesh.num_points, mesh.num_cells, mesh.num_surface) == (990, 3182, 860)
        assert tet10_mesh(mesh).num_points == 6019


class TestTheRecordReachesTheResult:
    """The ladder is invisible unless the record travels: SimMesh -> describe()."""

    _DECLARED = (14, 14, 7)

    @pytest.fixture(scope="class")
    def result(self):
        pytest.importorskip("jax_fem")
        from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy

        sdf, _ = _scene(self._DECLARED, cells_thick=1.0, blend=0.06)
        sim_mesh = SimMesh(
            name="ribbed-disc",
            resolution=self._DECLARED,
            bounds=_BOUNDS,
            size=_SIZE,
            method="tet4",
            domain=sdf,
        )
        study = ThermalStudy(
            name="disc-conduction",
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("+z"), 0.0), HeatFlux(Nodes.side("-z"), 1.0)],
            mesh=sim_mesh,
        )
        return study.solve()

    def test_the_simmesh_path_refines_too(self, result):
        # SimMesh.build() does not special-case this; it gets the ladder
        # because sdf_to_tet_mesh owns it.
        assert result.mesh.grid.cells == (21, 21, 11)
        assert result.refinement["used"] == (21, 21, 11)

    def test_describe_reports_the_refinement_json_ready(self, result):
        import json

        payload = result.describe()
        assert payload["refinement"] == {
            "declared": [14, 14, 7],
            "used": [21, 21, 11],
            "attempts": 3,
        }
        assert json.loads(json.dumps(payload)) == payload

    def test_an_unrefined_result_reports_none(self, result):
        from cadjoint.fem.result import _refinement_summary

        assert _refinement_summary(None) is None
        assert _refinement_summary({"refined": False, "declared": (1,), "used": (1,)}) is None
