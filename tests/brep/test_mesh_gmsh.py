"""Tet10 from the exact B-rep via Gmsh: ownership, curvature, derivatives.

Three claims are worth a test and the rest is plumbing.

**Ownership survives the round trip.**  The graph writes exact STEP, OCCT
reads it back, and the CAD entities that come out are numbered by the
reader, not by the graph.  So every node Gmsh hands back has to be matched
to a :class:`~cadjoint.brep.graph.BRepFace` again, and its projection arity
has to fall out of the entity it sits on — one field on a surface, two on a
curve, three at a corner.  The plate is the case where every one of those
counts is known in advance: six planes, one cylinder, eight box corners.

**The midsides are on the geometry.**  This is the whole reason to go
through a kernel.  A straight-sided promotion puts the midside of a bore
edge at the chord's midpoint, a chord error inside the element; Gmsh's
``setOrder(2)`` puts it on the cylinder, and re-solving it against the
patch keeps it there when the radius moves.  The test measures the radius
of every bore midside and compares it to what the chord would have given.

**The positions differentiate.**  Topology is frozen and only positions
move, so a finite difference of the mesh volume against the bore radius has
to match ``jax.grad`` of the same thing — through
:func:`~cadjoint.brep.mesh_gmsh.parameterised_points`, which rebuilds the
patch fields under traced parameters.

The blends are the fourth case, and they are the starter's: a smooth union
has faces no patch owns, they leave the STEP as facets, and their nodes are
solved against the scene's own zero set instead.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from cadjoint.brep import extract_brep
from cadjoint.brep.mesh_gmsh import (
    TET_MESHER_KIND,
    GmshMesh,
    gmsh_available,
    gmsh_tet_mesh,
    gmsh_topology,
    gmsh_version,
    parameterised_points,
    recompute_gmsh_points,
    tet_mesh_from_gmsh,
)
from cadjoint.fem.quality import tet_radius_ratios, tet_volumes
from tests.brep.conftest import BORE_RADIUS, PLATE_GRID, PLATE_SIZE, plate_volume

pytestmark = pytest.mark.skipif(
    not gmsh_available(), reason="the optional 'gmsh' extra is not installed"
)

#: Coarse enough that the whole module costs about a second of Gmsh, fine
#: enough that the bore still carries several elements around its rim.
TARGET_SIZE = 0.16


@contextlib.contextmanager
def _x64():
    """Scope jax's x64 mode to one test, the way ``tests/fem`` does globally.

    The finite difference is against a mesh volume of order 1, at a step of
    1e-5: float32 has nowhere near the headroom, and turning x64 on at
    import time would poison every float32 suite that runs after this one.
    """
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def plate_mesh(plate_brep) -> GmshMesh:
    """The plate meshed to TET10 by Gmsh, with ownership assigned."""
    return gmsh_tet_mesh(plate_brep, target_size=TARGET_SIZE, order=2)


@pytest.fixture(scope="module")
def thermal_mesh(thermal_brep) -> GmshMesh:
    """The starter's thermal body: analytic where owned, faceted at blends."""
    return gmsh_tet_mesh(thermal_brep, target_size=0.16, order=2)


# ── the mesh itself ──────────────────────────────────────────────────────


def test_the_wheel_reports_a_version():
    assert gmsh_version().split(".")[0].isdigit()


class TestThePlateMeshes:
    """A hard CSG solid, exact all the way through, and no blend anywhere."""

    def test_it_is_a_tet10_mesh_of_positive_volume(self, plate_mesh):
        assert plate_mesh.order == 2
        assert plate_mesh.cells.shape[1] == 10
        volumes = tet_volumes(plate_mesh.points, plate_mesh.cells)
        assert (volumes > 0).all(), "cells should come back positively oriented"
        # Straight-sided volume of a mesh whose bore is an inscribed polygon
        # is under the analytic one, and by no more than the chord deficit.
        assert volumes.sum() == pytest.approx(plate_volume(), rel=0.02)

    def test_the_bounding_box_is_the_plate_in_metres(self, plate_mesh):
        # SI_UNIT(.METRE.) in, and OCCT's reader defaults to millimetres:
        # without Geometry.OCCTargetUnit the solid comes back 1000x.
        low = plate_mesh.points.min(axis=0)
        high = plate_mesh.points.max(axis=0)
        assert low == pytest.approx([-s for s in PLATE_SIZE], abs=1e-6)
        assert high == pytest.approx(list(PLATE_SIZE), abs=1e-6)

    def test_nothing_on_the_plate_is_a_blend(self, plate_mesh):
        assert not plate_mesh.blend_mask.any()
        assert plate_mesh.stats["blend_surfaces"] == 0

    def test_the_node_layout_is_the_one_tetmesh_documents(self, plate_mesh):
        corners = np.unique(plate_mesh.cells[:, :4])
        assert corners.max() == plate_mesh.num_corner_points - 1
        assert plate_mesh.num_surface <= plate_mesh.num_corner_points
        # Boundary corners lead, interior corners follow, midsides trail.
        assert (plate_mesh.entity_dim[: plate_mesh.num_surface] < 3).all()
        interior = plate_mesh.entity_dim[plate_mesh.num_surface : plate_mesh.num_corner_points]
        assert (interior == 3).all()

    def test_the_quality_beats_the_dual_contour_route(self, plate_mesh):
        ratios = tet_radius_ratios(plate_mesh.points, plate_mesh.cells)
        # The DC path measures 0.04 on this plate; a kernel mesher has no
        # lattice to graze a feature with, so the bar is an order up.
        assert ratios.min() > 0.15
        assert float(np.median(ratios)) > 0.7


class TestEveryNodeIsOwned:
    """Arity is codimension: the entity a node sits on says how many fields."""

    def test_arity_is_the_codimension_of_the_entity(self, plate_mesh):
        # ``3 - dim`` is the *cap*, not the count: a node is solved against
        # as many distinct patches as actually bound it.  The plate makes
        # the difference visible — its two circle-seam vertices are dim-0
        # entities where only two patches meet (the cylinder and one cap),
        # so they solve two fields, not three, and would be over-determined
        # if the codimension were taken literally.
        for dim in (0, 1, 2, 3):
            rows = plate_mesh.entity_dim == dim
            arities = plate_mesh.owner_arity[rows]
            assert (arities <= 3 - dim).all(), f"dim {dim} over-determined"
            assert (arities >= min(1, 3 - dim)).all(), f"dim {dim} unowned"

    def test_a_volume_node_is_owned_by_nothing(self, plate_mesh):
        interior = plate_mesh.entity_dim == 3
        assert (plate_mesh.owner_arity[interior] == 0).all()
        assert not plate_mesh.blend_mask[interior].any()

    def test_the_two_circle_seams_are_the_two_field_corners(self, plate_brep, plate_mesh):
        rows = np.flatnonzero((plate_mesh.entity_dim == 0) & (plate_mesh.owner_arity == 2))
        assert rows.size == 2, "the STEP writes 8 box corners + 2 circle seams"
        radius = np.linalg.norm(plate_mesh.points[rows, :2], axis=1)
        assert radius == pytest.approx(BORE_RADIUS, abs=1e-6)
        kinds = {plate_brep.faces[int(index)].kind for index in plate_mesh.owner_face[rows]}
        assert kinds <= {"plane", "cylinder"}

    def test_the_owner_row_carries_exactly_that_many_patches(self, plate_mesh):
        filled = (plate_mesh.owner_patches >= 0).sum(axis=1)
        assert (filled == plate_mesh.owner_arity).all()
        assert (plate_mesh.owner_patches[:, 0][plate_mesh.owner_arity > 0] >= 0).all()

    def test_the_eight_box_corners_are_the_only_three_field_nodes(self, plate_mesh):
        corners = np.flatnonzero(plate_mesh.owner_arity == 3)
        assert corners.size == 8, "a box has eight corners; the bore rims are curves"
        magnitudes = np.abs(plate_mesh.points[corners])
        # 1e-6, not 1e-9: the graph's parameters are float32 and the STEP
        # carries the half-extent as 0.60000002, so that is the plate the
        # kernel was actually handed.
        assert magnitudes == pytest.approx(np.broadcast_to(PLATE_SIZE, (8, 3)), abs=1e-6)

    def test_every_owned_node_actually_satisfies_its_own_fields(self, plate_brep, plate_mesh):
        # The claim ownership makes: |f_p(x)| vanishes at the node for each
        # patch p it was assigned.  Checked on the surface nodes, where a
        # wrong assignment would be a large residual rather than a small one.
        import jax

        fields = [patch.field for patch in plate_brep.patches]
        rows = np.flatnonzero(plate_mesh.owner_arity > 0)
        points = np.asarray(plate_mesh.points[rows], dtype=np.float32)
        residuals = np.stack([np.asarray(jax.vmap(field)(points)) for field in fields], axis=-1)
        for slot in range(3):
            owners = plate_mesh.owner_patches[rows, slot]
            live = owners >= 0
            worst = np.abs(residuals[np.flatnonzero(live), owners[live]]).max()
            assert worst < 1e-4, f"slot {slot} residual {worst}"

    def test_the_face_a_node_is_matched_to_is_a_real_face(self, plate_brep, plate_mesh):
        matched = plate_mesh.owner_face[plate_mesh.entity_dim < 3]
        assert (matched >= 0).all()
        assert matched.max() < len(plate_brep.faces)
        kinds = {plate_brep.faces[int(index)].kind for index in np.unique(matched)}
        assert kinds == {"plane", "cylinder"}


class TestTheMidsidesAreOnTheGeometry:
    """The reason for the kernel: a curved midside a promotion cannot give."""

    def _bore_edges(self, brep, mesh):
        """Midside rows whose two parents both sit on the cylinder patch."""
        cylinder = next(face for face in brep.faces if face.kind == "cylinder")
        on_bore = np.zeros(mesh.points.shape[0], dtype=bool)
        on_bore[mesh.owner_face == cylinder.index] = True
        parents = mesh.edge_parents
        both = on_bore[parents[:, 0]] & on_bore[parents[:, 1]]
        return np.flatnonzero(both) + mesh.num_corner_points, parents[both]

    def test_a_bore_midside_sits_on_the_cylinder_not_the_chord(self, plate_brep, plate_mesh):
        rows, parents = self._bore_edges(plate_brep, plate_mesh)
        assert rows.size > 8, "the bore should carry a ring of curved edges"
        radius = np.linalg.norm(plate_mesh.points[rows, :2], axis=1)
        chord = np.linalg.norm(plate_mesh.points[parents][:, :, :2].mean(axis=1), axis=1)
        # Only the edges that actually go around the bore bow; an edge along
        # the axis has its chord on the cylinder already.
        bowed = chord < BORE_RADIUS - 1e-4
        assert bowed.any(), "some bore edges must span an arc"
        assert radius[bowed] == pytest.approx(BORE_RADIUS, abs=1e-6)
        assert (radius[bowed] - chord[bowed]).max() > 1e-3, "the chord deficit is real"

    def test_the_midside_block_is_in_meshio_order(self, plate_mesh):
        # Gmsh's tet10 swaps the last two midsides against meshio's; a wrong
        # remap puts a midside nowhere near its own corner pair.  Every
        # midside must lie within a chord of the midpoint of its parents.
        from cadjoint.fem.elements import TET10_EDGES

        cells = plate_mesh.cells
        corners = plate_mesh.points[cells[:, :4]]
        midpoints = corners[:, TET10_EDGES].mean(axis=2)
        actual = plate_mesh.points[cells[:, 4:]]
        offsets = np.linalg.norm(actual - midpoints, axis=-1)
        edges = np.linalg.norm(
            corners[:, TET10_EDGES[:, 1]] - corners[:, TET10_EDGES[:, 0]], axis=-1
        )
        assert (offsets < 0.2 * edges).all(), "a midside is near the midpoint of its own edge"

    def test_edge_parents_matches_the_connectivity(self, plate_mesh):
        from cadjoint.fem.elements import TET10_EDGES

        cells = plate_mesh.cells
        pairs = np.sort(cells[:, :4][:, TET10_EDGES], axis=2)
        rows = cells[:, 4:] - plate_mesh.num_corner_points
        assert (plate_mesh.edge_parents[rows] == pairs).all()


# ── differentiable positions over the frozen topology ────────────────────


class TestPositionsRecompute:
    """Topology frozen, positions re-solved: the contract the plugin rests on."""

    def test_the_nominal_re_solve_is_a_fixed_point(self, plate_brep, plate_mesh):
        solved = recompute_gmsh_points(plate_brep, plate_mesh)
        owned = plate_mesh.owner_arity > 0
        moved = np.linalg.norm(solved[owned] - plate_mesh.points[owned], axis=1)
        # Gmsh already put these on the CAD surface, so the projection has
        # nothing left to do: anything but a fixed point here means the
        # patch a node was assigned is not the surface it is on.
        assert moved.max() < 1e-5, f"worst nominal drift {moved.max()}"

    def test_a_volume_node_is_left_where_gmsh_put_it(self, plate_brep, plate_mesh):
        solved = recompute_gmsh_points(plate_brep, plate_mesh)
        interior = plate_mesh.entity_dim == 3
        assert (solved[interior] == plate_mesh.points[interior]).all()

    def test_smoothing_moves_only_the_interior(self, plate_brep, plate_mesh):
        solved = recompute_gmsh_points(plate_brep, plate_mesh, smooth_passes=2)
        boundary = plate_mesh.entity_dim < 3
        assert np.allclose(solved[boundary], plate_mesh.points[boundary], atol=1e-5)
        assert not np.allclose(solved[~boundary], plate_mesh.points[~boundary], atol=1e-9)

    def test_handing_over_to_the_fem_layer_keeps_the_mesh_valid(self, plate_brep, plate_mesh):
        mesh = tet_mesh_from_gmsh(plate_brep, plate_mesh)
        assert mesh.order == 2
        assert mesh.ele_type == "TET10"
        assert mesh.num_corner_points == plate_mesh.num_corner_points
        assert mesh.boundary_tris.shape[1] == 3
        assert (tet_volumes(mesh.points, mesh.cells) > 0).all()
        assert mesh.edge_parents is not None


class TestTheDerivative:
    """``jax.grad`` through the projection against a central difference."""

    @pytest.fixture(scope="class")
    def parametric(self):
        """A plate whose bore radius and half-thickness are free parameters."""
        from cadjoint.geometry import Scalar, Vector
        from cadjoint.sdf.boolean import Difference
        from cadjoint.sdf.primitives import Box, Cylinder

        scene = Difference(
            (
                Box(size=Vector(list(PLATE_SIZE), free=True, name="plate")),
                Cylinder(radius=Scalar(BORE_RADIUS, free=True, name="bore"), height=Scalar(0.9)),
            ),
            smoothness=0.0,
        )
        brep = extract_brep(scene, PLATE_GRID)
        return scene, brep, gmsh_tet_mesh(brep, target_size=TARGET_SIZE, order=2)

    def _volume_of(self, mesh):
        import jax.numpy as jnp

        cells = np.asarray(mesh.cells[:, :4], dtype=np.int64)

        def volume(points):
            corners = points[cells]
            return jnp.abs(jnp.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0).sum()

        return volume

    @pytest.mark.parametrize(
        ("name", "component", "step"), [("bore", None, 1e-5), ("plate", 2, 1e-5)]
    )
    def test_the_mesh_volume_differentiates_in_the_design(self, parametric, name, component, step):
        import jax
        import jax.numpy as jnp

        from cadjoint.extraction import extract_parameters

        scene, _brep, mesh = parametric
        with _x64():
            baseline, _fixed, _metadata = extract_parameters(scene)
            nominal = {key: jnp.asarray(value, jnp.float64) for key, value in baseline.items()}
            volume = self._volume_of(mesh)

            def objective(value):
                params = dict(nominal)
                params[name] = (
                    value if component is None else nominal[name].at[component].set(value)
                )
                return volume(parameterised_points(scene, mesh, params, steps=10))

            start = nominal[name] if component is None else nominal[name][component]
            analytic = float(jax.grad(objective)(start))
            difference = float((objective(start + step) - objective(start - step)) / (2 * step))
            assert analytic == pytest.approx(difference, rel=1e-6)
            assert abs(analytic) > 1e-2, "the volume must actually depend on this parameter"

    def test_the_bore_radius_moves_the_bore_and_nothing_else(self, parametric):
        import jax
        import jax.numpy as jnp

        from cadjoint.extraction import extract_parameters

        scene, brep, mesh = parametric
        cylinder = next(face for face in brep.faces if face.kind == "cylinder")
        rows = jnp.asarray(np.flatnonzero(mesh.owner_face == cylinder.index))
        with _x64():
            baseline, _fixed, _metadata = extract_parameters(scene)
            nominal = {key: jnp.asarray(value, jnp.float64) for key, value in baseline.items()}

            def mean_radius(value):
                points = parameterised_points(scene, mesh, {**nominal, "bore": value}, steps=10)
                return jnp.sqrt(points[rows, 0] ** 2 + points[rows, 1] ** 2).mean()

            # Every node on the cylinder rides the radius one for one, and
            # the midsides are in that set — which is the claim.
            assert float(jax.grad(mean_radius)(nominal["bore"])) == pytest.approx(1.0, abs=1e-6)


# ── the blends ───────────────────────────────────────────────────────────


class TestBlendsAreOwnedByTheScene:
    """A smooth union's faces belong to no patch, so the scene holds them."""

    def test_the_thermal_body_has_blend_nodes(self, thermal_mesh):
        assert thermal_mesh.stats["blend_nodes"] > 0
        assert thermal_mesh.stats["blend_surfaces"] > 0
        assert thermal_mesh.blend_mask.sum() == thermal_mesh.stats["blend_nodes"]
        # A blend node has no patch, so it has no arity either.
        assert (thermal_mesh.owner_arity[thermal_mesh.blend_mask] == 0).all()
        assert (thermal_mesh.owner_patches[thermal_mesh.blend_mask] == -1).all()

    def test_blend_nodes_are_spread_over_several_faces(self, thermal_brep, thermal_mesh):
        counts = thermal_mesh.blend_nodes_by_face()
        assert len(counts) > 1
        blends = {
            index for index in counts if index >= 0 and thermal_brep.faces[index].kind == "blend"
        }
        assert blends, "the starter's smooth unions must produce blend faces"

    def test_re_solving_blends_without_a_scene_is_refused(self, thermal_brep, thermal_mesh):
        with pytest.raises(ValueError, match="blend faces"):
            recompute_gmsh_points(thermal_brep, thermal_mesh)

    def test_the_scene_pulls_blend_nodes_onto_its_own_zero_set(
        self, starter_namespace, thermal_brep, thermal_mesh
    ):
        import jax
        import jax.numpy as jnp

        body = starter_namespace["thermal_body"]
        solved = recompute_gmsh_points(thermal_brep, thermal_mesh, scene=body)
        rows = np.flatnonzero(thermal_mesh.blend_mask)
        before = np.abs(np.asarray(jax.vmap(body)(jnp.asarray(thermal_mesh.points[rows]))))
        after = np.abs(np.asarray(jax.vmap(body)(jnp.asarray(solved[rows]))))
        # The STEP carries the blends as facets, so Gmsh's nodes sit on the
        # chord; the projection is what puts them on the surface itself.
        assert after.max() < 1e-4
        assert after.max() < before.max()

    def test_the_re_solve_does_not_spoil_the_mesh(
        self, starter_namespace, thermal_brep, thermal_mesh
    ):
        # The regression this guards is the one that made
        # :func:`~cadjoint.brep.mesh_gmsh._surface_owner` confirm a match on
        # the worst node rather than the median: an entity straddling a
        # blend used to pass as the neighbouring plane, and the projection
        # then dragged its outlying nodes onto that plane's unbounded
        # extension.  A re-solve at the nominal design must be a *repair* of
        # the STEP's faceting, never a degradation.
        body = starter_namespace["thermal_body"]
        solved = recompute_gmsh_points(thermal_brep, thermal_mesh, scene=body)
        before = tet_radius_ratios(thermal_mesh.points, thermal_mesh.cells).min()
        after = tet_radius_ratios(solved, thermal_mesh.cells).min()
        assert after >= 0.95 * before, f"worst radius ratio {before:.4f} -> {after:.4f}"
        # Nothing moves further than the chord of the facets the graph wrote,
        # which is bounded by the cell the tessellation came off.
        moved = np.linalg.norm(solved - thermal_mesh.points, axis=1)
        assert moved.max() < float(np.linalg.norm(thermal_brep.grid.spacing))

    def test_an_owned_node_never_moves_further_than_the_blend_bar(
        self, starter_namespace, thermal_brep, thermal_mesh
    ):
        # The bar a patch had to clear to own the node is also the distance
        # the node could have been from it, so it bounds the whole re-solve
        # — and that bound is the guarantee.  Before ``_owner_rows`` applied
        # the bar at the node (a curve node used to inherit its surfaces'
        # patches unchecked) the worst owned move here was 1.7e-2, nearly
        # six times a bar of 3.0e-3: nodes on the blend being dragged onto a
        # neighbouring plane's unbounded extension.
        body = starter_namespace["thermal_body"]
        solved = recompute_gmsh_points(thermal_brep, thermal_mesh, scene=body)
        extent = np.asarray(thermal_brep.grid.spacing) * np.asarray(thermal_brep.grid.cells)
        bar = 1e-3 * float(np.linalg.norm(extent))
        owned = thermal_mesh.owner_arity > 0
        assert owned.sum() > 1000
        moved = np.linalg.norm(solved[owned] - thermal_mesh.points[owned], axis=1)
        assert moved.max() <= bar, f"bar {bar:.2e}, worst owned move {moved.max():.2e}"
        # And most of them do not move at all: an exact face leaves the STEP
        # exact, so its nodes arrive already on their own patch.
        assert float(np.median(moved)) < 1e-6


# ── the plugin ───────────────────────────────────────────────────────────


class TestThePluginSlot:
    """``tet_mesher`` is a registry kind, and ``tet_gmsh`` is what fills it."""

    def test_the_served_plugin_answers_exactly_as_the_import_does(self, plate_brep, plate_mesh):
        # The licence argument only holds if the ABI route is the *same*
        # route: if going through the Tesseract changed the answer, the
        # container would not be an isolation of this code but a fork of it.
        pytest.importorskip("tesseract_core")
        served = gmsh_tet_mesh(plate_brep, target_size=TARGET_SIZE, order=2, plugin="tet_gmsh")
        assert np.array_equal(served.points, plate_mesh.points)
        assert np.array_equal(served.cells, plate_mesh.cells)
        assert np.array_equal(served.owner_patches, plate_mesh.owner_patches)
        assert np.array_equal(served.owner_arity, plate_mesh.owner_arity)
        assert served.num_surface == plate_mesh.num_surface

    def test_the_slot_resolves_by_kind_and_declares_what_it_can_do(self):
        pytest.importorskip("tesseract_core")
        from cadjoint.plugins import plugin_for_kind

        plugin = plugin_for_kind(TET_MESHER_KIND)
        assert plugin.name == "tet_gmsh"
        capabilities = plugin.capabilities
        assert capabilities.differentiable_inputs == frozenset({"node_positions"})
        assert capabilities.differentiable_outputs == frozenset({"nodes"})
        assert capabilities.supports("vjp")
        assert capabilities.supports("frozen_topology")
        assert plugin.probe().status == "ok"

    def test_the_kind_is_the_one_the_registry_knows(self):
        from cadjoint.plugins.registry import BUILTIN_DEFAULTS, BUILTIN_PACKAGES, KINDS

        assert TET_MESHER_KIND in KINDS
        assert BUILTIN_PACKAGES["tet_gmsh"][0] == TET_MESHER_KIND
        assert BUILTIN_DEFAULTS[TET_MESHER_KIND] == "tet_gmsh"

    def test_the_package_is_a_complete_tesseract(self):
        from cadjoint.plugins.registry import BUILTIN_PACKAGES

        package = BUILTIN_PACKAGES["tet_gmsh"][1]
        for name in ("tesseract_api.py", "tesseract_config.yaml", "tesseract_requirements.txt"):
            assert (package / name).is_file(), name

    def test_the_vjp_is_a_pass_through(self):
        # The whole differentiable contract in one assertion: the frozen
        # call returns node_positions unchanged, so its transpose is the
        # identity on the cotangent, exactly and with no tolerance.
        from cadjoint.fem.tesseracts.tet_gmsh import tesseract_api

        cells = np.array([[0, 1, 2, 3]], np.int32)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        inputs = tesseract_api.InputSchema(
            step="",
            target_size=np.float64(1.0),
            order=np.int32(1),
            algorithm=np.int32(10),
            node_positions=positions,
            node_ids=np.arange(4, dtype=np.int32),
            cell_template=cells,
            entity_dim_template=np.zeros(4, np.int32),
            bounding_template=np.zeros((4, 1), np.int32),
            edge_parent_template=np.zeros((0, 2), np.int32),
        )
        served = tesseract_api.apply(inputs)
        assert np.asarray(served.nodes) == pytest.approx(positions)
        assert np.asarray(served.cells) == pytest.approx(cells)

        cotangent = np.arange(12, dtype=np.float64).reshape(4, 3)
        back = tesseract_api.vector_jacobian_product(
            inputs, {"node_positions"}, {"nodes"}, {"nodes": cotangent}
        )
        assert (back["node_positions"] == cotangent).all()

    def test_a_discovery_call_carries_no_derivative(self):
        from cadjoint.fem.tesseracts.tet_gmsh import tesseract_api

        inputs = tesseract_api.InputSchema(
            step="",
            target_size=np.float64(1.0),
            order=np.int32(2),
            algorithm=np.int32(10),
            node_positions=np.zeros((0, 3)),
            node_ids=np.zeros(0, np.int32),
            cell_template=np.zeros((0, 0), np.int32),
            entity_dim_template=np.zeros(0, np.int32),
            bounding_template=np.zeros((0, 0), np.int32),
            edge_parent_template=np.zeros((0, 2), np.int32),
        )
        with pytest.raises(ValueError, match="frozen call"):
            tesseract_api.vector_jacobian_product(
                inputs, {"node_positions"}, {"nodes"}, {"nodes": np.zeros((0, 3))}
            )
        with pytest.raises(ValueError, match="frozen topology"):
            tesseract_api.abstract_eval(inputs)

    def test_the_topology_endpoint_is_the_whole_black_box(self, plate_brep, tmp_path):
        # gmsh_topology takes STEP text and nothing else -- which is what
        # makes the container boundary possible, and what the GPL boundary
        # needs it to be.
        from cadjoint.brep import save_brep_step

        path = tmp_path / "plate.step"
        save_brep_step(plate_brep, path)
        found = gmsh_topology(path.read_text(), target_size=0.3, order=1)
        assert found["cells"].shape[1] == 4
        assert found["edge_parents"] is None
        assert found["points"].shape[0] == found["entity_dim"].shape[0]
        assert found["bounds"][1] == pytest.approx(list(PLATE_SIZE), abs=1e-6)

    def test_an_unmeshable_order_is_refused(self, plate_brep, tmp_path):
        from cadjoint.brep import save_brep_step

        path = tmp_path / "plate.step"
        save_brep_step(plate_brep, path)
        with pytest.raises(ValueError, match="order must be"):
            gmsh_topology(path.read_text(), target_size=0.3, order=3)
