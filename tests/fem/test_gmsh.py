"""The public Gmsh route: DC surface in, tet10 out, every node tagged.

The private tier owns the map from a design to node *positions*; what is
public is the mesher and the *tag* (``research/two-tier.md`` §1.2, D4, D7).
Four claims are worth a test and the rest is plumbing.

**A triangle soup becomes a part.**  Gmsh has to be handed a solid whose
surfaces are the part's faces or it constrains the mesh to every facet edge.
The public input is the dual-contour surface as an STL through
``classifySurfaces`` + ``createGeometry`` (tutorial ``t13``): the plate comes
back as a dozen CAD surfaces rather than three thousand facets, and the mesh
that follows is sized by the part.

**Ownership is a residual test and nothing else.**  No graph, no vote — for
every Gmsh surface entity ``|f_p|`` is evaluated on its nodes against the
scene's *public* patch decomposition, and a patch whose worst node clears
the bar owns the surface.  Arity then falls out of the entity: one field on
a surface, two on a curve, three at a corner.  The plate is the case where
every one of those counts is known in advance — six planes, one cylinder,
eight box corners.

**The snap is what makes the bore a face.**  An STL's nodes lie on facets,
and on a 0.25 bore at this lattice the chord sags 3.5e-3 — above the 2.7e-3
bar, so the whole bore would read as a blend.  A clamped arity-1 projection
onto the scene (the public :func:`~cadjoint.fem.motion.project_points`, the
DC path's own tool) is applied first and kept only where it confirms *more*
patches.  It buys ownership, not position: the bore's nodes land within the
bar of ``r = 0.25``, which is exactly the node map's precondition.

**The mesh is a real FEM mesh.**  ``SimMesh(mesher="gmsh")`` builds one and
a study solves on it, with no private tier anywhere in the call.

The derivative — what moves these nodes when the design moves — is the
``node_map`` plugin kind, which nothing in this repository provides.  It is
tested where it lives, in the private ``diff-brep`` distribution; what is
tested *here* is that a Gmsh mesh without it is honest frozen geometry
(``tests/plugins/test_degradation.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint import tier
from cadjoint.fem.gmsh import (
    CLASSIFY_ANGLE,
    TET_MESHER_KIND,
    GmshMesh,
    assign_ownership,
    dc_surface_stl,
    design_values,
    gmsh_available,
    gmsh_tet_mesh,
    gmsh_topology,
    gmsh_version,
    owned_nodes,
    patch_table,
    sdf_gmsh_tet_mesh,
    surface_stl,
    tet_mesh_from_gmsh,
)
from cadjoint.fem.quality import tet_radius_ratios, tet_volumes
from cadjoint.geometry import Scalar, Vector
from cadjoint.meshing.edge_detection import GridSpec
from cadjoint.sdf.boolean import Difference
from cadjoint.sdf.primitives import Box, Cylinder
from cadjoint.sdf.transforms import Translate

pytestmark = pytest.mark.skipif(
    not gmsh_available(), reason="the optional 'gmsh' extra is not installed"
)

#: Coarse enough that the whole module costs about a second of Gmsh, fine
#: enough that the bore still carries several elements around its rim.
TARGET_SIZE = 0.16

#: Half-extents of the test plate and the radius of the bore through it.
#: The plate is the case where every count is known in advance — six planes,
#: one cylinder, twelve straight edges, two rim circles, eight corners — which
#: is what makes the ownership assertions below assertions rather than
#: measurements.
PLATE_SIZE = (0.6, 0.6, 0.4)
BORE_RADIUS = 0.25

#: Spacing 0.083 from -0.83 puts no lattice plane on a plate face, so the
#: dual-contour pass never has to place a vertex on a bit-exact zero.
PLATE_GRID = GridSpec.from_bounds((-0.83, -0.83, -0.63), (1.66, 1.66, 1.26), 20)


def plate_scene():
    """A plate with a through bore: a hard Difference with sharp edges."""
    box = Box(size=Vector(list(PLATE_SIZE)))
    bore = Translate(
        Cylinder(radius=Scalar(BORE_RADIUS), height=Scalar(0.9)),
        Vector([0.0, 0.0, 0.0]),
    )
    return Difference((box, bore), smoothness=0.0)


def plate_volume() -> float:
    """The plate's exact volume, box minus cylinder."""
    return float(
        8.0 * PLATE_SIZE[0] * PLATE_SIZE[1] * PLATE_SIZE[2]
        - np.pi * BORE_RADIUS**2 * 2.0 * PLATE_SIZE[2]
    )


@pytest.fixture(scope="module")
def plate():
    return plate_scene()


@pytest.fixture(scope="module")
def plate_mesh(plate) -> GmshMesh:
    """The plate, dual-contoured, handed to Gmsh as STL, meshed to TET10."""
    return sdf_gmsh_tet_mesh(plate, PLATE_GRID, order=2, target_size=TARGET_SIZE)


@pytest.fixture(scope="module")
def unsnapped(plate) -> GmshMesh:
    """The same route with the snap off — the comparison the snap is judged on."""
    return sdf_gmsh_tet_mesh(plate, PLATE_GRID, order=2, target_size=TARGET_SIZE, snap=False)


def _bore_patch(plate) -> int:
    """The global patch index of the bore's cylinder, found by evaluation."""
    import jax
    import jax.numpy as jnp

    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    ring = np.stack(
        [BORE_RADIUS * np.cos(angles), BORE_RADIUS * np.sin(angles), np.zeros(32)], axis=1
    )
    probes = jnp.asarray(ring, dtype=jnp.float32)
    worst = [
        float(np.abs(np.asarray(jax.vmap(field)(probes))).max()) for field in patch_table(plate)
    ]
    return int(np.argmin(worst))


def test_the_wheel_reports_a_version():
    assert gmsh_version().split(".")[0].isdigit()


# ── the geometry: a triangle soup Gmsh can read as a part ────────────────────


class TestTheDcSurfaceIsTheInput:
    """Option (iii) of the memo: STL in, classified into faces by Gmsh."""

    def test_the_stl_is_ascii_and_closes(self, plate):
        stl, surface = dc_surface_stl(plate, PLATE_GRID)
        assert stl.startswith("solid ")
        assert stl.count("facet normal") == np.asarray(surface.faces).shape[0]
        assert surface_stl(surface) == stl

    def test_classification_turns_facets_into_a_handful_of_faces(self, plate):
        """3 000 facets in, a dozen surfaces out — the whole point of (iii)."""
        stl, surface = dc_surface_stl(plate, PLATE_GRID)
        found = gmsh_topology(stl, geometry_format="stl", target_size=TARGET_SIZE, order=1)
        entities = found["cad_entities"]
        assert entities[3] == 1, "one volume"
        assert entities[2] < 0.05 * np.asarray(surface.faces).shape[0]
        assert entities[2] >= 7, "six sides and a bore, at least"

    def test_the_element_size_is_the_part_not_the_lattice(self, plate_mesh):
        """A lattice-sized mesh would carry an order of magnitude more cells."""
        assert plate_mesh.stats["target_size"] == TARGET_SIZE
        assert plate_mesh.cells.shape[0] < 4000
        assert plate_mesh.stats["dc_triangles"] > 10 * plate_mesh.cells.shape[0] / 10

    def test_the_classification_angle_is_a_named_constant(self):
        assert 20.0 < CLASSIFY_ANGLE < 80.0

    def test_an_unmeshable_order_is_refused(self, plate):
        stl, _surface = dc_surface_stl(plate, PLATE_GRID)
        with pytest.raises(ValueError, match="order must be"):
            gmsh_topology(stl, geometry_format="stl", target_size=0.3, order=3)

    def test_an_unknown_geometry_format_names_the_two(self, plate):
        stl, _surface = dc_surface_stl(plate, PLATE_GRID)
        with pytest.raises(ValueError, match="step|stl"):
            gmsh_topology(stl, geometry_format="iges", target_size=0.3, order=1)


# ── the mesh itself ──────────────────────────────────────────────────────────


class TestThePlateMeshes:
    """A hard CSG solid meshed through the public route alone."""

    def test_it_is_a_tet10_mesh_of_positive_volume(self, plate_mesh):
        assert plate_mesh.order == 2
        assert plate_mesh.cells.shape[1] == 10
        volumes = tet_volumes(plate_mesh.points, plate_mesh.cells)
        assert (volumes > 0).all(), "cells should come back positively oriented"
        # The DC surface is the plate to a lattice cell, and the bore is an
        # inscribed polygon of it, so the straight-sided volume is close but
        # not exact.
        assert volumes.sum() == pytest.approx(plate_volume(), rel=0.02)

    def test_the_bounding_box_is_the_plate(self, plate_mesh):
        low = plate_mesh.points.min(axis=0)
        high = plate_mesh.points.max(axis=0)
        assert low == pytest.approx([-s for s in PLATE_SIZE], abs=2e-2)
        assert high == pytest.approx(list(PLATE_SIZE), abs=2e-2)

    def test_a_solid_of_the_wrong_extent_is_refused(self, plate):
        """The backstop against a silent STEP unit conversion."""
        stl, _surface = dc_surface_stl(plate, PLATE_GRID)
        with pytest.raises(RuntimeError, match="unit conversion|other than the part"):
            gmsh_tet_mesh(
                stl,
                plate,
                grid=PLATE_GRID,
                target_size=TARGET_SIZE,
                order=1,
                expected_bounds=np.array([[-600.0, -600.0, -400.0], [600.0, 600.0, 400.0]]),
            )

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
        # The DC/TetGen path measures 0.04 on this plate; Gmsh has no lattice
        # to graze a feature with, so the bar is an order up.  Measured 0.293
        # from the STL against 0.308 from an analytic STEP of the same part.
        assert ratios.min() > 0.15
        assert float(np.median(ratios)) > 0.7

    def test_the_midside_block_is_in_meshio_order(self, plate_mesh):
        # Gmsh's tet10 swaps the last two midsides against meshio's; a wrong
        # remap puts a midside against the wrong corner pair.  The measure is
        # the bow off the chord midpoint, which is *not* zero here — a
        # midside on the bore's arc is meant to bow, and the widest does so
        # by a third of its edge (HXT's tetrahedralisation varies run to
        # run, so only the bulk of the distribution is pinned) — the swap is
        # caught by comparing the two orderings, not by a threshold.
        from cadjoint.fem.elements import TET10_EDGES

        cells = plate_mesh.cells
        corners = plate_mesh.points[cells[:, :4]]
        midpoints = corners[:, TET10_EDGES].mean(axis=2)
        actual = plate_mesh.points[cells[:, 4:]]
        edges = np.linalg.norm(
            corners[:, TET10_EDGES[:, 1]] - corners[:, TET10_EDGES[:, 0]], axis=-1
        )
        bow = np.linalg.norm(actual - midpoints, axis=-1) / edges
        assert float(np.median(bow)) < 1e-9, "a flat face's midsides sit on the chord"
        assert float(np.percentile(bow, 99)) < 0.15, "and a curved one's bows a little"

        swapped = np.linalg.norm(actual[:, [0, 1, 2, 3, 5, 4]] - midpoints, axis=-1) / edges
        assert bow.mean() < 0.1 * swapped.mean(), "Gmsh's own order would be far worse"

    def test_edge_parents_matches_the_connectivity(self, plate_mesh):
        from cadjoint.fem.elements import TET10_EDGES

        cells = plate_mesh.cells
        pairs = np.sort(cells[:, :4][:, TET10_EDGES], axis=2)
        rows = cells[:, 4:] - plate_mesh.num_corner_points
        assert (plate_mesh.edge_parents[rows] == pairs).all()

    def test_order_one_is_a_tet4_mesh_with_no_midside_block(self, plate):
        mesh = sdf_gmsh_tet_mesh(plate, PLATE_GRID, order=1, target_size=0.3)
        assert mesh.order == 1
        assert mesh.cells.shape[1] == 4
        assert mesh.edge_parents is None
        assert mesh.owned.order == 1
        assert mesh.owned.edge_parents.shape == (0, 2)


# ── ownership by residual, no graph ──────────────────────────────────────────


class TestEveryNodeIsOwnedByResidualAlone:
    """Arity is codimension, capped: the entity says how many fields."""

    def test_arity_is_at_most_the_codimension_of_the_entity(self, plate_mesh):
        # ``3 - dim`` is the *cap*, not the count: the plate's two circle-seam
        # vertices are dim-0 entities where only two patches meet, so they
        # solve two fields and would be over-determined by three.
        for dim in (0, 1, 2, 3):
            arities = plate_mesh.owner_arity[plate_mesh.entity_dim == dim]
            assert (arities <= 3 - dim).all(), f"dim {dim} over-determined"

    def test_a_volume_node_is_owned_by_nothing(self, plate_mesh):
        interior = plate_mesh.entity_dim == 3
        assert (plate_mesh.owner_arity[interior] == 0).all()
        assert not plate_mesh.blend_mask[interior].any()

    def test_the_owner_row_carries_exactly_that_many_patches(self, plate_mesh):
        filled = (plate_mesh.owner_patches >= 0).sum(axis=1)
        assert (filled == plate_mesh.owner_arity).all()
        assert (plate_mesh.owner_patches[:, 0][plate_mesh.owner_arity > 0] >= 0).all()

    def test_the_eight_box_corners_are_the_only_three_field_nodes(self, plate_mesh):
        corners = np.flatnonzero(plate_mesh.owner_arity == 3)
        assert corners.size == 8, "a box has eight corners; the bore rims are curves"
        magnitudes = np.abs(plate_mesh.points[corners])
        assert magnitudes == pytest.approx(np.broadcast_to(PLATE_SIZE, (8, 3)), abs=2e-2)

    def test_every_owned_node_actually_satisfies_its_own_fields(self, plate, plate_mesh):
        """The claim ownership makes: ``|f_p(x)| <= bar`` at every node it tags."""
        import jax
        import jax.numpy as jnp

        fields = patch_table(plate)
        rows = np.flatnonzero(plate_mesh.owner_arity > 0)
        probes = jnp.asarray(plate_mesh.points[rows], dtype=jnp.float32)
        residuals = np.abs(
            np.stack([np.asarray(jax.vmap(field)(probes)) for field in fields], axis=-1)
        )
        for slot in range(3):
            owners = plate_mesh.owner_patches[rows, slot]
            live = np.flatnonzero(owners >= 0)
            worst = residuals[live, owners[live]].max()
            assert worst <= plate_mesh.stats["bar"], f"slot {slot} residual {worst}"

    def test_nothing_on_the_snapped_plate_is_a_blend(self, plate_mesh):
        """A hard CSG solid has no blend faces, so no node may read as one."""
        assert not plate_mesh.blend_mask.any()
        assert plate_mesh.stats["blend_surfaces"] == 0

    def test_the_record_is_the_one_the_node_map_consumes(self, plate, plate_mesh):
        owned = plate_mesh.owned
        assert owned.count == plate_mesh.points.shape[0]
        assert np.array_equal(owned.seeds, plate_mesh.points)
        assert owned.bar == plate_mesh.stats["bar"]
        assert owned.num_corner == plate_mesh.num_corner_points
        assert owned.num_surface == plate_mesh.num_surface
        assert owned.arity_counts() == plate_mesh.stats["arity_counts"]
        # Every patch index names a patch of the public table.
        assert owned.patches.max() < len(patch_table(plate))

    def test_the_design_travels_with_the_record(self, plate_mesh):
        """So the map can refuse an ``OwnedNodes`` from another design."""
        assert set(plate_mesh.owned.design) == set(design_values(plate_scene()))
        assert plate_mesh.owned.design_digest()

    def test_a_plain_callable_has_no_table_so_every_node_is_a_blend(self, plate):
        """No decomposition, no ownership — and no crash."""
        import jax.numpy as jnp

        field = lambda point: jnp.asarray(plate(point))  # noqa: E731
        mesh = sdf_gmsh_tet_mesh(field, PLATE_GRID, order=1, target_size=0.3)
        assert mesh.stats["patches"] == 0
        boundary = mesh.entity_dim < 3
        assert mesh.blend_mask[boundary].all()
        assert (mesh.owner_arity == 0).all()


class TestTheSnapBuysOwnershipNotPosition:
    """The measured claim of the module docstring (D4)."""

    def test_without_the_snap_the_bore_reads_as_a_blend(self, unsnapped):
        assert unsnapped.stats["snapped_nodes"] == 0
        assert unsnapped.stats["blend_surfaces"] > 0
        assert unsnapped.blend_mask.any()

    def test_with_it_the_bore_is_a_face_and_the_quality_is_unchanged(self, plate_mesh, unsnapped):
        assert plate_mesh.stats["snapped_nodes"] > 0
        assert plate_mesh.stats["blend_surfaces"] == 0
        assert plate_mesh.stats["arity_counts"][1] > unsnapped.stats["arity_counts"][1]
        before = tet_radius_ratios(unsnapped.points, unsnapped.cells).min()
        after = tet_radius_ratios(plate_mesh.points, plate_mesh.cells).min()
        assert after >= 0.95 * before, f"worst radius ratio {before:.4f} -> {after:.4f}"

    def test_no_node_moves_further_than_the_bar(self, plate_mesh, unsnapped):
        """Clamped: the snap is a tag repair, not a re-placement."""
        moved = np.linalg.norm(plate_mesh.points - unsnapped.points, axis=1)
        assert moved.max() <= plate_mesh.stats["bar"] + 1e-12

    def test_the_bore_nodes_land_inside_the_bar_of_the_true_radius(self, plate, plate_mesh):
        """The node map's precondition: seeds within ``bar`` of their patch."""
        bore = _bore_patch(plate)
        owned = (plate_mesh.owner_patches == bore).any(axis=1)
        assert owned.sum() > 50, "the bore should own a ring of nodes"
        radius = np.linalg.norm(plate_mesh.points[owned, :2], axis=1)
        error = np.abs(radius - BORE_RADIUS)
        assert error.max() <= plate_mesh.stats["bar"]
        # And most of them land on it outright, midsides included.
        assert float(np.median(error)) < 1e-6
        midsides = owned.copy()
        midsides[: plate_mesh.num_corner_points] = False
        assert midsides.sum() > 20, "the bore's arcs carry midsides"


class TestOwnershipIsIndependentOfTheMesher:
    """``assign_ownership`` takes a topology and a table, and nothing else."""

    def test_it_tags_a_topology_handed_to_it_directly(self, plate):
        stl, _surface = dc_surface_stl(plate, PLATE_GRID)
        topology = gmsh_topology(stl, geometry_format="stl", target_size=0.3, order=1)
        bar = 1e-3 * float(
            np.linalg.norm(np.asarray(PLATE_GRID.spacing) * np.asarray(PLATE_GRID.cells))
        )
        tagged = assign_ownership(patch_table(plate), topology, bar=bar)
        assert tagged["owner_patches"].shape == (topology["points"].shape[0], 3)
        assert tagged["stats"]["patches"] == len(patch_table(plate))
        assert sum(tagged["stats"]["arity_counts"].values()) == topology["points"].shape[0]

    def test_owned_nodes_wraps_the_same_answer_as_a_record(self, plate):
        stl, _surface = dc_surface_stl(plate, PLATE_GRID)
        topology = gmsh_topology(stl, geometry_format="stl", target_size=0.3, order=1)
        bar = 2.664e-3
        tagged = assign_ownership(patch_table(plate), topology, bar=bar)
        record, stats = owned_nodes(patch_table(plate), topology, bar=bar)
        assert np.array_equal(record.patches, tagged["owner_patches"])
        assert np.array_equal(record.arity, tagged["owner_arity"])
        assert np.array_equal(record.blend, tagged["blend_mask"])
        assert stats == tagged["stats"]


# ── the handover to the FEM layer ────────────────────────────────────────────


class TestTheFemHandover:
    """A ``TetMesh`` like any other, plus the ownership record."""

    def test_the_wrapped_mesh_is_a_valid_tet10_mesh(self, plate_mesh):
        mesh = tet_mesh_from_gmsh(plate_mesh, grid=PLATE_GRID)
        assert mesh.order == 2
        assert mesh.ele_type == "TET10"
        assert mesh.num_corner_points == plate_mesh.num_corner_points
        assert mesh.boundary_tris.shape[1] == 3
        assert (tet_volumes(mesh.points, mesh.cells) > 0).all()
        assert mesh.edge_parents is not None

    def test_it_carries_the_record_and_says_which_mesher_made_it(self, plate_mesh):
        mesh = tet_mesh_from_gmsh(plate_mesh, grid=PLATE_GRID)
        assert mesh.mesher == "gmsh"
        assert mesh.owned is plate_mesh.owned
        assert mesh.stats["geometry_format"] == "stl"

    def test_the_positions_are_the_seeds(self, plate_mesh):
        """Static: what moves them is the ``node_map`` kind, not this call."""
        mesh = tet_mesh_from_gmsh(plate_mesh, grid=PLATE_GRID)
        assert np.array_equal(mesh.points, plate_mesh.owned.seeds)
        assert np.array_equal(mesh.base_points, plate_mesh.points)


class TestSimMeshTakesTheMesherAsAKeyword:
    """D7: ``mesher="gmsh"`` is a public keyword with a public implementation."""

    def test_the_option_set_is_the_two_meshers(self):
        from cadjoint.enums import TetMesher, values

        assert values(TetMesher) == ("tetgen", "gmsh")

    def test_an_unknown_mesher_names_the_accepted_ones(self):
        from cadjoint.fem.simmesh import SimMesh

        with pytest.raises(ValueError, match=r"mesher must be one of \['tetgen', 'gmsh'\]"):
            SimMesh(name="m", resolution=8, method="tet10", mesher="netgen")

    def test_a_hex_mesh_has_no_mesher_to_choose(self):
        from cadjoint.fem.simmesh import SimMesh

        with pytest.raises(ValueError, match="applies to the tet methods"):
            SimMesh(name="m", resolution=8, method="hex", mesher="gmsh")

    def test_the_mesher_is_part_of_the_cache_key(self, plate):
        """Changing the mesher must rebuild, not hand back the TetGen mesh."""
        from cadjoint.fem.simmesh import SimMesh

        common = {"resolution": 12, "method": "tet4", "bounds": [-0.9] * 3, "size": [1.8] * 3}
        tetgen = SimMesh(name="m", **common).build(plate)
        gmsh = SimMesh(name="m", mesher="gmsh", **common).build(plate)
        assert getattr(tetgen, "mesher", "tetgen") == "tetgen"
        assert gmsh.mesher == "gmsh"
        assert gmsh.num_points != tetgen.num_points

    def test_the_declaration_and_the_inspection_carry_the_choice(self, plate):
        from cadjoint.fem.simmesh import SimMesh

        declared = SimMesh(
            name="plate",
            resolution=12,
            method="tet10",
            mesher="gmsh",
            bounds=[-0.9] * 3,
            size=[1.8] * 3,
        )
        assert declared.describe()["mesher"] == "gmsh"
        report = declared.inspect(plate)
        assert report["mesher"] == "gmsh"
        assert report["nodes"] > 0 and report["elements"] > 0
        # Public cadjoint alone has no `node_map` provider, so the mesh says
        # its geometry is frozen: the nodes are right, but they cannot follow
        # a design parameter without the private tier. `tests/plugins/
        # test_degradation.py` covers the filled half against stub providers.
        assert report["frozen_geometry"] is not tier.available("node_map")


class TestAStudySolvesOnAGmshMesh:
    """The end of the public path: mesh with Gmsh, solve, no private tier."""

    @pytest.fixture(scope="class")
    def solved(self):
        from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
        from cadjoint.fem.simmesh import SimMesh
        from cadjoint.geometry import Vector
        from cadjoint.sdf.primitives import Box

        scene = Box(Vector([0.8, 0.5, 0.5]))
        mesh = SimMesh(
            name="bar",
            resolution=12,
            method="tet10",
            mesher="gmsh",
            bounds=[-1.1] * 3,
            size=[2.2] * 3,
        )
        study = ThermalStudy(
            name="bar",
            mesh=mesh,
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 0.0), Dirichlet(Nodes.side("+x"), 100.0)],
        )
        return study, study.solve(scene)

    def test_the_solve_spans_the_two_boundary_conditions(self, solved):
        _study, result = solved
        temperature = np.asarray(result.temperature)
        assert temperature.min() == pytest.approx(0.0, abs=1e-6)
        assert temperature.max() == pytest.approx(100.0, abs=1e-3)

    def test_the_field_is_monotone_along_the_gradient(self, solved):
        """A conduction bar's temperature rises with x and nothing else."""
        _study, result = solved
        points = np.asarray(result.mesh.points)
        temperature = np.asarray(result.temperature)
        order = np.argsort(points[:, 0])
        smoothed = np.interp(
            np.linspace(points[:, 0].min(), points[:, 0].max(), 20),
            points[order, 0],
            temperature[order],
        )
        assert (np.diff(smoothed) > -1e-6).all()

    def test_the_solved_mesh_is_the_gmsh_one(self, solved):
        _study, result = solved
        assert result.mesh.ele_type == "TET10"
        assert getattr(result.mesh, "mesher", None) == "gmsh"
        assert result.mesh.owned is not None


# ── the plugin and its package ───────────────────────────────────────────────


class TestThePluginSlot:
    """``tet_mesher`` is a registry kind and ``tet_gmsh`` is what fills it."""

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

    def test_the_package_needs_no_private_module(self):
        """The GPL image must never contain the private tier.

        Gmsh is GPL-2.0-or-later and ``diff-brep`` is proprietary, so the
        one thing that must stay true by construction is that they never
        share a process: the ``cadjoint_tet_gmsh`` image is built from this
        package alone, and this package names nothing private
        (``research/two-tier.md`` §3.5).
        """
        from cadjoint.plugins.registry import BUILTIN_PACKAGES

        package = BUILTIN_PACKAGES["tet_gmsh"][1]
        for name in ("tesseract_api.py", "tesseract_requirements.txt"):
            text = (package / name).read_text()
            assert "diff_brep" not in text and "diff-brep" not in text, name

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

    def test_the_served_plugin_answers_exactly_as_the_import_does(self, plate, plate_mesh):
        # The licence argument only holds if the ABI route is the *same*
        # route: if going through the Tesseract changed the answer, the
        # container would not be an isolation of this code but a fork of it.
        pytest.importorskip("tesseract_core")
        served = sdf_gmsh_tet_mesh(
            plate, PLATE_GRID, order=2, target_size=TARGET_SIZE, plugin="tet_gmsh"
        )
        assert np.array_equal(served.points, plate_mesh.points)
        assert np.array_equal(served.cells, plate_mesh.cells)
        assert np.array_equal(served.owner_patches, plate_mesh.owner_patches)
        assert np.array_equal(served.owner_arity, plate_mesh.owner_arity)
        assert served.num_surface == plate_mesh.num_surface

    def test_the_input_schema_carries_a_geometry_and_its_format(self):
        from cadjoint.fem.tesseracts.tet_gmsh import tesseract_api

        fields = tesseract_api.InputSchema.model_fields
        assert "geometry" in fields and "geometry_format" in fields
        assert fields["geometry_format"].default == "stl"
        assert "step" not in fields, "renamed to 'geometry' when STL became an input"

    def test_the_vjp_is_a_pass_through(self):
        # The whole differentiable contract in one assertion: the frozen
        # call returns node_positions unchanged, so its transpose is the
        # identity on the cotangent, exactly and with no tolerance.
        from cadjoint.fem.tesseracts.tet_gmsh import tesseract_api

        cells = np.array([[0, 1, 2, 3]], np.int32)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        inputs = tesseract_api.InputSchema(
            geometry="",
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
            geometry="",
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
