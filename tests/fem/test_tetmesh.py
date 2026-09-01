"""Tests for the DC-surface -> tet-mesh -> jax-fem prototype (research route).

Everything here skips cleanly when ``tetgen`` is not installed; the solver
tests additionally skip without ``jax_fem``, and the mesher-tesseract tests
without ``tesseract_core``/``tesseract_jax``.  See ``research/tet-vs-hex.md``
for the measurements these tests guard.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tetgen")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cadjoint.fem.backends import ElasticBCs  # noqa: E402
from cadjoint.fem.selection import Nodes  # noqa: E402
from cadjoint.fem.tetmesh import (  # noqa: E402
    TetMesh,
    load_work_tri6,
    load_work_tris,
    recompute_tet_points,
    sdf_to_tet_mesh,
    tet10_from_tet4,
    tet_aspect_ratios,
    tet_elastic_solve,
    tet_faces_from_nodes,
    tet_radius_ratios,
    tet_volumes,
)
from cadjoint.meshing import GridSpec  # noqa: E402


def _sphere_sdf(radius: float = 1.0):
    def sdf(p):
        return jnp.linalg.norm(jnp.asarray(p), axis=-1) - radius

    return sdf


def _bar_sdf(half_height: float = 0.16):
    """Axis-aligned bar with a traced-friendly half-height (off-tie, like the FEM tests)."""

    def sdf(p):
        p = jnp.asarray(p)
        q = jnp.abs(p) - jnp.stack([jnp.asarray(0.5), jnp.asarray(half_height), jnp.asarray(0.15)])
        return jnp.max(q, axis=-1)

    return sdf


_SPHERE_GRID = GridSpec.from_bounds((-1.4, -1.4, -1.4), (2.8, 2.8, 2.8), (11, 11, 11))
_BAR_GRID = GridSpec.from_bounds((-0.65, -0.32, -0.3), (1.3, 0.64, 0.6), (13, 7, 6))


@pytest.fixture(scope="module")
def sphere_mesh() -> TetMesh:
    return sdf_to_tet_mesh(_sphere_sdf(), _SPHERE_GRID, sharp=False)


@pytest.fixture(scope="module")
def bar_mesh() -> TetMesh:
    return sdf_to_tet_mesh(_bar_sdf(), _BAR_GRID)


class TestMeshExtraction:
    def test_boundary_vertices_are_the_leading_block(self, sphere_mesh):
        # tetgen -Y preserves the input surface: the boundary node set of
        # the volume mesh is exactly the leading DC-vertex block.
        boundary_nodes = np.unique(sphere_mesh.boundary_tris)
        assert np.array_equal(boundary_nodes, np.arange(sphere_mesh.num_surface))

    def test_boundary_vertices_lie_on_the_surface(self, sphere_mesh):
        values = np.asarray(
            _sphere_sdf()(jnp.asarray(sphere_mesh.points[: sphere_mesh.num_surface]))
        )
        assert np.abs(values).max() < 1e-6

    def test_steiner_vertices_are_interior(self, sphere_mesh):
        values = np.asarray(
            _sphere_sdf()(jnp.asarray(sphere_mesh.points[sphere_mesh.num_surface :]))
        )
        assert values.max() < 0.0

    def test_all_volumes_positive(self, sphere_mesh):
        assert tet_volumes(sphere_mesh.points, sphere_mesh.cells).min() > 0.0

    def test_boundary_is_watertight(self, sphere_mesh):
        edges = sphere_mesh.boundary_tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        keys, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        assert (counts == 2).all()

    def test_boundary_normals_point_outward(self, sphere_mesh):
        group = sphere_mesh.all_boundary_faces()
        alignment = np.einsum("md,md->m", group.normals, group.centers)
        assert (alignment > 0.0).all()  # radial direction on a sphere


class TestQualityMetrics:
    def test_regular_tet_scores_one(self):
        points = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], dtype=np.float64)
        cells = np.array([[0, 1, 2, 3]])
        assert tet_radius_ratios(points, cells)[0] == pytest.approx(1.0, abs=1e-12)
        assert tet_aspect_ratios(points, cells)[0] == pytest.approx(1.0, abs=1e-12)

    def test_sliver_scores_low(self):
        points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0.5, 1e-3)], dtype=np.float64)
        cells = np.array([[0, 1, 2, 3]])
        assert tet_radius_ratios(points, cells)[0] < 0.02


class TestVonMises:
    def test_linear_field_is_exact_for_both_orders(self):
        from cadjoint.fem.tetmesh import _TET10_EDGES, tet_von_mises

        rng = np.random.default_rng(0)
        points = rng.standard_normal((4, 3))
        gradient = np.array([[0.002, 0.001, 0.0], [0.0, -0.001, 0.0005], [0.0003, 0.0, 0.004]])
        strain = 0.5 * (gradient + gradient.T)
        lam = 1000.0 * 0.3 / (1.3 * 0.4)
        mu = 1000.0 / 2.6
        stress = lam * np.trace(strain) * np.eye(3) + 2 * mu * strain
        deviator = stress - np.trace(stress) / 3.0 * np.eye(3)
        exact = np.sqrt(1.5 * (deviator * deviator).sum())

        cells4 = np.array([[0, 1, 2, 3]])
        vm4 = tet_von_mises(points, cells4, points @ gradient.T, youngs=1000.0, poisson=0.3)
        assert vm4[0] == pytest.approx(exact, rel=1e-12)

        midsides = points[_TET10_EDGES].mean(axis=1)
        points10 = np.concatenate([points, midsides])
        cells10 = np.arange(10, dtype=np.int64).reshape(1, 10)
        vm10 = tet_von_mises(points10, cells10, points10 @ gradient.T, youngs=1000.0, poisson=0.3)
        assert vm10[0] == pytest.approx(exact, rel=1e-12)


class TestSelections:
    def test_nodes_selections_resolve_on_tet_meshes(self, sphere_mesh):
        # Duck-typed against the HexMesh surface NodeSelection expects.
        cap = Nodes.halfspace([0.0, 0.0, 0.8], [0.0, 0.0, 1.0]).resolve(sphere_mesh)
        assert cap.size > 0
        assert (sphere_mesh.points[cap][:, 2] >= 0.8 - 1e-9).all()
        assert (cap < sphere_mesh.num_surface).all()  # boundary restriction

    def test_faces_from_nodes_spans_selected_triangles(self, sphere_mesh):
        cap = Nodes.halfspace([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]).resolve(sphere_mesh)
        faces = tet_faces_from_nodes(sphere_mesh, cap)
        assert faces.shape[0] > 0
        assert np.isin(faces, cap).all()


class TestRecompute:
    def test_nominal_recompute_is_exact_fixed_point(self, sphere_mesh):
        points = np.asarray(recompute_tet_points(_sphere_sdf(), sphere_mesh))
        assert np.abs(points - sphere_mesh.points).max() < 1e-12

    def test_tet10_nominal_recompute_is_exact_fixed_point(self, sphere_mesh):
        from cadjoint.fem.tetmesh import tet10_mesh

        promoted = tet10_mesh(sphere_mesh)
        points = np.asarray(recompute_tet_points(_sphere_sdf(), promoted))
        assert points.shape == promoted.points.shape
        assert np.abs(points - promoted.points).max() < 1e-12

    def test_interior_stays_frozen_without_smoothing(self, sphere_mesh):
        points = np.asarray(recompute_tet_points(_sphere_sdf(0.97), sphere_mesh))
        count = sphere_mesh.num_surface
        assert np.array_equal(points[count:], sphere_mesh.base_points[count:])
        boundary_radii = np.linalg.norm(points[:count], axis=1)
        assert np.abs(boundary_radii - 0.97).max() < 1e-6

    def test_gradient_of_boundary_radius_wrt_design(self, sphere_mesh):
        def mean_radius(radius):
            points = recompute_tet_points(_sphere_sdf(radius), sphere_mesh)
            boundary = points[: sphere_mesh.num_surface]
            return jnp.mean(jnp.linalg.norm(boundary, axis=1))

        gradient = jax.grad(mean_radius)(jnp.asarray(1.0))
        assert float(gradient) == pytest.approx(1.0, rel=1e-4)

    def test_laplacian_smoothing_moves_interior_differentiably(self, sphere_mesh):
        def mean_interior_shift(radius):
            points = recompute_tet_points(_sphere_sdf(radius), sphere_mesh, smooth_passes=2)
            interior = points[sphere_mesh.num_surface :]
            frozen = jnp.asarray(sphere_mesh.base_points[sphere_mesh.num_surface :])
            return jnp.mean(jnp.linalg.norm(interior - frozen, axis=1))

        shift = mean_interior_shift(jnp.asarray(0.95))
        assert float(shift) > 1e-4  # boundary motion propagated inward
        gradient = jax.grad(mean_interior_shift)(jnp.asarray(0.95))
        assert np.isfinite(float(gradient))


class TestTet10:
    def test_promotion_counts_and_midpoints(self, sphere_mesh):
        points10, cells10, parents = tet10_from_tet4(sphere_mesh.points, sphere_mesh.cells)
        assert cells10.shape == (sphere_mesh.num_cells, 10)
        assert points10.shape[0] == sphere_mesh.num_points + parents.shape[0]
        assert np.array_equal(cells10[:, :4], np.asarray(sphere_mesh.cells))
        midpoints = points10[sphere_mesh.num_points :]
        expected = sphere_mesh.points[parents].mean(axis=1)
        assert np.abs(midpoints - expected).max() < 1e-12

    def test_midside_order_matches_meshio_tetra10(self, sphere_mesh):
        points10, cells10, _ = tet10_from_tet4(sphere_mesh.points, sphere_mesh.cells)
        cell = cells10[0]
        corners = points10[cell[:4]]
        expected = np.stack(
            [
                (corners[0] + corners[1]) / 2,
                (corners[1] + corners[2]) / 2,
                (corners[2] + corners[0]) / 2,
                (corners[0] + corners[3]) / 2,
                (corners[1] + corners[3]) / 2,
                (corners[2] + corners[3]) / 2,
            ]
        )
        assert np.abs(points10[cell[4:]] - expected).max() < 1e-12


def _bar_bcs(mesh: TetMesh, traction=(0.0, 0.0, -1.0)):
    clamp = Nodes.halfspace([-0.49, 0.0, 0.0], [-1.0, 0.0, 0.0]).resolve(mesh)
    tip_nodes = Nodes.halfspace([0.49, 0.0, 0.0], [1.0, 0.0, 0.0]).resolve(mesh)
    faces = tet_faces_from_nodes(mesh, tip_nodes)
    bcs = ElasticBCs(
        fixed_nodes=[clamp],
        traction_nodes=[np.unique(faces).astype(np.int32)],
        traction_vectors=[np.asarray(traction)],
    )
    return bcs, faces


class TestElasticSolve:
    @pytest.fixture(scope="class")
    def solved_bar(self, bar_mesh):
        pytest.importorskip("jax_fem")
        bcs, faces = _bar_bcs(bar_mesh)
        displacement = tet_elastic_solve(
            bar_mesh.points,
            bar_mesh.cells,
            bcs,
            youngs=1000.0,
            poisson=0.3,
            traction_faces=[faces],
        )
        return bcs, faces, np.asarray(displacement)

    def test_tet4_bends_downward_under_tip_load(self, bar_mesh, solved_bar):
        _, _, displacement = solved_bar
        tip = displacement[np.asarray(bar_mesh.points)[:, 0] > 0.45]
        assert tip[:, 2].mean() < 0.0
        assert np.isfinite(displacement).all()

    def test_clamped_nodes_stay_fixed(self, bar_mesh, solved_bar):
        bcs, _, displacement = solved_bar
        assert np.abs(displacement[bcs.fixed_nodes[0]]).max() < 1e-9

    def test_tet10_is_softer_than_tet4(self, bar_mesh, solved_bar):
        pytest.importorskip("jax_fem")
        bcs4, faces, displacement4 = solved_bar
        work4 = float(
            load_work_tris(bar_mesh.points, displacement4, faces, np.array([0.0, 0.0, -1.0]))
        )
        points10, cells10, parents = tet10_from_tet4(bar_mesh.points, bar_mesh.cells)
        count = bar_mesh.num_points

        def with_midsides(corner_set):
            both = np.isin(parents, corner_set).all(axis=1)
            return np.concatenate([corner_set, count + np.flatnonzero(both)]).astype(np.int32)

        edge_ids = {tuple(sorted(edge)): count + i for i, edge in enumerate(map(tuple, parents))}
        mids = np.array(
            [
                [
                    edge_ids[tuple(sorted((f[0], f[1])))],
                    edge_ids[tuple(sorted((f[1], f[2])))],
                    edge_ids[tuple(sorted((f[2], f[0])))],
                ]
                for f in faces
            ]
        )
        bcs10 = ElasticBCs(
            fixed_nodes=[with_midsides(bcs4.fixed_nodes[0])],
            traction_nodes=[with_midsides(bcs4.traction_nodes[0])],
            traction_vectors=[np.array([0.0, 0.0, -1.0])],
        )
        displacement10 = tet_elastic_solve(
            points10,
            cells10,
            bcs10,
            youngs=1000.0,
            poisson=0.3,
            ele_type="TET10",
            traction_faces=[faces],
        )
        work10 = float(
            load_work_tri6(
                points10,
                displacement10,
                np.concatenate([faces, mids], axis=1),
                np.array([0.0, 0.0, -1.0]),
            )
        )
        # TET4 locks in bending: the quadratic model must be softer.
        assert work10 > work4 > 0.0

    def test_traction_faces_must_be_within_node_set(self, bar_mesh):
        pytest.importorskip("jax_fem")
        bcs, faces = _bar_bcs(bar_mesh)
        wrong = ElasticBCs(
            fixed_nodes=bcs.fixed_nodes,
            traction_nodes=[bcs.traction_nodes[0][:3]],  # too small to span the faces
            traction_vectors=bcs.traction_vectors,
        )
        with pytest.raises(ValueError, match="matched"):
            tet_elastic_solve(
                bar_mesh.points,
                bar_mesh.cells,
                wrong,
                youngs=1000.0,
                poisson=0.3,
                traction_faces=[faces],
            )


class TestEndToEndGradient:
    def test_adjoint_matches_central_fd_on_smooth_geometry(self, sphere_mesh):
        """d(load work)/d(radius) through mesh re-projection + solve, adjoint vs FD.

        Uses the sphere: away from SDF kinks the frozen-topology objective
        is smooth and central FD is a tight comparator.  (On crease-heavy
        geometry the DC vertices sit exactly on subgradient kinks and FD
        legitimately disagrees with the one-sided adjoint — measured in
        ``research/tet-vs-hex.md``, not asserted here.)
        """
        pytest.importorskip("jax_fem")
        clamp = Nodes.halfspace([0.0, 0.0, -0.75], [0.0, 0.0, -1.0]).resolve(sphere_mesh)
        cap = Nodes.halfspace([0.0, 0.0, 0.75], [0.0, 0.0, 1.0]).resolve(sphere_mesh)
        faces = tet_faces_from_nodes(sphere_mesh, cap)
        traction = np.array([0.0, 0.0, -1.0])
        bcs = ElasticBCs(
            fixed_nodes=[clamp],
            traction_nodes=[np.unique(faces).astype(np.int32)],
            traction_vectors=[traction],
        )

        def objective(radius):
            points = recompute_tet_points(_sphere_sdf(radius), sphere_mesh)
            displacement = tet_elastic_solve(
                points,
                sphere_mesh.cells,
                bcs,
                youngs=1000.0,
                poisson=0.3,
                base_points=sphere_mesh.points,
                traction_faces=[faces],
            )
            return load_work_tris(points, displacement, faces, traction)

        gradient = float(jax.grad(objective)(jnp.asarray(1.0)))
        eps = 1e-3
        plus = float(objective(jnp.asarray(1.0 + eps)))
        minus = float(objective(jnp.asarray(1.0 - eps)))
        finite_difference = (plus - minus) / (2.0 * eps)
        assert gradient == pytest.approx(finite_difference, rel=5e-2)


class TestMesherTesseract:
    @pytest.fixture(scope="class")
    def tesseract(self):
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        from pathlib import Path

        from tesseract_core import Tesseract

        api = Path(__file__).parents[2] / "cadjoint" / "fem" / "tesseracts" / "mesher"
        return Tesseract.from_tesseract_api(str(api / "tesseract_api.py"))

    @pytest.fixture(scope="class")
    def sphere_setup(self, tesseract):
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), (12, 12, 12))
        lattice = grid.lattice_points()

        def field_of(radius):
            return jnp.linalg.norm(jnp.asarray(lattice), axis=-1) - radius

        static = {
            "origin": np.asarray(grid.origin),
            "spacing": np.asarray(grid.spacing),
            "element": np.int32(0),
            "sharp": np.int32(0),
            "min_ratio": np.float64(1.5),
            "min_dihedral": np.float64(10.0),
        }
        discovery = tesseract.apply(
            dict(
                field_values=np.asarray(field_of(1.0)),
                point_ids=np.zeros(0, np.int32),
                cell_template=np.zeros((0, 4), np.int32),
                num_surface=np.int32(0),
                **static,
            )
        )
        return grid, field_of, static, discovery

    def test_discovery_apply_meshes_the_sphere(self, sphere_setup):
        _, _, _, discovery = sphere_setup
        points = np.asarray(discovery["points"])
        mask = np.asarray(discovery["surface_mask"]).astype(bool)
        radii = np.linalg.norm(points[mask], axis=1)
        assert np.abs(radii - 1.0).max() < 5e-2  # interpolated sphere
        assert tet_volumes(points, np.asarray(discovery["cells"])).min() > 0.0
        assert mask[: mask.sum()].all()  # leading block

    def test_traced_gradient_through_interpolation_vjp(self, tesseract, sphere_setup):
        from tesseract_jax import apply_tesseract

        _, field_of, static, discovery = sphere_setup
        num_points = len(np.asarray(discovery["points"]))
        num_cells = len(np.asarray(discovery["cells"]))
        num_surface = int(np.asarray(discovery["surface_mask"]).sum())

        def mean_radius(radius):
            outputs = apply_tesseract(
                tesseract,
                dict(
                    field_values=field_of(radius),
                    point_ids=np.arange(num_points, dtype=np.int32),
                    cell_template=np.zeros((num_cells, 4), np.int32),
                    num_surface=np.int32(num_surface),
                    **static,
                ),
            )
            boundary = outputs["points"][:num_surface]
            return jnp.mean(jnp.sqrt(jnp.sum(boundary * boundary, axis=1)))

        gradient = float(jax.grad(mean_radius)(jnp.asarray(1.0)))
        assert gradient == pytest.approx(1.0, rel=5e-2)

    def test_vjp_matches_reference_interpolation_map(self, tesseract, sphere_setup):
        """The tesseract VJP equals the IFT map at frozen positions (plumbing check)."""
        import importlib.util
        from pathlib import Path

        grid, field_of, static, discovery = sphere_setup
        api = (
            Path(__file__).parents[2]
            / "cadjoint"
            / "fem"
            / "tesseracts"
            / "mesher"
            / "tesseract_api.py"
        )
        spec = importlib.util.spec_from_file_location("mesher_tesseract_api", api)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        points = np.asarray(discovery["points"])
        num_surface = int(np.asarray(discovery["surface_mask"]).sum())
        rng = np.random.default_rng(0)
        cotangent = rng.standard_normal(points.shape)
        cotangent[num_surface:] = 0.0
        field0 = np.asarray(field_of(1.0))
        result = tesseract.vector_jacobian_product(
            dict(
                field_values=field0,
                point_ids=np.arange(len(points), dtype=np.int32),
                cell_template=np.zeros((len(np.asarray(discovery["cells"])), 4), np.int32),
                num_surface=np.int32(num_surface),
                **static,
            ),
            vjp_inputs={"field_values"},
            vjp_outputs={"points"},
            cotangent_vector={"points": cotangent},
        )
        field_bar = np.asarray(result["field_values"])

        boundary = jnp.asarray(points[:num_surface])
        interpolant = module.make_interpolant(
            jnp.asarray(field0), np.asarray(grid.origin), np.asarray(grid.spacing)
        )
        gradients = jax.vmap(jax.grad(lambda p: interpolant(p).reshape(())))(boundary)
        squared = jnp.sum(gradients * gradients, axis=1)

        def reference(field_values):
            values = module.make_interpolant(
                field_values, np.asarray(grid.origin), np.asarray(grid.spacing)
            )(boundary)
            return boundary - (values / squared)[:, None] * gradients

        _, vjp_fn = jax.vjp(reference, jnp.asarray(field0))
        (expected,) = vjp_fn(jnp.asarray(cotangent[:num_surface]))
        scale = max(float(np.abs(np.asarray(expected)).max()), 1e-30)
        assert np.abs(field_bar - np.asarray(expected)).max() / scale < 1e-10


class TestMesherTesseractTet10:
    """The mesher tesseract's TET10 mode (element = 2).

    The promotion appended in front of the TET4 path is the exact linear
    map ``m = (a + b) / 2``, so the interpolation VJP extends by splitting
    each midside cotangent half-and-half onto its corner parents before
    the corner-level IFT pullback — checked mechanically against pure-JAX
    autodiff of the same corner -> TET10 composition.
    """

    @pytest.fixture(scope="class")
    def tesseract(self):
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        from pathlib import Path

        from tesseract_core import Tesseract

        api = Path(__file__).parents[2] / "cadjoint" / "fem" / "tesseracts" / "mesher"
        return Tesseract.from_tesseract_api(str(api / "tesseract_api.py"))

    @pytest.fixture(scope="class")
    def sphere_setup(self, tesseract):
        grid = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), (12, 12, 12))
        lattice = grid.lattice_points()

        def field_of(radius):
            return jnp.linalg.norm(jnp.asarray(lattice), axis=-1) - radius

        def static(element):
            return {
                "origin": np.asarray(grid.origin),
                "spacing": np.asarray(grid.spacing),
                "element": np.int32(element),
                "sharp": np.int32(0),
                "min_ratio": np.float64(1.5),
                "min_dihedral": np.float64(10.0),
            }

        def discover(element, width):
            return tesseract.apply(
                dict(
                    field_values=np.asarray(field_of(1.0)),
                    point_ids=np.zeros(0, np.int32),
                    cell_template=np.zeros((0, width), np.int32),
                    num_surface=np.int32(0),
                    **static(element),
                )
            )

        return grid, field_of, static, discover(0, 4), discover(2, 10)

    def test_apply_matches_the_local_promotion(self, sphere_setup):
        _, _, _, tet4, tet10 = sphere_setup
        points4 = np.asarray(tet4["points"])
        cells4 = np.asarray(tet4["cells"])
        expected_points, expected_cells, parents = tet10_from_tet4(points4, cells4)
        assert np.array_equal(np.asarray(tet10["cells"]), expected_cells)
        assert np.abs(np.asarray(tet10["points"]) - expected_points).max() < 1e-12
        mask4 = np.asarray(tet4["surface_mask"]).astype(bool)
        mask10 = np.asarray(tet10["surface_mask"]).astype(bool)
        count4 = len(points4)
        num_surface = int(mask4.sum())
        assert np.array_equal(mask10[:count4], mask4)
        # Midside nodes are marked exactly when both parents are on the surface.
        assert np.array_equal(mask10[count4:], (parents < num_surface).all(axis=1))

    def test_vjp_matches_autodiff_of_the_promoted_map(self, tesseract, sphere_setup):
        """Mechanical check: tesseract VJP == jax.vjp of corner IFT + promotion."""
        import importlib.util
        from pathlib import Path

        grid, field_of, static, tet4, tet10 = sphere_setup
        api = (
            Path(__file__).parents[2]
            / "cadjoint"
            / "fem"
            / "tesseracts"
            / "mesher"
            / "tesseract_api.py"
        )
        spec = importlib.util.spec_from_file_location("mesher_tesseract_api_t10", api)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        points4 = np.asarray(tet4["points"])
        _, _, parents = tet10_from_tet4(points4, np.asarray(tet4["cells"]))
        num_surface = int(np.asarray(tet4["surface_mask"]).sum())
        points10 = np.asarray(tet10["points"])
        cells10 = np.asarray(tet10["cells"])
        field0 = np.asarray(field_of(1.0))
        rng = np.random.default_rng(7)
        cotangent = rng.standard_normal(points10.shape)  # every node, incl. interior
        result = tesseract.vector_jacobian_product(
            dict(
                field_values=field0,
                point_ids=np.arange(len(points10), dtype=np.int32),
                cell_template=np.zeros((len(cells10), 10), np.int32),
                num_surface=np.int32(np.asarray(tet10["surface_mask"]).sum()),
                **static(2),
            ),
            vjp_inputs={"field_values"},
            vjp_outputs={"points"},
            cotangent_vector={"points": cotangent},
        )
        field_bar = np.asarray(result["field_values"])

        boundary = jnp.asarray(points4[:num_surface])
        interpolant = module.make_interpolant(
            jnp.asarray(field0), np.asarray(grid.origin), np.asarray(grid.spacing)
        )
        gradients = jax.vmap(jax.grad(lambda p: interpolant(p).reshape(())))(boundary)
        squared = jnp.sum(gradients * gradients, axis=1)

        def reference(field_values):
            values = module.make_interpolant(
                field_values, np.asarray(grid.origin), np.asarray(grid.spacing)
            )(boundary)
            moved = boundary - (values / squared)[:, None] * gradients
            corners = jnp.asarray(points4).at[:num_surface].set(moved)
            midsides = corners[jnp.asarray(parents)].mean(axis=1)
            return jnp.concatenate([corners, midsides], axis=0)

        _, vjp_fn = jax.vjp(reference, jnp.asarray(field0))
        (expected,) = vjp_fn(jnp.asarray(cotangent))
        scale = max(float(np.abs(np.asarray(expected)).max()), 1e-30)
        assert np.abs(field_bar - np.asarray(expected)).max() / scale < 1e-10

    def test_traced_gradient_through_the_tet10_mode(self, tesseract, sphere_setup):
        from tesseract_jax import apply_tesseract

        _, field_of, static, tet4, tet10 = sphere_setup
        num_points = len(np.asarray(tet10["points"]))
        num_cells = len(np.asarray(tet10["cells"]))
        num_surface = int(np.asarray(tet4["surface_mask"]).sum())  # corner block

        def mean_radius(radius):
            outputs = apply_tesseract(
                tesseract,
                dict(
                    field_values=field_of(radius),
                    point_ids=np.arange(num_points, dtype=np.int32),
                    cell_template=np.zeros((num_cells, 10), np.int32),
                    num_surface=np.int32(num_surface),
                    **static(2),
                ),
            )
            boundary = outputs["points"][:num_surface]
            return jnp.mean(jnp.sqrt(jnp.sum(boundary * boundary, axis=1)))

        gradient = float(jax.grad(mean_radius)(jnp.asarray(1.0)))
        assert gradient == pytest.approx(1.0, rel=5e-2)


class TestTwoTesseractChain:
    """Flagship demo: mesher tesseract composed with the unmodified elastic tesseract.

    CAD parameters -> SDF lattice samples -> mesher tesseract (HEX8 mode, frozen
    topology, surface-interpolation VJP) -> ``elastic_jaxfem`` tesseract ->
    compliance + smoothed-mass objective -> one ``jax.grad``.  Run with ``-s`` to
    see the measured numbers (``research/tet-vs-hex.md`` records a full run).
    """

    @pytest.fixture(scope="class")
    def chain(self):
        pytest.importorskip("jax_fem")
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        pytest.importorskip("optax")  # imported by the example at module level
        import importlib.util
        from pathlib import Path

        from tesseract_core import Tesseract

        root = Path(__file__).parents[2]
        spec = importlib.util.spec_from_file_location(
            "fem_bracket_optimization", root / "examples" / "fem_bracket_optimization.py"
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        tesseracts = root / "cadjoint" / "fem" / "tesseracts"
        mesher = Tesseract.from_tesseract_api(str(tesseracts / "mesher" / "tesseract_api.py"))
        elastic = Tesseract.from_tesseract_api(
            str(tesseracts / "elastic_jaxfem" / "tesseract_api.py")
        )
        grid = example.build_grid((30, 21, 16))
        lattice = grid.lattice_points()
        static = {
            "origin": np.asarray(grid.origin),
            "spacing": np.asarray(grid.spacing),
            "element": np.int32(1),  # HEX8: the packaged elastic tesseract's schema
            "sharp": np.int32(0),
            "min_ratio": np.float64(1.5),
            "min_dihedral": np.float64(10.0),
        }

        def samples_of(theta):
            return example.theta_sdf(theta)(jnp.asarray(lattice))

        def discover(theta):
            from cadjoint.fem.hexmesh import (
                FaceGroup,
                HexMesh,
                _boundary_face_rows,
                _face_geometry,
                faces_from_nodes,
            )

            found = mesher.apply(
                dict(
                    field_values=np.asarray(samples_of(theta)),
                    point_ids=np.zeros(0, np.int32),
                    cell_template=np.zeros((0, 8), np.int32),
                    num_surface=np.int32(0),
                    **static,
                )
            )
            points = np.asarray(found["points"])
            cells = np.asarray(found["cells"]).astype(np.int32)
            mask = np.asarray(found["surface_mask"]).astype(bool)
            boundary = _boundary_face_rows(cells)
            centers, normals = _face_geometry(points, boundary)
            hex_mesh = HexMesh(
                points=points,
                cells=cells,
                boundary_faces={"all": FaceGroup(boundary, centers, normals)},
                base_points=points,
                snap_mask=mask,
                max_step=0.5 * float(np.linalg.norm(grid.spacing)),
                grid=grid,
            )
            clamp = example.BOLT_CLAMP.resolve(hex_mesh)
            quads = faces_from_nodes(hex_mesh, example.WEB_TIP_LOAD.resolve(hex_mesh)).nodes
            return {
                "N": len(points),
                "T": len(cells),
                "S": int(mask.sum()),
                "cells": cells,
                "clamp": clamp.astype(np.int32),
                "span": np.unique(quads).astype(np.int32),
            }

        cell_volume = float(np.prod(grid.spacing))
        sharpness = 0.5 * float(min(grid.spacing))

        def make_objective(frozen):
            from tesseract_jax import apply_tesseract

            templates = {
                "point_ids": np.arange(frozen["N"], dtype=np.int32),
                "cell_template": np.zeros((frozen["T"], 8), np.int32),
                "num_surface": np.int32(frozen["S"]),
                **static,
            }

            def objective(theta):
                samples = samples_of(theta)
                meshed = apply_tesseract(mesher, dict(field_values=samples, **templates))
                solved = apply_tesseract(
                    elastic,
                    {
                        "points": meshed["points"],
                        "cells": frozen["cells"],
                        "fixed_nodes": frozen["clamp"],
                        "traction_nodes": frozen["span"],
                        "traction_offsets": np.array([0, len(frozen["span"])], np.int32),
                        "traction_vectors": np.asarray([[0.0, -2.0, 0.0]]),
                        "traction_faces": np.zeros((0, 3), np.int32),
                        "traction_face_offsets": np.zeros(0, np.int32),
                        "youngs": np.float64(1000.0),
                        "poisson": np.float64(0.3),
                    },
                )
                compliance = jnp.sum(solved["displacement"] ** 2)
                mass = cell_volume * jnp.sum(jax.nn.sigmoid(-samples / sharpness))
                return compliance + mass, (compliance, mass)

            return objective

        theta0 = jnp.asarray(example.NOMINAL)
        frozen = discover(theta0)
        return example, discover, make_objective, frozen, theta0

    def test_gradient_flows_through_both_tesseracts(self, chain):
        _, _, make_objective, frozen, theta0 = chain
        objective = make_objective(frozen)
        (value, (compliance, mass)), gradient = jax.value_and_grad(objective, has_aux=True)(theta0)
        gradient = np.asarray(gradient)
        print(
            f"\nchain J={float(value):.6f} (C={float(compliance):.6f} M={float(mass):.6f}) "
            f"grad={gradient.tolist()}"
        )
        assert np.isfinite(gradient).all()
        assert (np.abs(gradient) > 1e-6).all()  # every parameter is live
        assert gradient[2] < 0.0  # thicker plate -> stiffer under the prying load

        # Central FD on plate_thickness (the smooth, crease-light parameter);
        # shrink eps if the re-run mesher crosses a topology change.
        for eps in (1e-3, 3e-4, 1e-4):
            offset = np.zeros(3)
            offset[2] = eps
            try:
                plus = float(objective(jnp.asarray(np.asarray(theta0) + offset))[0])
                minus = float(objective(jnp.asarray(np.asarray(theta0) - offset))[0])
            except Exception:
                continue
            finite_difference = (plus - minus) / (2.0 * eps)
            print(f"plate: adjoint {gradient[2]:+.4f} vs FD {finite_difference:+.4f} (eps {eps})")
            assert gradient[2] == pytest.approx(finite_difference, rel=5e-2)
            break
        else:
            pytest.skip("no topology-stable FD window found")

    def test_short_descent_decreases_the_objective(self, chain):
        example, discover, make_objective, frozen, theta0 = chain
        learning_rate = np.array([1e-3, 2e-3, 2e-5])  # per-parameter step scaling
        lower = np.asarray(example.LOWER_BOUNDS)
        upper = np.asarray(example.UPPER_BOUNDS)
        theta = np.asarray(theta0, dtype=np.float64)
        objective = make_objective(frozen)
        values = []
        for _ in range(3):
            try:
                (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(
                    jnp.asarray(theta)
                )
            except Exception:
                objective = make_objective(discover(jnp.asarray(theta)))  # refreeze topology
                (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(
                    jnp.asarray(theta)
                )
            values.append(float(value))
            theta = np.clip(theta - learning_rate * np.asarray(gradient), lower, upper)
        print(f"\ndescent J: {[round(v, 6) for v in values]}")
        assert values[-1] < values[0]


# The bracket meshing box shared by the TET10 chain fixtures: deepened in z
# (dual contouring needs the closed surface — the fillet dips below z = 0).
_BRACKET_TET_BOUNDS = (-1.3, -0.95, -0.16)
_BRACKET_TET_SIZE = (2.6, 1.9, 1.52)
_BRACKET_TET_RESOLUTIONS = ((26, 19, 16), (28, 21, 17), (30, 22, 18), (24, 18, 14))


def _meshable_bracket_lattice(mesher, example):
    """First (grid, sharp) whose TET4 discovery healthily meshes the bracket.

    DC on the trilinear interpolant of the bracket's crease-heavy surface
    is borderline: most lattices self-intersect in TetGen's eyes, and a
    "successful" run can still contain numerically degenerate tets (now
    rejected by ``surface_to_tet_mesh``), which no linear solver survives.
    Scan a few configurations and skip the chain tests when none is
    healthy in this surface realization — the sphere and starter-scene
    validations cover the TET10 tesseract mode itself.
    """
    last_error = "no configuration attempted"
    for resolution in _BRACKET_TET_RESOLUTIONS:
        grid = GridSpec.from_bounds(_BRACKET_TET_BOUNDS, _BRACKET_TET_SIZE, resolution)
        field = np.asarray(
            example.theta_sdf(jnp.asarray(example.NOMINAL))(jnp.asarray(grid.lattice_points()))
        )
        for sharp in (1, 0):
            try:
                mesher.apply(
                    {
                        "field_values": field,
                        "point_ids": np.zeros(0, np.int32),
                        "cell_template": np.zeros((0, 4), np.int32),
                        "num_surface": np.int32(0),
                        "origin": np.asarray(grid.origin),
                        "spacing": np.asarray(grid.spacing),
                        "element": np.int32(0),
                        "sharp": np.int32(sharp),
                        "min_ratio": np.float64(1.5),
                        "min_dihedral": np.float64(10.0),
                    }
                )
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                continue
            return grid, sharp
    pytest.skip(f"no healthy tet lattice for the bracket interpolant here: {last_error[-120:]}")


class TestTet10MesherChain:
    """Mesher tesseract (TET10 mode) composed with the direct TET10 solve.

    CAD parameters -> SDF lattice samples -> mesher tesseract (element = 2,
    frozen topology, midside-splitting interpolation VJP) -> direct
    ``tet_elastic_solve`` (jax-fem adjoint) -> compliance + smoothed-mass
    objective -> one ``jax.grad``.  The in-process second stage is the
    reference for ``TestTet10TwoTesseractChain`` below, which runs the
    same problem through the packaged ``elastic_jaxfem`` solver tesseract
    (its schema is element-agnostic since the TET10 extension).  Uses the
    deepened meshing box (dual contouring needs the closed surface — the
    bracket's fillet dips below z = 0) at the first lattice
    ``_meshable_bracket_lattice`` finds healthy, skipping when none is.
    """

    @pytest.fixture(scope="class")
    def chain(self):
        pytest.importorskip("jax_fem")
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        pytest.importorskip("optax")  # imported by the example at module level
        import importlib.util
        from pathlib import Path

        from tesseract_core import Tesseract

        from cadjoint.fem.tetmesh import (
            tet10_complete_nodes,
            tet10_face_midsides,
            tet10_mesh,
            tet_boundary_faces,
        )

        root = Path(__file__).parents[2]
        spec = importlib.util.spec_from_file_location(
            "fem_bracket_optimization", root / "examples" / "fem_bracket_optimization.py"
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        tesseracts = root / "cadjoint" / "fem" / "tesseracts"
        mesher = Tesseract.from_tesseract_api(str(tesseracts / "mesher" / "tesseract_api.py"))
        grid, sharp = _meshable_bracket_lattice(mesher, example)
        lattice = grid.lattice_points()

        def samples_of(theta):
            return example.theta_sdf(theta)(jnp.asarray(lattice))

        def static(element):
            return {
                "origin": np.asarray(grid.origin),
                "spacing": np.asarray(grid.spacing),
                "element": np.int32(element),
                "sharp": np.int32(sharp),  # the mode _meshable_bracket_lattice picked
                "min_ratio": np.float64(1.5),
                "min_dihedral": np.float64(10.0),
            }

        def discover(theta):
            found = mesher.apply(
                dict(
                    field_values=np.asarray(samples_of(theta)),
                    point_ids=np.zeros(0, np.int32),
                    cell_template=np.zeros((0, 4), np.int32),
                    num_surface=np.int32(0),
                    **static(0),
                )
            )
            points4 = np.asarray(found["points"])
            cells4 = np.asarray(found["cells"]).astype(np.int32)
            num_surface = int(np.asarray(found["surface_mask"]).sum())
            tet10 = tet10_mesh(
                TetMesh(
                    points=points4,
                    cells=cells4,
                    num_surface=num_surface,
                    boundary_tris=tet_boundary_faces(cells4),
                    base_points=points4,
                    max_step=0.5 * float(np.linalg.norm(grid.spacing)),
                    grid=grid,
                )
            )
            clamp = tet10_complete_nodes(tet10, example.BOLT_CLAMP.resolve(tet10))
            faces = tet_faces_from_nodes(tet10, example.WEB_TIP_LOAD.resolve(tet10))
            span = np.concatenate(
                [np.unique(faces), np.unique(tet10_face_midsides(tet10, faces))]
            ).astype(np.int32)
            return {"mesh": tet10, "clamp": clamp, "faces": faces, "span": span}

        cell_volume = float(np.prod(grid.spacing))
        sharpness = 0.5 * float(min(grid.spacing))

        def make_objective(frozen):
            from tesseract_jax import apply_tesseract

            mesh = frozen["mesh"]
            templates = {
                "point_ids": np.arange(mesh.num_points, dtype=np.int32),
                "cell_template": np.zeros((mesh.num_cells, 10), np.int32),
                "num_surface": np.int32(mesh.num_surface),
                **static(2),
            }
            bcs = ElasticBCs(
                fixed_nodes=[frozen["clamp"]],
                traction_nodes=[frozen["span"]],
                traction_vectors=[np.asarray([0.0, -2.0, 0.0])],
            )

            def objective(theta):
                samples = samples_of(theta)
                meshed = apply_tesseract(mesher, dict(field_values=samples, **templates))
                displacement = tet_elastic_solve(
                    meshed["points"],
                    mesh.cells,
                    bcs,
                    youngs=1000.0,
                    poisson=0.3,
                    ele_type="TET10",
                    base_points=mesh.points,
                    traction_faces=[frozen["faces"]],
                )
                compliance = jnp.sum(displacement**2)
                mass = cell_volume * jnp.sum(jax.nn.sigmoid(-samples / sharpness))
                return compliance + mass, (compliance, mass)

            return objective

        theta0 = jnp.asarray(example.NOMINAL)
        frozen = discover(theta0)
        return example, discover, make_objective, frozen, theta0

    def test_gradient_flows_through_mesher_vjp_and_solver_adjoint(self, chain):
        _, _, make_objective, frozen, theta0 = chain
        objective = make_objective(frozen)
        (value, (compliance, mass)), gradient = jax.value_and_grad(objective, has_aux=True)(theta0)
        gradient = np.asarray(gradient)
        print(
            f"\ntet10 chain J={float(value):.6f} (C={float(compliance):.6f} "
            f"M={float(mass):.6f}) grad={gradient.tolist()}"
        )
        assert np.isfinite(gradient).all()
        assert (np.abs(gradient) > 1e-6).all()  # every parameter is live
        assert gradient[2] < 0.0  # thicker plate -> stiffer under the prying load

        # Central FD on plate_thickness (the smooth, crease-light parameter);
        # shrink eps if the re-run mesher crosses a topology change.
        for eps in (1e-3, 3e-4, 1e-4):
            offset = np.zeros(3)
            offset[2] = eps
            try:
                plus = float(objective(jnp.asarray(np.asarray(theta0) + offset))[0])
                minus = float(objective(jnp.asarray(np.asarray(theta0) - offset))[0])
            except Exception:
                continue
            finite_difference = (plus - minus) / (2.0 * eps)
            print(f"plate: adjoint {gradient[2]:+.4f} vs FD {finite_difference:+.4f} (eps {eps})")
            assert gradient[2] == pytest.approx(finite_difference, rel=5e-2)
            break
        else:
            pytest.skip("no topology-stable FD window found")

    def test_short_descent_decreases_the_objective(self, chain):
        example, discover, make_objective, frozen, theta0 = chain
        learning_rate = np.array([1e-3, 2e-3, 2e-5])  # per-parameter step scaling
        lower = np.asarray(example.LOWER_BOUNDS)
        upper = np.asarray(example.UPPER_BOUNDS)
        theta = np.asarray(theta0, dtype=np.float64)
        objective = make_objective(frozen)
        values = []
        for _ in range(2):
            try:
                (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(
                    jnp.asarray(theta)
                )
            except Exception:
                try:
                    objective = make_objective(discover(jnp.asarray(theta)))  # refreeze
                except RuntimeError:
                    break  # interpolant DC fragility: no mesh at this design
                (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(
                    jnp.asarray(theta)
                )
            values.append(float(value))
            theta = np.clip(theta - learning_rate * np.asarray(gradient), lower, upper)
        print(f"\ntet10 descent J: {[round(v, 6) for v in values]}")
        assert len(values) >= 2
        assert values[-1] < values[0]


class TestTet10TwoTesseractChain:
    """The full TET10 two-Tesseract chain: mesher + packaged elastic solver.

    CAD parameters -> SDF lattice samples -> mesher tesseract (element = 2,
    frozen topology, midside-splitting interpolation VJP) -> the packaged
    ``elastic_jaxfem`` tesseract (element-agnostic ``cells``; TET10 with
    exact ``traction_faces`` targeting) -> compliance + smoothed-mass
    objective -> one ``jax.grad``.  Same problem as ``TestTet10MesherChain``
    with the second stage crossing the tesseract boundary too — forward
    parity against the direct in-process solve is asserted at 1e-9.
    """

    @pytest.fixture(scope="class")
    def chain(self):
        pytest.importorskip("jax_fem")
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        pytest.importorskip("optax")  # imported by the example at module level
        import importlib.util
        from pathlib import Path

        from tesseract_core import Tesseract

        from cadjoint.fem.tetmesh import (
            tet10_complete_nodes,
            tet10_face_midsides,
            tet10_mesh,
            tet_boundary_faces,
        )

        root = Path(__file__).parents[2]
        spec = importlib.util.spec_from_file_location(
            "fem_bracket_optimization", root / "examples" / "fem_bracket_optimization.py"
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        tesseracts = root / "cadjoint" / "fem" / "tesseracts"
        mesher = Tesseract.from_tesseract_api(str(tesseracts / "mesher" / "tesseract_api.py"))
        elastic = Tesseract.from_tesseract_api(
            str(tesseracts / "elastic_jaxfem" / "tesseract_api.py")
        )
        grid, sharp = _meshable_bracket_lattice(mesher, example)
        lattice = grid.lattice_points()

        def samples_of(theta):
            return example.theta_sdf(theta)(jnp.asarray(lattice))

        def static(element):
            return {
                "origin": np.asarray(grid.origin),
                "spacing": np.asarray(grid.spacing),
                "element": np.int32(element),
                "sharp": np.int32(sharp),  # the mode _meshable_bracket_lattice picked
                "min_ratio": np.float64(1.5),
                "min_dihedral": np.float64(10.0),
            }

        def discover(theta):
            found = mesher.apply(
                dict(
                    field_values=np.asarray(samples_of(theta)),
                    point_ids=np.zeros(0, np.int32),
                    cell_template=np.zeros((0, 4), np.int32),
                    num_surface=np.int32(0),
                    **static(0),
                )
            )
            points4 = np.asarray(found["points"])
            cells4 = np.asarray(found["cells"]).astype(np.int32)
            num_surface = int(np.asarray(found["surface_mask"]).sum())
            tet10 = tet10_mesh(
                TetMesh(
                    points=points4,
                    cells=cells4,
                    num_surface=num_surface,
                    boundary_tris=tet_boundary_faces(cells4),
                    base_points=points4,
                    max_step=0.5 * float(np.linalg.norm(grid.spacing)),
                    grid=grid,
                )
            )
            clamp = tet10_complete_nodes(tet10, example.BOLT_CLAMP.resolve(tet10))
            faces = tet_faces_from_nodes(tet10, example.WEB_TIP_LOAD.resolve(tet10))
            span = np.concatenate(
                [np.unique(faces), np.unique(tet10_face_midsides(tet10, faces))]
            ).astype(np.int32)
            return {"mesh": tet10, "clamp": clamp, "faces": faces, "span": span}

        cell_volume = float(np.prod(grid.spacing))
        sharpness = 0.5 * float(min(grid.spacing))

        def solver_inputs(frozen):
            faces = np.asarray(frozen["faces"], dtype=np.int32)
            return {
                "cells": np.asarray(frozen["mesh"].cells, dtype=np.int32),
                "fixed_nodes": frozen["clamp"].astype(np.int32),
                "traction_nodes": frozen["span"].astype(np.int32),
                "traction_offsets": np.array([0, len(frozen["span"])], np.int32),
                "traction_vectors": np.asarray([[0.0, -2.0, 0.0]]),
                "traction_faces": faces,
                "traction_face_offsets": np.array([0, len(faces)], np.int32),
                "youngs": np.float64(1000.0),
                "poisson": np.float64(0.3),
            }

        def make_objective(frozen):
            from tesseract_jax import apply_tesseract

            mesh = frozen["mesh"]
            templates = {
                "point_ids": np.arange(mesh.num_points, dtype=np.int32),
                "cell_template": np.zeros((mesh.num_cells, 10), np.int32),
                "num_surface": np.int32(mesh.num_surface),
                **static(2),
            }
            inputs = solver_inputs(frozen)

            def objective(theta):
                samples = samples_of(theta)
                meshed = apply_tesseract(mesher, dict(field_values=samples, **templates))
                solved = apply_tesseract(elastic, dict(points=meshed["points"], **inputs))
                compliance = jnp.sum(solved["displacement"] ** 2)
                mass = cell_volume * jnp.sum(jax.nn.sigmoid(-samples / sharpness))
                return compliance + mass, (compliance, mass)

            return objective

        theta0 = jnp.asarray(example.NOMINAL)
        frozen = discover(theta0)
        return example, discover, make_objective, solver_inputs, elastic, frozen, theta0

    def test_forward_parity_against_the_direct_solve(self, chain):
        """The packaged stage 2 equals ``tet_elastic_solve`` at 1e-9."""
        _, _, _, solver_inputs, elastic, frozen, _ = chain
        mesh = frozen["mesh"]
        bcs = ElasticBCs(
            fixed_nodes=[frozen["clamp"]],
            traction_nodes=[frozen["span"]],
            traction_vectors=[np.asarray([0.0, -2.0, 0.0])],
        )
        direct = np.asarray(
            tet_elastic_solve(
                mesh.points,
                mesh.cells,
                bcs,
                youngs=1000.0,
                poisson=0.3,
                ele_type="TET10",
                traction_faces=[frozen["faces"]],
            )
        )
        solved = elastic.apply(dict(points=np.asarray(mesh.points), **solver_inputs(frozen)))
        packaged = np.asarray(solved["displacement"])
        assert np.abs(packaged - direct).max() < 1e-9
        assert np.abs(direct).max() > 1e-4

    def test_gradient_flows_through_both_tesseracts(self, chain):
        _, _, make_objective, _, _, frozen, theta0 = chain
        objective = make_objective(frozen)
        (value, (compliance, mass)), gradient = jax.value_and_grad(objective, has_aux=True)(theta0)
        gradient = np.asarray(gradient)
        print(
            f"\ntet10 two-tesseract J={float(value):.6f} (C={float(compliance):.6f} "
            f"M={float(mass):.6f}) grad={gradient.tolist()}"
        )
        assert np.isfinite(gradient).all()
        assert (np.abs(gradient) > 1e-6).all()  # every parameter is live
        assert gradient[2] < 0.0  # thicker plate -> stiffer under the prying load

        # Central FD on plate_thickness (the smooth, crease-light parameter);
        # shrink eps if the re-run mesher crosses a topology change.
        for eps in (1e-3, 3e-4, 1e-4):
            offset = np.zeros(3)
            offset[2] = eps
            try:
                plus = float(objective(jnp.asarray(np.asarray(theta0) + offset))[0])
                minus = float(objective(jnp.asarray(np.asarray(theta0) - offset))[0])
            except Exception:
                continue
            finite_difference = (plus - minus) / (2.0 * eps)
            print(f"plate: adjoint {gradient[2]:+.4f} vs FD {finite_difference:+.4f} (eps {eps})")
            assert gradient[2] == pytest.approx(finite_difference, rel=5e-2)
            break
        else:
            pytest.skip("no topology-stable FD window found")

    def test_short_descent_decreases_the_objective(self, chain):
        example, discover, make_objective, _, _, frozen, theta0 = chain
        learning_rate = np.array([1e-3, 2e-3, 2e-5])  # per-parameter step scaling
        lower = np.asarray(example.LOWER_BOUNDS)
        upper = np.asarray(example.UPPER_BOUNDS)
        theta = np.asarray(theta0, dtype=np.float64)
        objective = make_objective(frozen)
        values = []
        for _ in range(4):
            try:
                (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(
                    jnp.asarray(theta)
                )
            except Exception:
                try:
                    objective = make_objective(discover(jnp.asarray(theta)))  # refreeze
                except RuntimeError:
                    break  # interpolant DC fragility: no mesh at this design
                (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(
                    jnp.asarray(theta)
                )
            values.append(float(value))
            theta = np.clip(theta - learning_rate * np.asarray(gradient), lower, upper)
        print(f"\ntet10 two-tesseract descent J: {[round(v, 6) for v in values]}")
        assert len(values) >= 2
        assert values[-1] < values[0]
