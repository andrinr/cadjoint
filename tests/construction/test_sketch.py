"""End-to-end tests for the sketch construction tree.

Covers the full prototype workflow: 2D profile → constraints → solve →
extrude/revolve into SDF solids → parameter extraction → gradient-based
optimization on the constraint manifold.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxcad import apply_parameters, extract_parameters, functionalize
from jaxcad.constraints import DistanceConstraint, FixedConstraint
from jaxcad.constraints.solve import project_to_manifold, solve_constraints
from jaxcad.construction import PolygonProfile, SketchPlane, extrude, revolve
from jaxcad.geometry import Vector2, as_parameter
from jaxcad.sdf import Sphere, volume
from jaxcad.sdf.primitives import Box, polygon_sdf_2d

SQUARE = [[-1.0, -0.75], [1.0, -0.75], [1.0, 0.75], [-1.0, 0.75]]


# ── Vector2 parameter ─────────────────────────────────────────────────────────


class TestVector2:
    def test_shape_enforced(self):
        with pytest.raises(ValueError):
            Vector2(value=jnp.array([1.0, 2.0, 3.0]))

    def test_as_parameter_2d(self):
        p = as_parameter(jnp.array([1.0, 2.0]))
        assert isinstance(p, Vector2)
        assert not p.free

    def test_pytree_roundtrip(self):
        v = Vector2(value=jnp.array([1.0, 2.0]), free=True, name="v")
        leaves, treedef = jax.tree_util.tree_flatten(v)
        v2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert v2.name == "v" and jnp.allclose(v2.value, v.value)


# ── FixedConstraint ───────────────────────────────────────────────────────────


class TestFixedConstraint:
    def test_residual_and_dof(self):
        v = Vector2([0.5, -0.5], free=True, name="v")
        c = FixedConstraint(v, [0.0, 0.0])
        assert c.dof_reduction() == 2
        r = c.compute_residual({"v": jnp.array([0.5, -0.5])})
        assert jnp.allclose(r, jnp.array([0.5, -0.5]))

    def test_shape_mismatch_raises(self):
        v = Vector2([0.0, 0.0], free=True, name="v")
        with pytest.raises(ValueError):
            FixedConstraint(v, [0.0, 0.0, 0.0])

    def test_registered_on_parameter(self):
        v = Vector2([0.0, 0.0], free=True, name="v")
        c = FixedConstraint(v, [0.0, 0.0])
        assert any(x is c for x in v.get_constraints())


# ── 2D polygon SDF ────────────────────────────────────────────────────────────


class TestPolygonSDF:
    def test_square_signs_and_distances(self):
        verts = jnp.array(SQUARE)
        assert float(polygon_sdf_2d(jnp.array([0.0, 0.0]), verts)) == pytest.approx(-0.75, abs=1e-6)
        assert float(polygon_sdf_2d(jnp.array([2.0, 0.0]), verts)) == pytest.approx(1.0, abs=1e-6)
        # corner distance
        d = float(polygon_sdf_2d(jnp.array([2.0, 1.75]), verts))
        assert d == pytest.approx(jnp.sqrt(2.0), abs=1e-5)

    def test_winding_invariance(self):
        verts = jnp.array(SQUARE)
        p = jnp.array([0.3, -0.2])
        d_ccw = polygon_sdf_2d(p, verts)
        d_cw = polygon_sdf_2d(p, verts[::-1])
        assert jnp.allclose(d_ccw, d_cw, atol=1e-6)

    def test_concave_polygon(self):
        # L-shape: unit notch cut from a 2×2 square
        el = jnp.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
        assert float(polygon_sdf_2d(jnp.array([0.5, 0.5]), el)) < 0  # in the leg
        assert float(polygon_sdf_2d(jnp.array([1.5, 1.5]), el)) > 0  # in the notch
        assert float(polygon_sdf_2d(jnp.array([1.5, 1.5]), el)) == pytest.approx(0.5, abs=1e-6)

    def test_batched_evaluation(self):
        verts = jnp.array(SQUARE)
        pts = jnp.zeros((7, 2))
        assert polygon_sdf_2d(pts, verts).shape == (7,)


# ── Extrude / Revolve solids ──────────────────────────────────────────────────


class TestGeneratedSolids:
    def test_extruded_square_matches_box(self):
        solid = extrude(PolygonProfile(SQUARE, name="sq"), depth=1.6)
        box = Box(size=jnp.array([1.0, 0.75, 0.8]))
        pts = jax.random.uniform(jax.random.PRNGKey(0), (500, 3), minval=-2.5, maxval=2.5)
        err = jnp.max(jnp.abs(jax.vmap(solid)(pts) - jax.vmap(box)(pts)))
        assert float(err) < 1e-5

    def test_extrude_on_tilted_plane(self):
        plane = SketchPlane(origin=[0.0, 1.0, 0.0], normal=[1.0, 0.0, 0.0])
        solid = extrude(PolygonProfile(SQUARE, plane=plane, name="side"), depth=1.6)
        # plane origin is the solid's center
        assert float(solid(jnp.array([0.0, 1.0, 0.0]))) == pytest.approx(-0.75, abs=1e-5)
        # 2.0 along the plane normal: outside by 2.0 - depth/2
        assert float(solid(jnp.array([2.0, 1.0, 0.0]))) == pytest.approx(1.2, abs=1e-5)

    def test_revolved_washer(self):
        profile = PolygonProfile([[1.0, -0.2], [1.4, -0.2], [1.4, 0.2], [1.0, 0.2]], name="washer")
        ring = revolve(profile)
        assert float(ring(jnp.array([1.2, 0.0, 0.0]))) == pytest.approx(-0.2, abs=1e-5)
        assert float(ring(jnp.array([0.0, 0.0, 0.0]))) == pytest.approx(1.0, abs=1e-5)
        # rotational symmetry
        d1 = ring(jnp.array([1.2, 0.05, 0.0]))
        d2 = ring(jnp.array([0.0, 0.05, 1.2]))
        assert jnp.allclose(d1, d2, atol=1e-6)

    def test_boolean_composition(self):
        solid = extrude(PolygonProfile(SQUARE, name="comp"), depth=1.6)
        combined = solid | Sphere(0.5)
        assert float(combined(jnp.array([0.0, 0.0, 0.0]))) < 0


# ── Shared parameters across the two trees ────────────────────────────────────


class TestParameterSharing:
    def test_extraction_finds_sketch_vertices(self):
        profile = PolygonProfile(SQUARE, name="p")
        solid = extrude(profile, depth=1.6)
        free, _, meta = extract_parameters(solid)
        assert {"p_v0", "p_v1", "p_v2", "p_v3"} <= set(free)
        # identical objects, not copies — the construction tree edits the solid
        for i in range(4):
            assert meta[f"p_v{i}"] is profile.vertices[i]

    def test_functionalize_matches_direct_eval(self):
        profile = PolygonProfile(SQUARE, name="f")
        solid = extrude(profile, depth=1.6)
        free, fixed, _ = extract_parameters(solid)
        fn = functionalize(solid)(free, fixed)
        p = jnp.array([1.7, 0.2, 0.1])
        assert jnp.allclose(fn(p), solid(p), atol=1e-6)

    def test_vertex_gradient_flows_through_solid(self):
        profile = PolygonProfile(SQUARE, name="g")
        solid = extrude(profile, depth=1.6)
        free, fixed, _ = extract_parameters(solid)
        fn = functionalize(solid)

        def loss(params):
            return fn(params, fixed)(jnp.array([2.0, 0.0, 0.0]))

        grads = jax.grad(loss)(free)
        # moving the right edge (v1, v2) outward reduces the distance
        assert float(grads["g_v1"][0]) < -0.1
        assert float(grads["g_v2"][0]) < -0.1


# ── Constraint solving on the sketch ──────────────────────────────────────────


class TestConstraintSolve:
    def test_anchored_triangle_solve(self):
        # 3 free 2D vertices (6 DOF): anchor two, apex fixed by two distances
        profile = PolygonProfile([[0.1, -0.1], [1.9, 0.2], [1.0, 1.5]], name="tri")
        v0, v1, v2 = profile.vertices
        FixedConstraint(v0, [0.0, 0.0])
        FixedConstraint(v1, [2.0, 0.0])
        DistanceConstraint(v0, v2, 2.0)
        DistanceConstraint(v1, v2, 2.0)

        solved = solve_constraints(profile)
        assert jnp.allclose(solved["tri_v0"], jnp.array([0.0, 0.0]), atol=1e-4)
        assert jnp.allclose(solved["tri_v1"], jnp.array([2.0, 0.0]), atol=1e-4)
        assert jnp.allclose(solved["tri_v2"], jnp.array([1.0, jnp.sqrt(3.0)]), atol=1e-4)

    def test_solved_params_drive_the_solid(self):
        profile = PolygonProfile([[0.1, -0.1], [1.9, 0.2], [1.0, 1.5]], name="tri2")
        v0, v1, v2 = profile.vertices
        FixedConstraint(v0, [0.0, 0.0])
        FixedConstraint(v1, [2.0, 0.0])
        DistanceConstraint(v0, v2, 2.0)
        DistanceConstraint(v1, v2, 2.0)

        solid = extrude(profile, depth=1.0)
        free, fixed, _ = extract_parameters(solid)
        solved = solve_constraints(profile)
        fn = functionalize(solid)(solved, fixed)
        # centroid of the solved equilateral triangle is inside the solid
        centroid = jnp.array([1.0, float(jnp.sqrt(3.0)) / 3.0, 0.0])
        assert float(fn(centroid)) < 0


# ── Constrained gradient optimization end-to-end ──────────────────────────────


class TestConstrainedOptimization:
    def test_volume_target_on_constraint_manifold(self):
        # Quad sketch: anchor v0, pin edge v0-v1 length; optimize remaining DOF
        # so the extruded volume hits a target, projecting back to the manifold
        # after every gradient step.
        profile = PolygonProfile([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], name="opt")
        v0, v1, _, _ = profile.vertices
        FixedConstraint(v0, [0.0, 0.0])
        DistanceConstraint(v0, v1, 2.0)

        solid = extrude(profile, depth=1.5)
        free, fixed, meta = extract_parameters(solid)
        fn = functionalize(solid)
        target = 6.0

        def loss(params):
            vol = volume(fn(params, fixed), bounds=(-1, -1, -1), size=(4, 4, 2), resolution=32)
            return (vol - target) ** 2

        params = dict(free)
        initial_loss = float(loss(params))
        for _ in range(40):
            grads = jax.grad(loss)(params)
            params = {k: v - 0.05 * grads[k] for k, v in params.items()}
            params = project_to_manifold(params, meta, steps=2)

        final_loss = float(loss(params))
        assert final_loss < initial_loss * 0.05

        # constraints still hold after optimization
        assert jnp.allclose(params["opt_v0"], jnp.array([0.0, 0.0]), atol=1e-3)
        edge = float(jnp.linalg.norm(params["opt_v1"] - params["opt_v0"]))
        assert edge == pytest.approx(2.0, abs=1e-3)


# ── apply_parameters: writing values back into the shared tree ────────────────


class TestApplyParameters:
    def test_apply_updates_construction_and_solid(self):
        profile = PolygonProfile([[0.1, -0.1], [1.9, 0.2], [1.0, 1.5]], name="ap")
        v0, v1, v2 = profile.vertices
        FixedConstraint(v0, [0.0, 0.0])
        FixedConstraint(v1, [2.0, 0.0])
        DistanceConstraint(v0, v2, 2.0)
        DistanceConstraint(v1, v2, 2.0)
        solid = extrude(profile, depth=1.0)

        solved = solve_constraints(profile)
        apply_parameters(profile, solved)

        # construction tree updated
        assert jnp.allclose(profile.vertices[2].value, jnp.array([1.0, jnp.sqrt(3.0)]), atol=1e-4)
        # the solid shares the parameters, so direct evaluation sees the solve
        centroid = jnp.array([1.0, float(jnp.sqrt(3.0)) / 3.0, 0.0])
        assert float(solid(centroid)) < 0
        # applying via the solid works identically (same walk, shared params)
        apply_parameters(solid, solved)
        assert jnp.allclose(profile.vertices[2].value, jnp.array([1.0, jnp.sqrt(3.0)]), atol=1e-4)

    def test_unknown_name_raises(self):
        profile = PolygonProfile(SQUARE, name="apu")
        with pytest.raises(ValueError, match="No free parameter"):
            apply_parameters(profile, {"nope": jnp.zeros(2)})

    def test_shape_mismatch_raises(self):
        profile = PolygonProfile(SQUARE, name="aps")
        with pytest.raises(ValueError, match="shape"):
            apply_parameters(profile, {"aps_v0": jnp.zeros(3)})
