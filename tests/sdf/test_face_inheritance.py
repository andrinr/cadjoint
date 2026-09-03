"""Face references surviving the rest of the feature tree.

A face reference is only useful if it outlives the feature that declared it.
``extrude(...).cap("+")`` was always fine; the moment the body was cut,
patterned or unioned, the reference was gone and the body stopped being
something a later sketch could sit on — which is most of feature-based CAD.

These tests pin which nodes forward a base child's faces and, just as
importantly, which deliberately do not: a node that *moves* the surface must
not hand back a plane that is no longer on it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.construction import Axis, PolygonProfile, Solid, extrude, revolve
from cadjoint.geometry import Scalar
from cadjoint.sdf.boolean import Difference, Intersection, Union, Xor
from cadjoint.sdf.transforms.fields import Mirror, Offset, Shell
from cadjoint.sdf.transforms.patterns import LinearPattern, PolarPattern
from cadjoint.sdf.transforms import Translate

SQUARE = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]


def _body(depth: float = 0.4):
    return extrude(PolygonProfile(SQUARE, name="body"), depth=Scalar(depth))


_TOOLS = iter(range(1000))


def _tool():
    """A construction primitive with a unique parameter name each time."""
    return Solid.cylinder(
        radius=0.1, height=0.5, position=[0.0, 0.0, 0.0], name=f"tool{next(_TOOLS)}"
    )


class TestBooleansForwardTheBaseOperand:
    @pytest.mark.parametrize("op", [Union, Difference, Intersection])
    def test_boolean_keeps_the_first_operand_faces(self, op):
        body = _body()
        combined = op(body, _tool(), smoothness=0.0)
        assert combined.faces.keys() == body.faces.keys()
        np.testing.assert_allclose(
            np.asarray(combined.cap("+").origin), np.asarray(body.cap("+").origin)
        )

    def test_the_cap_plane_is_the_uncut_one(self):
        """The reference is the plane, and the plane is exact after a cut."""
        body = _body(depth=0.4)
        cut = Difference(body, _tool(), smoothness=0.0)
        np.testing.assert_allclose(np.asarray(cut.cap("+").origin), [0.0, 0.0, 0.2], atol=1e-6)
        np.testing.assert_allclose(np.asarray(cut.cap("+").normal), [0.0, 0.0, 1.0], atol=1e-6)

    def test_forwarding_survives_nesting(self):
        body = _body()
        deep = Difference(Union(body, _tool()), _tool(), smoothness=0.0)
        assert deep.cap("-").key == "cap-"

    def test_xor_forwards_too(self):
        """Xor takes no smoothness, so it gets its own case."""
        body = _body()
        assert Xor(body, _tool()).faces.keys() == body.faces.keys()

    def test_a_tool_with_faces_does_not_donate_them(self):
        """Only the BASE operand's faces are the result's; a subtracted tool's are not."""
        plain = Union((_tool(),), smoothness=0.0)
        # The tool is a construction primitive and has its own faces, but a
        # boolean whose base lacks them must not fall through to a later
        # operand -- that would hand back a face of the thing being removed.
        faceless = Difference(_faceless_sphere(), _body(), smoothness=0.0)
        assert plain.faces.keys() == _tool().faces.keys()
        with pytest.raises(AttributeError):
            faceless.faces  # noqa: B018


def _faceless_sphere():
    from cadjoint.sdf.primitives import Sphere

    return Sphere(radius=0.8)


class TestPatternsForwardCopyZero:
    def test_polar_pattern_keeps_the_child_faces(self):
        body = _body()
        assert PolarPattern(body, count=5).faces.keys() == body.faces.keys()

    def test_linear_pattern_keeps_the_child_faces(self):
        body = _body()
        pattern = LinearPattern(body, direction=[1.0, 0.0, 0.0], count=3, spacing=1.5)
        np.testing.assert_allclose(
            np.asarray(pattern.cap("+").origin), np.asarray(body.cap("+").origin)
        )


class TestNodesThatMoveTheSurfaceRefuse:
    """A forwarded face must lie on the forwarding node's own surface."""

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda body: Shell(body, 0.05),
            lambda body: Offset(body, 0.05),
            lambda body: Mirror(body, "x"),
            lambda body: Translate(body, offset=jnp.array([1.0, 0.0, 0.0])),
        ],
    )
    def test_displacing_nodes_declare_no_faces(self, wrap):
        with pytest.raises(AttributeError):
            wrap(_body()).faces  # noqa: B018

    def test_an_unrelated_attribute_still_raises(self):
        cut = Difference(_body(), _tool(), smoothness=0.0)
        with pytest.raises(AttributeError, match="nonsense"):
            cut.nonsense  # noqa: B018

    def test_the_error_names_the_node_that_failed(self):
        with pytest.raises(AttributeError, match="Shell"):
            Shell(_body(), 0.05).faces  # noqa: B018


class TestRevolveAxisForwards:
    def test_a_boolean_keeps_the_axis_of_its_base_revolve(self):
        ring = revolve(PolygonProfile([[0.2, -0.1], [0.5, -0.1], [0.5, 0.1]], name="ring"))
        cut = Difference(ring, _tool(), smoothness=0.0)
        np.testing.assert_allclose(np.asarray(cut.axis.direction), [0.0, 1.0, 0.0], atol=1e-6)


class TestPolarPatternAboutALine:
    def test_named_axis_stays_z_only(self):
        with pytest.raises(ValueError, match="axis"):
            PolarPattern(_body(), count=3, axis="x")

    def test_default_matches_the_axis_aligned_formula_exactly(self):
        """The general Rodrigues path must not perturb the historical default."""
        import math

        shape = Solid.sphere(radius=0.3, position=[0.7, 0.2, -0.3], name="s")
        pattern = PolarPattern(shape, count=4)
        points = np.random.default_rng(0).uniform(-2.0, 2.0, (128, 3)).astype(np.float32)
        expected = np.minimum.reduce(
            [
                np.asarray(
                    shape(
                        jnp.stack(
                            [
                                points[:, 0] * math.cos(t) + points[:, 1] * math.sin(t),
                                points[:, 1] * math.cos(t) - points[:, 0] * math.sin(t),
                                points[:, 2],
                            ],
                            axis=-1,
                        )
                    )
                )
                for t in [2.0 * math.pi * i / 4 for i in range(4)]
            ]
        )
        np.testing.assert_allclose(np.asarray(pattern(points)), expected, atol=1e-6)

    def test_pattern_about_an_offset_horizontal_line(self):
        """A bolt circle needs the axis's POSITION, which a letter cannot carry."""
        axis = Axis(origin=[0.0, 0.0, 0.6], direction=[1.0, 0.0, 0.0])
        stud = Solid.sphere(radius=0.12, position=[0.0, 0.25, 0.6], name="stud")
        pattern = PolarPattern(stud, count=4, axis=axis)
        # Copy 0 sits where the child does; the others are rotated a quarter
        # turn about the line, so they land on the axis's own circle.
        for point in ([0.0, 0.25, 0.6], [0.0, -0.25, 0.6], [0.0, 0.0, 0.85], [0.0, 0.0, 0.35]):
            assert float(pattern(jnp.asarray(point))) < 0.0

    def test_a_revolves_own_axis_drives_the_pattern(self):
        ring = revolve(PolygonProfile([[0.2, -0.1], [0.5, -0.1], [0.5, 0.1]], name="ring"))
        pattern = PolarPattern(_body(), count=6, axis=ring.axis)
        np.testing.assert_allclose(np.asarray(pattern.params["direction"].xyz), [0.0, 1.0, 0.0])


class TestMirrorAcrossAPlane:
    def test_named_axes_are_unchanged(self):
        body = _body()
        points = np.random.default_rng(1).uniform(-2.0, 2.0, (128, 3)).astype(np.float32)
        mirrored = Mirror(body, "x")
        expected = np.asarray(body(points * np.array([-1.0, 1.0, 1.0], dtype=np.float32)))
        np.testing.assert_allclose(np.asarray(mirrored(points)), expected, atol=1e-6)

    def test_a_bad_axis_name_still_raises(self):
        with pytest.raises(ValueError, match="axis"):
            Mirror(_body(), "w")

    def test_mirror_across_a_sketch_plane_off_the_origin(self):
        from cadjoint.construction import SketchPlane

        body = _body()
        seam = SketchPlane(origin=[1.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
        mirrored = Mirror(body, seam)
        # The body straddles x = 0, so its image straddles x = 2.
        assert float(mirrored(jnp.array([2.0, 0.0, 0.0]))) < 0.0
        assert float(mirrored(jnp.array([0.0, 0.0, 0.0]))) > 0.0

    def test_mirror_across_a_face(self):
        body = _body(depth=0.4)
        mirrored = Mirror(body, body.cap("+"))
        # Reflected across z = 0.2, the solid's image spans z in [0.2, 0.6].
        assert float(mirrored(jnp.array([0.0, 0.0, 0.4]))) < 0.0
        assert float(mirrored(jnp.array([0.0, 0.0, 0.7]))) > 0.0


class TestCompilationSurvives:
    def test_the_new_parameters_functionalize_and_mesh(self):
        from cadjoint import extract_parameters, functionalize
        from cadjoint.construction import SketchPlane

        axis = Axis(origin=[0.0, 0.0, 0.0], direction=[0.0, 0.0, 1.0])
        scene = Union(
            Difference(_body(), PolarPattern(_tool(), 5, axis=axis), smoothness=0.0),
            Mirror(_tool(), SketchPlane(origin=[0.9, 0.0, 0.0], normal=[1.0, 0.0, 0.0])),
            smoothness=0.0,
        )
        free, fixed, _ = extract_parameters(scene)
        compiled = functionalize(scene)(free, fixed)
        direct = float(scene(jnp.array([0.3, 0.1, 0.0])))
        np.testing.assert_allclose(float(compiled(jnp.array([0.3, 0.1, 0.0]))), direct, atol=1e-6)
