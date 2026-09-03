"""The motor-shield scene: it builds what it claims, captures what it declares.

``scenes/motor_shield.py`` is the second push-the-limits part
(``research/complex-scene-2.md``). These tests pin the modelling promises the
file's comments make — the feature chain, the helix, the tangent junctions,
the patterns — and that its studies, meshes and optimisation are captured the
way the viewer captures them.

The scene is executed once per session: it runs a constraint solve and builds
a large SDF tree, and every test here reads the same built model.
"""

from __future__ import annotations

import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def shield():
    return importlib.import_module("scenes.motor_shield")


def _inside(sdf, point):
    return float(sdf(jnp.asarray(point, dtype=jnp.float32))) < 0.0


def _field(solid):
    """One compiled, vmapped copy of a solid's field.

    Probing this part point by point costs a full Python trace of a
    fifty-leaf tree each time; a test that wants a hundred samples asks for
    them in one batch instead.
    """
    from cadjoint import extract_parameters, functionalize

    free, fixed, _ = extract_parameters(solid)
    return jax.jit(jax.vmap(functionalize(solid)(free, fixed)))


def _ring(field, radius, height, count, *, phase=0.0):
    """Signed distances around a horizontal circle, as a numpy array."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + phase
    points = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(count, height)], axis=-1
    )
    return np.asarray(field(jnp.asarray(points, dtype=jnp.float32)))


def _tooth_widths(occupied):
    """Angular widths, in degrees, of the runs of `True` around a closed ring."""
    count = len(occupied)
    widths, index = [], 0
    while index < count:
        if occupied[index] and not occupied[index - 1]:
            end = index
            while occupied[end % count]:
                end += 1
            widths.append(360.0 * (end - index) / count)
            index = end
        else:
            index += 1
    return widths


class TestTheFeatureStack:
    @pytest.mark.parametrize(
        ("feature", "low", "high"),
        [
            ("flange", 0.0, 0.20),
            ("tower", 0.20, 1.00),
            ("seal_boss", 1.00, 1.14),
            ("retainer_pad", 1.14, 1.20),
        ],
    )
    def test_the_chain_stacks_without_gaps(self, shield, feature, low, high):
        solid = getattr(shield, feature)
        assert float(solid.cap("-").origin[2]) == pytest.approx(low, abs=1e-5)
        assert float(solid.cap("+").origin[2]) == pytest.approx(high, abs=1e-5)

    def test_the_chain_is_three_sketch_planes_deep(self, shield):
        for profile in (shield.tower_profile, shield.boss_profile, shield.pad_profile):
            assert profile.plane.reference is not None

    def test_the_lug_and_the_flange_share_their_planes(self, shield):
        """The coplanar join is deliberate: lug caps lie in the flange's cap planes."""
        for sign in ("+", "-"):
            assert float(shield.lug.cap(sign).origin[2]) == pytest.approx(
                float(shield.flange.cap(sign).origin[2]), abs=1e-6
            )

    def test_the_riser_flange_top_is_coplanar_with_the_riser_and_tower_caps(self, shield):
        top = float(shield.riser_flange.cap("+").origin[2])
        assert top == pytest.approx(float(shield.riser.cap("+").origin[2]), abs=1e-6)
        assert top == pytest.approx(float(shield.tower.cap("+").origin[2]), abs=1e-6)


class TestTheGeometryIsWhatItClaims:
    def test_the_bore_is_open_and_the_seat_is_cut(self, shield):
        assert not _inside(shield.shield, [0.0, 0.0, 0.5])
        assert not _inside(shield.shield, [0.40, 0.0, 0.20])  # the seat, r = 0.47
        assert _inside(shield.shield, [0.60, 0.0, 0.20])  # metal past the seat

    def test_the_helix_is_a_helix(self, shield):
        """The channel is at angle 180° at both end discs and 0° at mid depth."""
        low, high = shield._channel_low, shield._channel_high
        mid = (low + high) / 2.0
        assert not _inside(shield.shield, [-0.75, 0.0, low])
        assert not _inside(shield.shield, [-0.75, 0.0, high])
        assert not _inside(shield.shield, [0.75, 0.0, mid])
        # A quarter turn up from the bottom the tube is at -90°: on the -y side.
        assert not _inside(shield.shield, [0.0, -0.75, low + (high - low) / 4.0])
        assert _inside(shield.shield, [0.0, 0.75, low + (high - low) / 4.0])
        # And the land between turns is metal.
        assert _inside(shield.shield, [0.75, 0.0, low])

    def test_the_inlet_bore_reaches_the_helix_start_and_the_riser(self, shield):
        assert not _inside(shield.shield, [-0.75, -0.30, shield._channel_low])
        assert not _inside(shield.shield, [-0.75, -shield._riser_y, 0.80])  # riser bore

    def test_the_risers_are_externally_tangent_to_the_tower(self, shield):
        distance = math.hypot(-0.75, -shield._riser_y)
        assert distance == pytest.approx(1.0 + 0.16, abs=1e-6)
        assert _inside(shield.shield, [-0.75, shield._riser_y + 0.115, 0.6])  # the mirrored copy

    def test_the_stiffener_inner_face_is_the_tower_tangent_plane(self, shield):
        assert float(shield.stiffener.cap("-").origin[1]) == pytest.approx(1.0, abs=1e-6)
        assert _inside(shield.shield, [0.3, 1.06, 0.35])
        assert _inside(shield.shield, [-1.06, 0.3, 0.35])  # a polar copy

    def test_the_gusset_ring_is_eight_stations_with_two_suppressed(self, shield):
        """Six ribs on an eight-way ring: 3 and 5 are where the gallery is cast."""
        distances = _ring(_field(shield.ribs), 1.10, 0.30, 8)
        assert [i for i, d in enumerate(distances) if d < 0.0] == [0, 1, 2, 4, 6, 7]
        assert shield.ribs.skip == (3, 5)

    def test_the_kept_ribs_did_not_move_up_into_the_gaps(self, shield):
        """Suppression leaves the ring's other stations exactly where they were."""
        from cadjoint.sdf.transforms.patterns import PolarPattern

        full = _field(PolarPattern(shield.rib, count=8, axis=shield.bore_axis))
        thinned = _field(shield.ribs)
        for index in (0, 1, 2, 4, 6, 7):
            angle = 2.0 * math.pi * index / 8
            point = jnp.asarray([[1.10 * math.cos(angle), 1.10 * math.sin(angle), 0.30]], "float32")
            assert float(thinned(point)[0]) == pytest.approx(float(full(point)[0]), abs=1e-6)

    def test_the_cast_gallery_stands_where_the_two_ribs_were(self, shield):
        """The suppressed stations carry a bored barrel of metal, not a slot in air."""
        field = _field(shield.shield)
        low, high = shield._channel_low, shield._channel_high
        axis = np.asarray(
            [[-0.75, y, low] for y in (-0.60, -0.74, -0.885)]
            + [[-0.75, y, high] for y in (0.60, 0.74, 0.885)],
            dtype=np.float32,
        )
        assert (np.asarray(field(jnp.asarray(axis))) > 0.0).all()  # the passage is open
        walls = np.asarray(
            [[-0.75 + 0.17, y, low] for y in (-0.60, -0.74)]
            + [[-0.75, y, low + 0.17] for y in (-0.60, -0.74)]
            + [[-0.75 + 0.17, y, high] for y in (0.60, 0.74)],
            dtype=np.float32,
        )
        assert (np.asarray(field(jnp.asarray(walls))) < 0.0).all()  # wrapped in metal

    def test_every_bolt_and_tap_is_drilled(self, shield):
        for i in range(4):
            angle = math.pi / 4 + i * math.pi / 2
            radius = 1.25 * math.sqrt(2.0)
            assert not _inside(
                shield.shield, [radius * math.cos(angle), radius * math.sin(angle), 0.1]
            )
        for i in range(4):
            angle = math.radians(22.5) + i * math.pi / 2
            assert not _inside(shield.shield, [1.16 * math.cos(angle), 1.16 * math.sin(angle), 0.1])
        for i in range(4):
            angle = i * math.pi / 2
            for dy in (-0.08, 0.08):
                x, y = 0.80, dy
                point = [
                    x * math.cos(angle) - y * math.sin(angle),
                    x * math.sin(angle) + y * math.cos(angle),
                    0.97,
                ]
                assert not _inside(shield.shield, point)

    def test_the_encoder_pocket_and_gland(self, shield):
        x, y = shield._pocket_center
        dx, dy = shield._pocket_direction[:2]
        assert not _inside(shield.shield, [x, y, 0.15])  # the pocket
        assert _inside(shield.shield, [x - 0.1 * dy, y + 0.1 * dx, 0.04])  # its floor
        assert not _inside(shield.shield, [1.45 * dx, 1.45 * dy, 0.10])  # the gland bore
        assert _inside(shield.shield, [1.48 * dx, 1.48 * dy, 0.17])  # the gland boss wall

    def test_the_thread_and_the_knurl_and_the_spline(self, shield):
        assert _inside(shield.thread, [0.35, 0.0, 1.81])
        assert not _inside(shield.thread, [0.35, 0.0, 1.81 + 0.075])  # half a pitch up
        assert _inside(shield.locknut, [0.45, 0.0, 1.80])
        assert not _inside(shield.locknut, [0.20, 0.0, 1.80])  # bored for the shaft
        assert _inside(shield.spline, [0.29, 0.0, -0.18])
        gap = math.pi / 24
        assert not _inside(shield.spline, [0.29 * math.cos(gap), 0.29 * math.sin(gap), -0.18])

    def test_the_drain_boss_is_drafted_and_bored_right_through(self, shield):
        """The one cast feature with mould draft: narrower at the top, open end to end."""
        boss = _field(shield.drain_boss)
        centre = shield._drain_center

        def radius_at(height, lo=0.05, hi=0.30):
            for _ in range(30):
                mid = (lo + hi) / 2.0
                point = jnp.asarray([[centre[0] + mid, centre[1], height]], "float32")
                lo, hi = (mid, hi) if float(boss(point)[0]) < 0.0 else (lo, mid)
            return (lo + hi) / 2.0

        bottom, top = radius_at(0.205), radius_at(0.375)
        assert bottom == pytest.approx(0.16, abs=5e-3)  # exact at the bottom cap
        expected = 0.16 - math.tan(math.radians(float(shield.drain_draft.value))) * 0.17
        assert top == pytest.approx(expected, abs=5e-3)
        assert top < bottom

        shell = _field(shield.shield)
        column = np.asarray([[centre[0], centre[1], z] for z in (-0.02, 0.10, 0.30, 0.42)], "f4")
        assert (np.asarray(shell(jnp.asarray(column))) > 0.0).all()

    def test_a_drafted_extrusion_declares_no_faces(self, shield):
        """Why the drain had to be a free-standing cylinder — the documented price."""
        assert len(shield.drain_boss.faces) == 0
        assert len(shield.flange.faces) > 0

    def test_the_nameplate_pad_sits_on_the_tower_wall(self, shield):
        """`SketchPlane.tangent` put the pad on a surface with no analytic face."""
        plane = shield.plate_plane
        # `SketchPlane.origin`/`.normal` hand back the live `Vector` parameters,
        # where a `Face`'s hand back plain arrays; `.xyz` is the bridge.
        origin = np.asarray(plane.origin.xyz, dtype=float)
        # The tower is a 28-gon, not a cylinder, so the projection lands on a
        # facet: between the apothem and the circumradius, with the facet's
        # own normal rather than the radial one.
        apothem = math.cos(math.pi / 28)
        assert apothem <= math.hypot(origin[0], origin[1]) <= 1.0 + 1e-6
        assert float(origin[2]) == pytest.approx(0.62, abs=1e-6)
        normal = np.asarray(plane.normal.xyz, dtype=float)
        radial = np.asarray([math.cos(shield._plate_angle), math.sin(shield._plate_angle), 0.0])
        assert float(normal @ radial) > math.cos(math.pi / 28)  # points out of the wall

        field = _field(shield.shield)
        angle = shield._plate_angle
        proud = np.asarray([[1.03 * math.cos(angle), 1.03 * math.sin(angle), 0.62]], "float32")
        assert float(field(jnp.asarray(proud))[0]) < 0.0
        away = angle + math.radians(15.0)
        beside = np.asarray([[1.03 * math.cos(away), 1.03 * math.sin(away), 0.62]], "float32")
        assert float(field(jnp.asarray(beside))[0]) > 0.0  # bare tower wall

    def test_the_knurl_is_a_diamond_and_not_a_straight_one(self, shield):
        """Two counter-twisted copies of one profile: the teeth pinch toward the caps."""
        field = _field(shield.locknut)
        widths = {}
        for height in (1.80, 1.84, 1.88):
            occupied = _ring(field, 0.455, height, 720) < 0.0
            runs = _tooth_widths(occupied)
            assert len(runs) == 16, height
            widths[height] = sum(runs) / len(runs)
        assert widths[1.80] > widths[1.84] > widths[1.88]
        assert widths[1.88] < 0.5 * widths[1.80]

    def test_the_shroud_is_a_thin_shell_with_windows(self, shield):
        wall = math.pi / 8
        assert _inside(shield.shroud, [1.30 * math.cos(wall), 1.30 * math.sin(wall), 1.38])
        assert not _inside(shield.shroud, [1.30, 0.0, 1.38])  # a window
        assert not _inside(shield.shroud, [0.5, 0.0, 1.38])  # open inside
        assert not _inside(shield.shroud, [1.6, 0.0, 1.38])  # open outside


class TestConstraints:
    def test_every_constrained_sketch_is_satisfied(self, shield):
        for a, b in [
            (shield.lug_a, shield.lug_b),
            (shield.stiffener_a, shield.stiffener_b),
            (shield.pocket_a, shield.pocket_b),
            (shield.rib_heel, shield.rib_toe),
        ]:
            assert float(a.value[1]) == pytest.approx(float(b.value[1]), abs=1e-4)
        assert np.linalg.norm(
            np.asarray(shield.lug_b.value) - np.asarray(shield.lug_a.value)
        ) == pytest.approx(float(shield.lug_size.value), abs=1e-3)
        assert np.linalg.norm(
            np.asarray(shield.rib_crest.value) - np.asarray(shield.rib_toe.value)
        ) == pytest.approx(float(shield.rib_height.value), abs=1e-3)

    def test_no_parameter_came_back_nan(self, shield):
        for name, value in shield.shield_parameters.items():
            assert np.isfinite(np.asarray(value)).all(), name

    def test_the_named_free_parameters_reach_the_model(self, shield):
        assert {"flange_thickness", "bore_radius", "rib_thickness"} <= set(shield.shield_parameters)
        assert len(shield.shield_parameters) == 23


class TestDifferentiability:
    @pytest.mark.parametrize(
        ("name", "step"),
        [("flange_thickness", 2e-3), ("bore_radius", 2e-3), ("rib_thickness", 2e-3)],
    )
    def test_volume_gradient_matches_finite_differences(self, shield, name, step):
        base = dict(shield.shield_parameters)

        def volume(value):
            return shield.shield_volume({**base, name: value})

        start = jnp.asarray(float(base[name]))
        analytic = float(jax.grad(volume)(start))
        finite = (float(volume(start + step)) - float(volume(start - step))) / (2 * step)
        assert analytic == pytest.approx(finite, rel=5e-3)
        assert abs(analytic) > 1e-3

    def test_the_manufacturing_penalty_is_differentiable(self, shield):
        grads = jax.grad(shield.manufacturing_penalty)(dict(shield.shield_parameters))
        assert float(grads["flange_thickness"]) > 0.0  # more flange, more mass
        assert float(grads["bore_radius"]) != 0.0


class TestTheFieldTheSimulationReads:
    """The material field, sampled the way a `FROM_MATERIAL` study samples it.

    Every other test in this file asserts *geometry* — where the metal is —
    and geometry is what `research/complex-scene-2.md` §4.1 says is the only
    reliable evidence on a part this size. It is not enough. Both studies
    take their conductivity, modulus, Poisson ratio and density from the
    scene's own materials, and a boolean bug (§2.6) made every one of them
    `nan` at every interior point while the render stayed perfect and every
    geometric assertion here stayed green. So this class asserts the field
    the solver actually reads.
    """

    #: Interior points of the casting, well clear of every cut wall.
    _INSIDE = (
        (0.00, 0.00, 0.05),  # flange, under the tower
        (1.20, 0.00, 0.05),  # flange rim
        (0.00, 0.00, 0.70),  # bearing tower wall
        (0.75, 0.00, 0.10),  # between the bolt circle and the tower
        (-1.10, -1.10, 0.10),  # a corner lug
    )

    @pytest.mark.parametrize("point", _INSIDE)
    @pytest.mark.parametrize("key", ["conductivity", "youngs_modulus", "poisson_ratio", "density"])
    def test_the_casting_reports_its_own_alloy(self, shield, point, key):
        sampled = shield.shield.material_at(jnp.asarray(point, dtype=jnp.float32))[key]
        expected = float(shield.aluminium.params[key].value)
        assert float(np.asarray(sampled)) == pytest.approx(expected, rel=1e-5)

    def test_no_property_the_studies_need_is_unspecified_anywhere(self, shield):
        """A hole is geometry, not a substance; it must not erase the alloy."""
        from cadjoint.fem.properties import sample_cell_property

        mesh = shield.shield_mesh.build(shield.shield)
        for key in ("conductivity", "youngs_modulus", "poisson_ratio", "density"):
            values = np.asarray(sample_cell_property(shield.shield, mesh.points, mesh.cells, key))
            assert np.isfinite(values).all(), f"{key}: {np.isnan(values).sum()} nan elements"
            assert values.min() > 0.0

    def test_a_study_solves_off_its_simmeshs_domain_alone(self, shield):
        """The call shape `Optimization` uses: a prebuilt mesh and no SDF."""
        mesh = shield.shield_mesh.build(shield.shield)
        result = shield.bearing_heat.solve(mesh=mesh)
        temperature = np.asarray(result.solution.temperature)
        assert np.isfinite(temperature).all()
        assert temperature.max() > 0.0


class TestTheSceneIsViewerReady:
    def test_the_program_assigns_scene(self, shield):
        from cadjoint.sdf.base import SDF

        assert isinstance(shield.scene, SDF)

    def test_meshes_and_studies_and_optimisation_are_captured(self, shield):
        from cadjoint.viewer._worker_scene import _execute_scene

        source = open("scenes/motor_shield.py").read()
        namespace = _execute_scene(source)
        assert "scene" in namespace
        meshes = namespace["__sim_meshes__"]
        studies = namespace["__studies__"]
        opts = namespace["__optimizations__"]
        assert sorted(m.name for m in meshes) == ["shield-hex", "shield-tet10"]
        assert sorted(s.name for s in studies) == ["bearing-heat", "belt-pull"]
        assert [o.name for o in opts] == ["stiff-shield"]

    def test_every_boundary_condition_is_serializable(self, shield):
        for study in (shield.bearing_heat, shield.belt_pull):
            assert all(bc.nodes.serializable for bc in study.bcs)
            for bc in study.bcs:
                payload = bc.describe()
                assert payload["nodes"]["kind"] in {"cylinder", "halfspace"}

    def test_seven_materials_carry_physical_properties(self, shield):
        for material in (
            shield.aluminium,
            shield.steel,
            shield.plastic,
            shield.titanium,
            shield.brass,
            shield.board,
            shield.gap_pad,
        ):
            assert float(material.params["density"].value) > 0.0
