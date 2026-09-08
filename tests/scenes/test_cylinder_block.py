"""The cylinder block: that the model is the casting it was measured from.

``scenes/cylinder_block.py`` is a parametric idealisation of a real production
STEP, and every number in it came from ``research/reverse/step_probe.py``
rather than from a textbook.  These tests pin the measurements *into the
geometry*: not "the file says the bore pitch is 86" — an assertion about a
constant proves nothing — but "there is iron at the siamese web between two
bores 86 mm apart, and open bore either side of it".

The measurements these assert are:

===============================  =========  ==========
quantity                         guessed    measured
===============================  =========  ==========
bore diameter                    78.0       75.50
bore pitch                       92.0       86.00
deck height above the crank      212.0      206.43
main saddle radius               27.0       26.90
bulkhead thickness at the saddle 20.0       20.00
head bolt offset across          58.0       43.00
pan rail below the crank axis    78.0       15.07
deck construction                open       closed
bores                            4 mm slot  siamesed
===============================  =========  ==========

Two of those changed the *shape* of the part, not just its numbers, and both
have a test here that the guessed version would have failed: the deck is
closed, and the barrels touch.

The STEP itself is not in this repository and is not needed: everything below
runs against the scene alone.  The comparison that does need the file lives in
``research/reverse/overlay.py``.

Distances are in millimetres throughout, converted at the scene's own ``MM``.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

#: Bore axes and main-bearing stations, in millimetres from the block's centre.
#: Read back off ``bore_pitch`` in the first test rather than trusted.
BORES = (-129.0, -43.0, 43.0, 129.0)
MAINS = (-172.0, -86.0, 0.0, 86.0, 172.0)


@pytest.fixture(scope="module")
def blk():
    return importlib.import_module("scenes.cylinder_block")


def solid(blk, x, y, z, *, field=None) -> bool:
    """Is the casting present at this millimetre point?"""
    fn = blk.block if field is None else field
    point = jnp.asarray([x * blk.MM, y * blk.MM, z * blk.MM], dtype=jnp.float32)
    return float(fn(point)) < 0.0


def perturbed(blk, **overrides):
    """The block's frozen field with some free parameters moved, in mm."""
    parameters = dict(blk.block_parameters)
    for name, value in overrides.items():
        parameters[name] = jnp.asarray(value * blk.MM)
    return blk.block_sdf(parameters, blk.block_fixed)


class TestTheMeasuredLayout:
    """One pitch sets the bores, the bulkheads, the bolts and the slots."""

    def test_the_pitch_is_the_measured_86_and_not_the_guessed_92(self, blk):
        pitch = float(blk.bore_pitch.value) / blk.MM
        assert pitch == pytest.approx(86.0, abs=1e-3)
        assert BORES == pytest.approx([pitch * (i - 1.5) for i in range(4)], abs=1e-3)
        assert MAINS == pytest.approx([pitch * (i - 2) for i in range(5)], abs=1e-3)

    def test_the_bore_is_the_measured_75_5(self, blk):
        assert float(blk.bore_radius.value) / blk.MM == pytest.approx(37.75, abs=1e-3)

    def test_the_deck_and_the_pan_rail_are_where_the_probe_put_them(self, blk):
        """206.43 above the crank axis and 15.07 below it — a shallow block."""
        assert float(blk.deck.origin[2]) / blk.MM == pytest.approx(206.43, abs=0.05)
        assert blk.RAIL_Z == pytest.approx(-15.0, abs=0.1)

    def test_the_patterns_are_seeded_at_the_minus_x_end(self, blk):
        """A LinearPattern's copy 0 is its seed and the rest march in +x."""
        assert float(blk.FIRST_BORE_X) == pytest.approx(BORES[0])
        assert float(blk.FIRST_MAIN_X) == pytest.approx(MAINS[0])


class TestTheEnvelope:
    def test_the_deck_is_the_top_of_the_casting(self, blk):
        assert solid(blk, 0, 20, 204)
        assert not solid(blk, 0, 20, 209)

    def test_the_pan_rail_is_the_bottom(self, blk):
        assert solid(blk, 0, -100, -10) and solid(blk, 0, 100, -10)
        assert not solid(blk, 0, 100, -19)

    def test_the_width_goes_outward_at_the_ends_only(self, blk):
        """349 mm across the two end flanges, 152 across the deck between them.

        The casting has no deep skirt; it spends its width on flanges instead,
        which is the single biggest thing the guessed version got wrong.
        """
        assert solid(blk, -178, -150, 0) and solid(blk, 178, 150, 0)
        assert not solid(blk, 0, -150, 0)
        assert not solid(blk, 0, 150, 0)


class TestTheDeckIsClosed:
    """The measurement that changed the part, not just its dimensions."""

    def test_there_is_a_deck_plate_between_the_bores(self, blk):
        for x in MAINS:
            assert solid(blk, x, 0, 202), f"the deck is open at x={x}"

    def test_the_bores_still_go_through_it(self, blk):
        for x in BORES:
            assert not solid(blk, x, 0, 210), f"bore {x} does not reach above the deck"
            assert not solid(blk, x, 0, 202), f"bore {x} is capped by the deck plate"
            assert not solid(blk, x, 0, 60), f"bore {x} does not open into the crankcase"

    def test_the_deck_covers_the_water_jacket(self, blk):
        """The jacket is open at deck level only through its sixteen slots."""
        for x in BORES:
            assert solid(blk, x, 55.0, 202), "the deck should close over the jacket"
            assert not solid(blk, x, 55.0, 140), "and the jacket should be open below it"

    def test_sixteen_coolant_slots_pierce_it(self, blk):
        for bore in BORES:
            for stagger in (-blk.SLOT_STAGGER, blk.SLOT_STAGGER):
                for y in (-blk.SLOT_Y, blk.SLOT_Y):
                    assert not solid(blk, bore + stagger, y, 202), "a coolant slot is missing"


class TestTheBoresAreSiamesed:
    """86 mm pitch, 86 mm over the barrels: there is no water between them."""

    def test_iron_between_neighbouring_bores(self, blk):
        for x in (-86.0, 0.0, 86.0):
            assert solid(blk, x, 0, 140), f"the siamese web at x={x} is open"

    def test_five_millimetres_of_barrel_wall_and_then_water(self, blk):
        for x in BORES:
            assert not solid(blk, x + 35.0, 0, 140), "the bore is not open to its own radius"
            assert solid(blk, x + 40.0, 0, 140), "the barrel wall is missing"
            assert not solid(blk, x, 55.0, 140), "the jacket is solid beside the barrel"

    def test_the_jacket_has_a_floor(self, blk):
        for x in BORES:
            assert solid(blk, x, 55.0, 78), "the jacket floor is missing"


class TestTheBottomEnd:
    def test_a_web_stands_at_every_main_station_and_nowhere_else(self, blk):
        for x in MAINS:
            for y in (-60.0, 60.0):
                assert solid(blk, x, y, 12), f"no bulkhead at x={x}"
        for x in BORES:
            for y in (-60.0, 60.0):
                assert not solid(blk, x, y, 12), f"a web has grown under the bore at x={x}"

    def test_the_parting_face_is_the_crank_axis(self, blk):
        """Measured: there is no block iron at all below z = 0 between the rails."""
        for x in MAINS:
            for y in (-60.0, 60.0):
                assert not solid(blk, x, y, -6), f"block iron below the parting plane at x={x}"

    def test_every_saddle_is_bored_to_the_fitted_radius(self, blk):
        radius = float(blk.main_saddle_radius.value) / blk.MM
        assert radius == pytest.approx(26.9, abs=1e-3)
        for x in MAINS:
            assert not solid(blk, x, 0, radius - 7.0), f"saddle {x} is not bored"
            assert solid(blk, x, 0, radius + 7.0), f"saddle {x} has no roof"

    def test_the_line_bore_runs_through_both_end_walls(self, blk):
        for x in (-190.0, -150.0, -80.0, 0.0, 80.0, 150.0, 190.0):
            assert not solid(blk, x, 0, 0), f"the crank tunnel is blocked at x={x}"

    def test_each_web_is_drilled_for_two_cap_bolts(self, blk):
        for x in MAINS:
            for y in (-blk.MAIN_BOLT_Y, blk.MAIN_BOLT_Y):
                assert not solid(blk, x, y, 16), f"cap bolt ({x}, {y}) is not drilled"

    def test_each_web_carries_its_lightening_window(self, blk):
        for x in MAINS:
            assert not solid(blk, x, blk.WINDOW_Y, blk.WINDOW_Z), f"no window in web {x}"
            assert solid(blk, x, blk.WINDOW_Y, blk.WINDOW_Z + 17.0), "the window swallowed the web"


class TestTheHeadBolts:
    def test_ten_of_them_at_the_measured_offset(self, blk):
        """43 mm across, not the 58 this file guessed: they land on the webs.

        On a siamesed bank there is no iron between the bores to bolt into, so
        the bolts go on the main-bearing stations instead of the bore ones.
        """
        assert blk.HEAD_BOLT_Y == pytest.approx(43.0)
        for x in MAINS:
            for y in (-43.0, 43.0):
                assert not solid(blk, x, y, 202), f"head bolt ({x}, {y}) is not drilled"
        for x in MAINS:
            for y in (-30.0, 30.0):
                assert solid(blk, x, y, 202), f"head bolt ({x}, {y}) has no boss"


class TestTheCastingIsHollow:
    """The check that would catch a core failing to cut."""

    @pytest.fixture(scope="class")
    def lattice(self, blk):
        axes = [
            np.arange(-190.0, 191.0, 7.0),
            np.arange(-180.0, 181.0, 7.0),
            np.arange(-18.0, 209.0, 7.0),
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        points = jnp.asarray((grid * blk.MM).astype(np.float32))
        chunks = [np.asarray(jax.vmap(blk.block)(c)) for c in jnp.array_split(points, 40)]
        return np.concatenate(chunks) / blk.MM

    def test_nothing_in_the_casting_is_far_from_a_surface(self, blk, lattice):
        """The real part's mean wall is 2V/A = 6.6 mm; this model's is similar.

        A solid billet of this envelope would read about -76 mm. Twenty is well
        past anything a wall of a casting like this can produce, and it is the
        assertion a jacket or crankcase that stopped cutting would fail.
        """
        assert lattice.min() > -20.0, f"deepest interior point {lattice.min():.1f} mm"

    def test_it_is_mostly_air(self, blk, lattice):
        fraction = float((lattice < 0).mean())
        assert 0.06 < fraction < 0.22, f"solid fraction {fraction:.3f} of the envelope box"

    def test_the_field_is_a_distance_field_the_viewer_can_march(self, blk):
        rng = np.random.default_rng(0)
        a = rng.uniform(-1.0, 1.0, size=(600, 3)).astype(np.float32)
        b = (a + rng.normal(0.0, 0.05, size=a.shape)).astype(np.float32)
        fa = np.asarray(jax.vmap(blk.block)(jnp.asarray(a)))
        fb = np.asarray(jax.vmap(blk.block)(jnp.asarray(b)))
        step = np.maximum(np.linalg.norm(a - b, axis=-1), 1e-9)
        assert np.isfinite(fa).all() and np.isfinite(fb).all()
        assert (np.abs(fa - fb) / step).max() < 1.15


class TestTheParametersMoveWhatTheyName:
    def test_opening_the_bore_thins_the_barrel_wall(self, blk):
        wall = perturbed(blk, bore_radius=41.0)
        for x in BORES:
            assert solid(blk, x + 40.0, 0, 140), "the wall should start out there"
            assert not solid(blk, x + 40.0, 0, 140, field=wall), "and be gone at 41"

    def test_a_deeper_jacket_drops_its_floor(self, blk):
        deep = perturbed(blk, jacket_depth=112.73 + 20.0)
        assert solid(blk, 43, 55, 80), "the floor starts above 80"
        assert not solid(blk, 43, 55, 80, field=deep), "and 10 mm of it should be gone"

    def test_a_bigger_journal_opens_every_saddle(self, blk):
        big = perturbed(blk, main_saddle_radius=34.0)
        for x in MAINS:
            assert solid(blk, x, 0, 31), "26.9 mm of tunnel leaves iron at 31"
            assert not solid(blk, x, 0, 31, field=big), "34 mm does not"

    def test_a_thicker_web_is_thicker(self, blk):
        thick = perturbed(blk, bulkhead_thickness=44.0)
        assert not solid(blk, 18, 60, 12), "a 20 mm web reaches x = 10"
        assert solid(blk, 18, 60, 12, field=thick), "a 44 mm web reaches x = 22"

    def test_block_height_is_a_driving_dimension_and_not_a_free_one(self, blk):
        """Deliberate, and the docstring says why.

        Five features are sketched on this extrusion's faces. Re-running the
        program re-derives all of them; the frozen field cannot, because by
        then a face-derived plane origin is a number. Freeing it would hand an
        optimizer a deck that rises away from its own bores.
        """
        assert blk.block_height.free is False
        assert "block_height" not in blk.block_parameters
        assert float(blk.block_height.value) / blk.MM == pytest.approx(221.43, abs=0.05)

    def test_re_running_the_program_carries_the_stack_with_the_deck(self, blk):
        from cadjoint.construction import PolygonProfile, extrude
        from cadjoint.geometry import Scalar

        def deck_and_bore(thickness):
            plate = extrude(
                PolygonProfile.rounded_rect(
                    1.87, 0.76, 0.11, segments=2, plane=blk.deck_profile.plane, name="p"
                ),
                depth=Scalar(thickness),
            )
            tool = plate.cap("+").hole(0.18, depth=0.6, at=(-0.645, 0.0))
            return float(plate.cap("+").origin[2]), float(tool.params["offset"].xyz[2])

        thin = deck_and_bore(0.051)
        thick = deck_and_bore(0.071)
        # The deck rises by half the change and the bore sunk from it follows.
        assert thick[0] - thin[0] == pytest.approx(0.01, abs=1e-5)
        assert thick[1] - thin[1] == pytest.approx(0.01, abs=1e-5)


class TestTheReferenceFrame:
    """The map back to the STEP, which the overlay reads."""

    def test_the_scene_declares_where_it_came_from(self, blk):
        frame = blk.REFERENCE
        assert set(frame) == {"step", "unit_mm", "origin_mm", "axes"}
        assert frame["unit_mm"] == 200.0

    def test_the_frame_carries_a_bore_axis_onto_the_bore_axis(self, blk):
        """world (826.4, 5, 416.46) is bore 1 at the deck; scene (-129, 0, 206.4)."""
        origin = np.asarray(blk.REFERENCE["origin_mm"])
        axes = np.asarray(blk.REFERENCE["axes"])
        scene = axes @ (np.array([826.4, 5.0, 416.46]) - origin)
        assert scene == pytest.approx([BORES[0], 0.0, 206.43], abs=0.05)

    def test_the_scene_imports_without_the_step_file(self, blk):
        """The reference lives on one machine; the scene must not need it."""
        import os

        assert blk.scene is not None
        assert isinstance(blk.REFERENCE["step"], str)
        # Whether the file happens to be here or not, the module already
        # imported — which is the whole assertion.
        assert os.path.isabs(blk.REFERENCE["step"])


class TestDifferentiability:
    """After a dozen booleans, the casting still carries a gradient."""

    #: Per-parameter tolerance, and the reason it differs. Three of the four
    #: move a surface that the volume lattice resolves cleanly and agree with a
    #: central difference to under a percent. ``jacket_depth`` does not, and
    #: the reason is worth knowing rather than papering over: it moves the
    #: jacket's roof *up into a 10 mm deck plate* and its floor down into a
    #: 10 mm web, and both of those surfaces are places where the block's hard
    #: booleans switch branch. `jax.grad` follows the branch that is active at
    #: the sample; a 0.8 mm difference step crosses some of them. The gradient
    #: is right — it is the finite difference that is a poor witness there.
    @pytest.mark.parametrize(
        ("name", "step", "sign", "tolerance"),
        [
            ("bore_radius", 2e-3, -1.0, 1e-2),
            ("jacket_depth", 4e-3, -1.0, 2.5e-1),
            ("main_saddle_radius", 2e-3, -1.0, 3e-2),
            ("bulkhead_thickness", 2e-3, +1.0, 3e-2),
        ],
    )
    def test_volume_gradient_matches_finite_differences(self, blk, name, step, sign, tolerance):
        base = dict(blk.block_parameters)

        def volume(value):
            return blk.block_volume({**base, name: value})

        start = jnp.asarray(float(base[name]))
        analytic = float(jax.grad(volume)(start))
        finite = (float(volume(start + step)) - float(volume(start - step))) / (2 * step)
        assert analytic == pytest.approx(finite, rel=tolerance)
        assert abs(analytic) > 1e-2, "the parameter must actually move the volume"
        assert analytic * sign > 0.0

    def test_every_free_parameter_reaches_the_frozen_model(self, blk):
        for name in ("bore_radius", "jacket_depth", "main_saddle_radius", "bulkhead_thickness"):
            assert name in blk.block_parameters

    def test_no_parameter_came_back_nan(self, blk):
        for name, value in blk.block_parameters.items():
            assert np.isfinite(np.asarray(value)).all(), name


class TestTheSceneIsViewerReady:
    def test_the_program_assigns_scene(self, blk):
        from cadjoint.sdf.base import SDF

        assert isinstance(blk.scene, SDF)

    def test_the_casting_is_named_and_the_caps_are_not_part_of_it(self, blk):
        assert blk.block.name == "block"
        assert solid(blk, -172, 40, -25, field=blk.scene)
        assert not solid(blk, -172, 40, -25)

    def test_the_two_materials_are_distinct(self, blk):
        colors = {
            tuple(round(float(c), 4) for c in m.params["color"].xyz)
            for m in (blk.cast_iron, blk.nodular_iron)
        }
        assert len(colors) == 2

    def test_the_scene_evaluates_a_material_at_a_point(self, blk):
        material = blk.scene.material_at(jnp.asarray([0.0, 0.15, 1.01], dtype=jnp.float32))
        assert len(material["color"]) == 3
