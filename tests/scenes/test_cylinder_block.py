"""The cylinder block: that the casting is the one the file describes.

``scenes/cylinder_block.py`` claims a specific engine.  Four bores open
through the deck on a 92 mm pitch, a water jacket that is a cored void rather
than a pocket, five main-bearing bulkheads parting on the crank axis, one
line bore through all of them, and ten head bolts that bottom out in the
jacket floor.  Each of those is a sentence that can be false while the scene
still renders as a convincing lump of iron, so each is measured here.

The assertions are chosen for the errors they would actually catch.  Two of
them caught real ones while this scene was being written: every boolean in
the file names its blend because ``Difference`` defaults to a 0.1-unit
smoothness — twenty millimetres at this scale — which had quietly eaten the
whole water jacket; and the head bolts are 132 mm deep because at 150 they
broke through the jacket floor into the crankcase.

Distances are in millimetres throughout, converted at the scene's own ``MM``.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

#: Bore axes and main-bearing stations, in millimetres from the block's centre.
#: Both are read back off ``bore_pitch`` in the first test rather than trusted.
BORES = (-138.0, -46.0, 46.0, 138.0)
MAINS = (-184.0, -92.0, 0.0, 92.0, 184.0)


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


class TestThePitchDrivesEverything:
    """One number sets the bores, the bulkheads and the head-bolt columns."""

    def test_the_stations_this_file_probes_are_the_ones_the_pitch_puts_there(self, blk):
        pitch = float(blk.bore_pitch.value) / blk.MM
        assert pitch == pytest.approx(92.0, abs=1e-3)
        # Bores sit on half-pitch offsets, main bearings on whole ones — which
        # is what puts a main bearing between and outside every cylinder.
        assert BORES == pytest.approx([pitch * (i - 1.5) for i in range(4)], abs=1e-3)
        assert MAINS == pytest.approx([pitch * (i - 2) for i in range(5)], abs=1e-3)

    def test_the_patterns_are_centred_rather_than_walked_off_one_end(self, blk):
        """A LinearPattern's copy 0 is its seed, so the seed must be the -x end."""
        assert float(blk.FIRST_BORE_X) == pytest.approx(BORES[0])
        assert float(blk.FIRST_MAIN_X) == pytest.approx(MAINS[0])


class TestTheEnvelope:
    def test_the_deck_and_the_pan_rail_are_where_block_height_puts_them(self, blk):
        assert float(blk.deck.origin[2]) / blk.MM == pytest.approx(212.0, abs=1e-2)
        assert float(blk.pan_rail.origin[2]) / blk.MM == pytest.approx(-78.0, abs=1e-2)

    def test_the_deck_is_152_wide_and_410_long(self, blk):
        assert solid(blk, 0, 70, 190) and solid(blk, 200, 70, 190)
        assert not solid(blk, 0, 82, 190)
        assert not solid(blk, 212, 70, 190)

    def test_the_skirt_flares_out_below_the_crank_axis(self, blk):
        """The taper is the whole reason the skirt is a loft and not a box."""
        assert solid(blk, 0, 95, -60), "the skirt should be wider than the deck"
        assert not solid(blk, 0, 95, 60), "the deck should not be"

    def test_the_pan_rail_flange_stands_proud_of_the_skirt(self, blk):
        assert solid(blk, 0, 106, -72)
        assert not solid(blk, 0, 106, -55)


class TestTheBores:
    def test_all_four_bores_are_open_from_above_the_deck_into_the_crankcase(self, blk):
        for x in BORES:
            for z in (218.0, 200.0, 120.0, 60.0):
                assert not solid(blk, x, 0, z), f"bore at x={x} is blocked at z={z}"

    def test_each_bore_is_walled_by_a_barrel_and_not_by_the_block(self, blk):
        """5 mm of iron between the bore and the water, and nothing more."""
        for x in BORES:
            assert not solid(blk, x + 36.0, 0, 150), "the bore is not open to its own radius"
            assert solid(blk, x + 41.5, 0, 150), "the barrel wall is missing"
            assert not solid(blk, x, 50.0, 150), "the jacket is solid beside the barrel"

    def test_the_bores_are_siamesed_but_not_welded(self, blk):
        """A 4 mm coolant slot between neighbours: 92 mm pitch, 88 mm barrels."""
        for x in (-92.0, 0.0, 92.0):
            assert not solid(blk, x, 0, 150), f"the slot at x={x} is cast shut"
        # …and the barrel walls either side of that slot are still there.
        assert solid(blk, -96.5, 0, 150)
        assert solid(blk, -87.5, 0, 150)


class TestTheWaterJacket:
    """A void, and a connected one — not a pocket and not solid iron."""

    def test_the_jacket_runs_the_length_of_the_bank_on_both_flanks(self, blk):
        for x in BORES:
            for y in (-50.0, 50.0):
                assert not solid(blk, x, y, 150), f"the jacket is blocked at ({x}, {y})"

    def test_the_jacket_has_a_floor_and_an_outer_wall(self, blk):
        """Coolant is contained; a jacket open to the crankcase is not a jacket."""
        for x in BORES:
            assert not solid(blk, x, 50, 100), "no jacket above the floor"
            assert solid(blk, x, 50, 80), "the jacket floor is missing"
            assert solid(blk, x, 70, 150), "the outer wall is missing"

    def test_the_ten_bolt_bosses_stand_in_the_jacket(self, blk):
        """Nothing models a boss: they are what the jacket cut steps around."""
        for x in MAINS:
            for y in (-50.0, 50.0):
                assert solid(blk, x, y, 150), f"no boss at the main station x={x}"


class TestTheBottomEnd:
    def test_a_bulkhead_stands_at_every_main_station_and_nowhere_else(self, blk):
        for x in MAINS:
            for y in (-35.0, 35.0):
                assert solid(blk, x, y, 15), f"no bulkhead at x={x}"
        for x in BORES:
            for y in (-35.0, 35.0):
                assert not solid(blk, x, y, 15), f"a web has grown under the bore at x={x}"

    def test_the_bulkheads_part_on_the_crank_axis(self, blk):
        """z = 0 is the main-cap joint: block above it, cap below it."""
        for x in MAINS:
            for y in (-35.0, 35.0):
                assert not solid(blk, x, y, -15), f"block iron below the parting plane at x={x}"

    def test_every_saddle_is_bored_to_the_tunnel_radius(self, blk):
        radius = float(blk.main_saddle_radius.value) / blk.MM
        assert radius == pytest.approx(27.0, abs=1e-3)
        for x in MAINS:
            assert not solid(blk, x, 0, radius - 7.0), f"saddle {x} is not bored"
            assert solid(blk, x, 0, radius + 8.0), f"saddle {x} has no roof"

    def test_the_line_bore_runs_through_both_end_walls(self, blk):
        """One cylinder opens all five saddles, which is how it is really cut."""
        for x in (-200.0, -150.0, -100.0, 0.0, 100.0, 150.0, 200.0):
            assert not solid(blk, x, 0, 0), f"the crank tunnel is blocked at x={x}"

    def test_each_bulkhead_is_drilled_for_two_cap_bolts(self, blk):
        for x in MAINS:
            for y in (-45.0, 45.0):
                assert not solid(blk, x, y, 20), f"cap bolt ({x}, {y}) is not drilled"


class TestTheHeadBolts:
    def test_ten_holes_open_at_the_deck(self, blk):
        for x in MAINS:
            for y in (-58.0, 58.0):
                assert not solid(blk, x, y, 190), f"head bolt ({x}, {y}) is not drilled"
                assert solid(blk, x, y * 49 / 58, 190), f"head bolt ({x}, {y}) has no boss"

    def test_they_bottom_out_in_the_jacket_floor(self, blk):
        """At 150 mm they broke into the crankcase; at 132 they stop in iron."""
        for x in MAINS:
            for y in (-58.0, 58.0):
                assert solid(blk, x, y, 78), f"head bolt ({x}, {y}) went through the floor"


class TestTheCastingIsHollow:
    """The check that would have caught the jacket not cutting at all."""

    @pytest.fixture(scope="class")
    def lattice(self, blk):
        axes = [
            np.arange(-210.0, 211.0, 7.0),
            np.arange(-112.0, 113.0, 7.0),
            np.arange(-84.0, 219.0, 7.0),
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        points = jnp.asarray((grid * blk.MM).astype(np.float32))
        chunks = [np.asarray(jax.vmap(blk.block)(c)) for c in jnp.array_split(points, 40)]
        return np.concatenate(chunks) / blk.MM

    def test_nothing_in_the_casting_is_far_from_a_surface(self, blk, lattice):
        """A block is walls. The thickest section here is a 20 mm floor web.

        A solid billet of this envelope would read about -76 mm; the jacket
        failing to cut read -28. Twenty-five is comfortably past both.
        """
        assert lattice.min() > -25.0, f"deepest interior point {lattice.min():.1f} mm"

    def test_it_is_mostly_air(self, blk, lattice):
        fraction = float((lattice < 0).mean())
        assert 0.12 < fraction < 0.35, f"solid fraction {fraction:.3f} of the envelope box"

    def test_the_field_is_a_distance_field_the_viewer_can_march(self, blk):
        """Raymarching needs 1-Lipschitz; a mis-composed field is what breaks it."""
        rng = np.random.default_rng(0)
        a = rng.uniform(-1.3, 1.3, size=(600, 3)).astype(np.float32)
        b = (a + rng.normal(0.0, 0.05, size=a.shape)).astype(np.float32)
        fa = np.asarray(jax.vmap(blk.block)(jnp.asarray(a)))
        fb = np.asarray(jax.vmap(blk.block)(jnp.asarray(b)))
        step = np.maximum(np.linalg.norm(a - b, axis=-1), 1e-9)
        assert np.abs(fa - fb).max() / step.min() < 1e9  # guards against NaN
        assert (np.abs(fa - fb) / step).max() < 1.05


class TestTheParametersMoveWhatTheyName:
    def test_opening_the_bore_thins_the_barrel_wall(self, blk):
        """The barrel OD is the casting core; the bore eats into it."""
        wall = perturbed(blk, bore_radius=42.5)
        for x in BORES:
            assert solid(blk, x + 41.5, 0, 150), "the wall should start out there"
            assert not solid(blk, x + 41.5, 0, 150, field=wall), "and be gone at 42.5"

    def test_a_deeper_jacket_drops_its_floor(self, blk):
        deep = perturbed(blk, jacket_depth=128.0 + 24.0)
        assert solid(blk, 46, 50, 85), "the floor starts above 85"
        assert not solid(blk, 46, 50, 85, field=deep), "and 12 mm of it should be gone"
        assert solid(blk, 46, 50, 74, field=deep), "but not all of it"

    def test_a_bigger_journal_opens_every_saddle(self, blk):
        big = perturbed(blk, main_saddle_radius=34.0)
        for x in MAINS:
            assert solid(blk, x, 0, 32), "27 mm of tunnel leaves iron at 32"
            assert not solid(blk, x, 0, 32, field=big), "34 mm does not"

    def test_a_thicker_bulkhead_is_thicker(self, blk):
        thick = perturbed(blk, bulkhead_thickness=44.0)
        assert not solid(blk, 18, 35, 30), "a 20 mm web reaches x = 10"
        assert solid(blk, 18, 35, 30, field=thick), "a 44 mm web reaches x = 22"

    def test_block_height_is_a_driving_dimension_and_not_a_free_one(self, blk):
        """Deliberate, and the docstring says why.

        Five features are sketched on this extrusion's faces. Re-running the
        program re-derives all of them — that is the test below. The frozen
        field cannot: it substitutes into the nodes that hold a parameter, and
        by then a face-derived plane origin is a number. Freeing it would hand
        an optimizer a deck that rises away from its own bores.
        """
        assert blk.block_height.free is False
        assert "block_height" not in blk.block_parameters
        # It still drives the part: the deck it puts at 212 mm is the face the
        # bores, the jacket and the head bolts are all sunk from.
        assert float(blk.block_height.value) / blk.MM == pytest.approx(290.0, abs=1e-2)

    def test_re_running_the_program_carries_the_whole_stack_with_the_deck(self, blk):
        from cadjoint.construction import PolygonProfile, extrude
        from cadjoint.geometry import Scalar

        def deck_and_flange(height):
            body = extrude(
                PolygonProfile.rounded_rect(
                    2.05, 0.76, 0.09, segments=2, plane=blk.body_profile.plane, name="b"
                ),
                depth=Scalar(height),
            )
            flange = extrude(
                PolygonProfile.rounded_rect(
                    2.09, 1.09, 0.12, segments=2, plane=body.cap("-").plane(offset=-0.035), name="f"
                ),
                depth=0.07,
            )
            return float(body.cap("+").origin[2]), float(flange.cap("+").origin[2])

        low = deck_and_flange(1.45)
        high = deck_and_flange(1.55)
        # The deck rises by half the change and the pan-rail flange, sketched
        # on the *other* face, drops by the same half. Neither is a number in
        # this file; both are expressions in the depth.
        assert high[0] - low[0] == pytest.approx(0.05, abs=1e-5)
        assert high[1] - low[1] == pytest.approx(-0.05, abs=1e-5)


class TestTheBulkheadSketch:
    def test_the_parting_pads_lie_on_the_crank_axis(self, blk):
        near = np.asarray(blk.bulkhead_pad_near.value)
        far = np.asarray(blk.bulkhead_pad_far.value)
        # Profile x is world height measured downward, so x = 0 IS z = 0.
        assert near[0] == pytest.approx(0.0, abs=1e-5)
        assert far[0] == pytest.approx(0.0, abs=1e-5)

    def test_the_pad_width_is_the_driving_dimension(self, blk):
        near = np.asarray(blk.bulkhead_pad_near.value)
        far = np.asarray(blk.bulkhead_pad_far.value)
        width = float(np.linalg.norm(far - near))
        assert width == pytest.approx(float(blk.saddle_pad_width.value), abs=1e-4)

    def test_the_crown_is_flat(self, blk):
        near = np.asarray(blk.bulkhead_crown_near.value)
        far = np.asarray(blk.bulkhead_crown_far.value)
        assert near[0] == pytest.approx(far[0], abs=1e-5)

    def test_no_parameter_came_back_nan(self, blk):
        for name, value in blk.block_parameters.items():
            assert np.isfinite(np.asarray(value)).all(), name


class TestDifferentiability:
    """After ten booleans nested four deep, the casting carries a gradient."""

    @pytest.mark.parametrize(
        ("name", "step", "sign"),
        [
            ("bore_radius", 2e-3, -1.0),
            ("jacket_depth", 4e-3, -1.0),
            ("main_saddle_radius", 2e-3, -1.0),
            ("bulkhead_thickness", 2e-3, +1.0),
        ],
    )
    def test_volume_gradient_matches_finite_differences(self, blk, name, step, sign):
        base = dict(blk.block_parameters)

        def volume(value):
            return blk.block_volume({**base, name: value})

        start = jnp.asarray(float(base[name]))
        analytic = float(jax.grad(volume)(start))
        finite = (float(volume(start + step)) - float(volume(start - step))) / (2 * step)
        assert analytic == pytest.approx(finite, rel=5e-3)
        assert abs(analytic) > 1e-2, "the parameter must actually move the volume"
        # Three of the four are cuts and can only remove iron; the bulkhead
        # is the one that adds it.
        assert analytic * sign > 0.0

    def test_every_free_parameter_reaches_the_frozen_model(self, blk):
        for name in ("bore_radius", "jacket_depth", "main_saddle_radius", "bulkhead_thickness"):
            assert name in blk.block_parameters


class TestTheSceneIsViewerReady:
    def test_the_program_assigns_scene(self, blk):
        from cadjoint.sdf.base import SDF

        assert isinstance(blk.scene, SDF)

    def test_the_casting_is_named_and_the_caps_are_not_part_of_it(self, blk):
        assert blk.block.name == "block"
        # The caps are context: iron the block is bolted to, in its own
        # material, bored by the same tunnel and unioned in at the very end.
        assert solid(blk, -184, 40, -25, field=blk.scene)
        assert not solid(blk, -184, 40, -25)

    def test_the_two_materials_are_distinct(self, blk):
        colors = {
            tuple(round(float(c), 4) for c in m.params["color"].xyz)
            for m in (blk.cast_iron, blk.nodular_iron)
        }
        assert len(colors) == 2

    def test_the_scene_evaluates_a_material_at_a_point(self, blk):
        material = blk.scene.material_at(jnp.asarray([0.0, 0.35, 0.95], dtype=jnp.float32))
        assert len(material["color"]) == 3
