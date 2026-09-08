"""Inline-four cylinder block, reverse-engineered from a measured STEP.

A parametric idealisation of a real production casting: the ``m15
cylinderblock`` CATIA V5R21 model, 4069 faces, 3776 cm³ of iron, 27 kg.
Every dimension in this file was **measured**, not assumed — see the table at
the end of this docstring for what the measurement changed.

**This is an idealisation, and deliberately so.**  A hand-authored SDF cannot
reproduce a 4069-face CATIA casting with 2418 B-spline surfaces, and chasing
the intersection-over-union toward its own ceiling is the wrong target.  The
goal here is that somebody who knows engine blocks would identify *this*
block from a render — from the silhouette and the feature layout — so the
work was driven by section overlays judged by eye
(``research/reverse/sections.py``) rather than by a scalar.

The part turns out to be a **closed-deck, siamesed-bore, cast-iron inline
four** with the main-bearing parting face on the crank axis:

  * **four bores**, 75.5 mm on an 86.0 mm pitch, so the barrels — 86.0 mm
    over the walls — touch.  That is not a modelling convenience, it is the
    casting: 5.2 mm of iron between a bore and the water, and none at all
    between one bore and the next.
  * a **closed deck**: a 10.2 mm plate over the water jacket, pierced by
    sixteen elongated coolant slots and ten head-bolt holes, not the open
    deck this file guessed at before it had the file to measure.
  * a **water jacket** 112.7 mm deep, floor to deck plate, cored around the
    barrel bank.
  * **five main-bearing bulkheads**, 20.0 mm thick on the same 86 mm pitch,
    each with a 53.8 mm saddle arch whose centre *is* the crank axis, and a
    Ø28 lightening window above it.
  * a **shallow crankcase**: the pan rail sits only 15 mm below the crank
    axis, and the width instead goes outward — 114 mm across the deck between
    its lugs, 232 across the crankcase rails, 349 across the two end flanges.
  * **lobed flanks**.  The outer wall follows the barrels, bulging over each
    cylinder and pinching at the waist between them, and it does it at two
    different radii — 63 mm at the jacket floor, 52 under the deck.  Drawn as
    a rounded rectangle the part reads as a box; this is the single feature
    that makes the plan view recognisable.
  * the **furniture that identifies a flank in a photograph**: a continuous
    rail under the deck, a raised bolt pad on every main-bearing station, an
    integral oil-gallery tube running the whole length at (y = +59, z = 113),
    scalloped lugs round the deck, and the Ø90 crank-seal opening in the
    drive-end face.

**How it was measured.**  ``research/reverse/step_probe.py`` walks the STEP's
faces.  A production CAD file is already a drawing: a bore is a
``GeomAbs_Cylinder`` with an exact radius and an exact axis, so the bore
pitch is the distance between two axes rather than an inference, and the deck
is the extreme plane normal to them.  What the faces do not say — wall
thickness, the height of a jacket floor, where a *cast* saddle sits, since
this block's saddle is spline surfaces and not a cylinder — comes from
casting axis-aligned rays through the tessellation.  The saddle here is a
circle fitted to the arch's half-width against height: radius 26.90 mm,
centre 0.29 mm from a perfect circle over 8 samples.

**Guessed against measured.**  The previous version of this file was built
from published class proportions for a 1.5 L four, because the reference was
behind a login.  The comparison is worth keeping::

    quantity                        guessed   measured    error
    bore diameter                      78.0      75.50    +3.3%
    bore pitch                         92.0      86.00    +7.0%
    deck height above the crank axis  212.0     206.43    +2.7%
    main saddle radius                 27.0      26.90    +0.4%
    bulkhead thickness                 20.0      20.00     exact
    bulkhead count                        5          5     exact
    barrel wall                         5.0       5.25    -4.8%
    head bolt count                      10         10     exact
    head bolt offset across the block  58.0      43.00   +34.9%
    pan rail below the crank axis      78.0      15.07   +418%
    block length                      410.0     373.20    +9.9%
    deck width                        152.0     164.40    -7.5%
    deck construction              open deck    closed     wrong
    bore relationship               4 mm slot  siamesed    wrong

The proportions a textbook fixes — saddle, bulkhead, barrel wall, bolt count
— were right to within a few percent.  Everything that is a *layout* decision
was wrong, and two were qualitatively wrong: this block has a closed deck and
siamesed bores, and it does not have a deep skirt at all.

Named design parameters:
  - ``bore_radius``: the bore.  Free.  The barrel outside radius is fixed by
    the core, so opening the bore thins a wall that is already 5.2 mm — on a
    siamesed block that is the whole overbore argument.
  - ``jacket_depth``: floor to deck plate.  Free; the extrusion straddles its
    plane so the floor drops by half of any change.
  - ``main_saddle_radius``: the fitted saddle.  Free.
  - ``bulkhead_thickness``: the main-bearing web.  Free.
  - ``bore_pitch``: driving dimension, and the number the whole engine is laid
    out around — it is the spacing of the bores, the bulkheads, the head bolts
    and the coolant slots at once.  Named rather than free because a pattern's
    spacing is live while its seed is a computed number.
  - ``block_height``, ``deck_plate_thickness``, ``barrel_radius``,
    ``head_bolt_radius``, ``main_bolt_radius``, ``saddle_pad_width``: driving
    dimensions.  ``block_height`` in particular is deliberately *not* free:
    five features are sketched on this extrusion's faces, and while re-running
    the program re-derives all of them, the frozen functional form substitutes
    only into the nodes that hold a parameter — a face-derived plane origin is
    a number by then.  Freeing it would raise the deck and leave the bores.

**Frame.**  The STEP is parked at x 650..1001, y -56..334, z 194..507 in
whatever axes CATIA's part had.  ``REFERENCE`` below records the map from
that frame to this one, which is the ordinary CAD one: the crank axis is the
origin, +x runs along the crank with cylinder 1 at -x, +y is across the block
and +z is up.  ``research/reverse/overlay.py`` reads it to score this scene
against the casting.

**Scale.**  One scene unit is 200 mm, so the whole casting sits inside the
±2.5 box ``tests/zeroset`` and ``tests/backends`` sample.  Every number below
is written in millimetres and multiplied by ``MM``.

**How close it is.**  ``research/reverse/overlay.py`` scores this scene
against the casting on a 3 mm lattice:

    IoU               0.398     ceiling 0.792 at 1 mm, 0.630 at 2 mm
    recall            0.607     of the casting, how much the model has
    precision         0.536     of the model, how much is really there
    casting           3742 cm³
    model             4240 cm³  (+13%)
    surface deviation median 1.7 mm, 79% within 5 mm, 93% within 10 mm

Per section (``research/reverse/sections.py``, mean 0.504, from 0.415 before
the lobes and the flank furniture went in)::

    z = 412  0.732     z = 380  0.480     z = 300  0.447    z = 285  0.690
    z = 250  0.328     z = 200  0.547     y =   5  0.405    y =  48  0.498
    y =  91  0.459     y = 134  0.520     y = 263  0.438

The ceiling is the number that makes the IoU readable.  This casting's mean
wall is 2V/A = 6.6 mm and its surface runs to 11,400 cm², so a lattice IoU is
almost all boundary: displacing the *real part against itself* by one
millimetre already costs it a fifth of its own score.  0.363 is what a model
that sits about 2 mm from the casting's surface scores on a part made of 6 mm
walls — not what a model that has the wrong shape scores.

Where the remaining difference is, and whether it was a choice:

  * **chosen.**  The outer jacket wall.  The casting's is *windowed* — between
    the bulkheads, over z = 90 to 165, it is simply not there, and the jacket
    is open to the air.  This file draws it continuous, which costs about
    60 cm² of added iron in every slice through that band.  A continuous wall
    is what the part would need to hold coolant, and it is the idealisation.
  * **chosen.**  The web taper is drawn as two plates — 20.0 mm at the saddle,
    10.6 mm at the bore bay — rather than a smooth taper, because an extrusion
    cannot taper its own depth.  Drawing one 20 mm web the whole height was the
    single largest error this model had: 160 cm² in every crankcase slice.
  * **chosen.**  The deck outline is drawn at its width over the bores plus
    five lugs and two end pads, which is the scallop the raster found; the
    casting's has more relief than that.
  * **stopped, and worth saying why.**  The crankcase wall is 45 mm thick at
    the pan rail and 5 mm thick fifty-five millimetres higher.  One lofted
    cavity cannot be both, and z = 250 is the worst section in the table
    because of it.  Two stacked cavity lofts would fix it — the same trick the
    bore bank already uses for its two lobe radii — and it is worth about
    40 cm² in one plane, so it stopped paying before it was done.
  * **stopped.**  Raising and widening the flank pads to cover the casting's
    upper fingers was tried and reverted: it cost more at z = 380 (0.462 to
    0.411) than it gained anywhere, because the casting's fingers there are
    discrete and a continuous swelling is not what they are.
  * **not modelled.**  The full-length oil gallery — a Ø20 drilling running
    the whole length at (y = +59, z = 113) — and its nine inclined feeds down
    to the mains, which the probe found as r = 5 cylinders on a
    (0.463, 0, 0.886) axis.  These are most of the *missing* iron on the +y
    flank.
  * **not modelled.**  The cam-drive end wall and its bosses; the ribs hanging
    under the deck plate; freeze plugs; the bellhousing bolt pattern; the
    motor-mount and accessory pads that make the real flanks so busy; draft on
    every wall; and the coolant slots' true kidney shape, drawn here as
    rounded rectangles.

**No simulation study**, for the same reason as before: a block's real load
case is bolt preload plus gas load on a mesh fine enough to resolve a 5 mm
barrel wall, and a study that could not carry that is worse than none.
"""

import math

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import (
    DistanceConstraint,
    FixedConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, loft
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.render import Material
from cadjoint.sdf.boolean import Difference, Union
from cadjoint.sdf.transforms.fields import Mirror
from cadjoint.sdf.transforms.patterns import LinearPattern

# ── the reference, and how this frame relates to it ──────────────────────────
# Read by research/reverse/overlay.py. The file is not in this repository and
# nothing here needs it: the scene builds, renders and tests without it.
#   scene_mm = axes @ (world_mm - origin_mm);  scene_units = scene_mm / unit_mm
# so world +y (along the bore row) becomes scene +x, world +x becomes scene +y,
# and the origin lands on the crank axis under the middle of the bank.
REFERENCE = {
    "step": "/Users/andrinrehnann/Downloads/m15 cylinderblock.stp",
    "unit_mm": 200.0,
    "origin_mm": [826.4, 134.0, 210.03],
    "axes": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
}

# ── scale ────────────────────────────────────────────────────────────────────
MM = 1.0 / 200.0

# ── measured proportions ─────────────────────────────────────────────────────
# Datum: z = 0 is the CRANK AXIS and the main-bearing parting plane; x runs
# along the crank with cylinder 1 at -x; y is across the block. Every number
# here is millimetres, from research/reverse/step_probe.py or from the
# silhouette sweep in research/reverse/sections.py.
DECK_Z = 206.43  # deck face above the crank axis
DECK_PLATE = 10.2  # the closed deck's plate, 406.3..416.5 in the STEP
RAIL_Z = -15.0  # pan rail below the crank axis (measured -15.07)
JACKET_FLOOR_Z = 83.6  # top of the jacket floor web
BORE_BOTTOM_Z = 73.0  # bore opens into the crankcase here
BLOCK_LENGTH = 374.0  # along the crank, over the end flanges

BORE_COUNT = 4
MAIN_COUNT = 5  # a main bearing between and outside every bore
BARREL_RADIUS = 43.0  # bore + 5.25 of wall; at 86 pitch the barrels touch

# The deck's outline is not a rectangle. Measured off the deck raster it runs
# -57 to +57 over the bores, steps out to +77 at each of the five main
# stations, and goes full width at the two ends: a base plate, five lugs down
# one flank, an end pad at each end.
DECK_WIDTH = 114.0
DECK_CENTRE_Y = 0.0
DECK_LUG_LENGTH = 36.0
DECK_LUG_WIDTH = 20.0
DECK_LUG_Y = 63.0
DECK_END_LENGTH = 22.0
DECK_END_WIDTH = 165.0
DECK_END_Y = -5.5

# The flanks. A silhouette sweep — the outermost material at every station, at
# a ladder of heights — gives a block that tapers from ±116 at the pan rail to
# ±52 at the deck, and that is very nearly SYMMETRIC about the bore axis. An
# earlier version of this file had the jacket band 20 mm proud on one flank
# because it had guessed a centre rather than measured one.
#
# Within that envelope the flank is lobed: the outer wall follows the barrels,
# bulging over each cylinder and pinching at the waist between them. It is
# drawn here as four cylinders on the bore stations, blended into a slab that
# carries the waist — a bulge of 52 and a waist of 44, which is the shallow
# end of what the casting does and reads correctly in silhouette.
# The lobe is not one radius: the silhouette sweep gives 63 at the jacket
# floor and 52 under the deck, so the bank is drawn as two stacked bands. A
# single band at either number is visibly wrong in the plan — at 52 the model
# sits inside the casting's waist all the way down the barrel, at 63 it
# stands 11 mm proud of the deck.
LOBE_RADIUS_LOW = 63.0
LOBE_RADIUS_HIGH = 52.0
WAIST_HALF_LOW = 53.0
WAIST_HALF_HIGH = 44.0
BAND_SPLIT_Z = 128.0
UPPER_BOTTOM_Z = 40.0  # see the note on coincident faces below
WALL = 3.0  # outer wall in the jacket band (measured 2.6)

# A continuous rail runs the length of each flank under the deck, and a row of
# raised bolt pads runs down each flank below that. Both are strong silhouette
# features and both are what a photograph of this block shows first.
DECK_RAIL_Y = 57.0  # centre of the rail, either flank
DECK_RAIL_WIDTH = 14.0
DECK_RAIL_TOP_Z = 196.0
DECK_RAIL_DEPTH = 20.0
# Tried larger and higher — 22 mm proud over the whole jacket band — and it
# cost more than it paid: z = 380 fell from 0.462 to 0.411 because the
# casting's fingers there are discrete, not a continuous swelling. Kept at
# the size the raster actually shows.
FLANK_PAD_Y = 59.0
FLANK_PAD_WIDTH = 16.0
FLANK_PAD_LENGTH = 38.0
FLANK_PAD_Z = 120.0
FLANK_PAD_HEIGHT = 40.0
FLANK_BOLT_RADIUS = 5.0

# The oil gallery is an integral tube down one flank: a Ø20 drilling the whole
# length of the block, measured at (y = +59, z = 113), inside a boss.
GALLERY_Y = 59.0
GALLERY_Z = 113.0
GALLERY_BOSS_RADIUS = 16.0
GALLERY_BORE_RADIUS = 10.0

RAIL_WIDTH = 232.0  # across the crankcase at the pan rail
CRANKCASE_TOP_WIDTH = 140.0  # and where it meets the bore bay
CAVITY_FOOT_WIDTH = 200.0  # the pan rails are thick: 45 mm one side, 23 the other
CAVITY_FOOT_Y = 2.0
CAVITY_TOP_WIDTH = 120.0
CAVITY_TOP_Y = 0.0
# NO skirt windows. The casting's crank-throw clearance is the crankcase
# cavity itself — measured, its pan rails are continuous at every station and
# a window bored through them removed 126 cm² of iron the part has. The
# arches between the bulkheads that read as windows in a photograph are what
# the cavity's own loft already makes.

# A Ø90 opening in the drive-end face — the crank seal and its housing. The
# probe found it as an r = 45 cylinder on a Y axis at (825.4, 363), which is
# scene (y = -1, z = 153) at the +x end. It is one of the two or three things
# that identify this face in a photograph.
END_BORE_RADIUS = 45.0
END_BORE_Y = -1.0
END_BORE_Z = 153.0

FLANGE_WIDTH = 349.0  # across the two end flanges
FLANGE_LENGTH = 24.0
FLANGE_TOP_Z = 30.0

HEAD_BOLT_Y = 43.0  # measured; a third closer in than this file once guessed
MAIN_BOLT_Y = 38.0
BULKHEAD_STEP_Z = 32.0  # where the 20 mm saddle web steps down to a 10.6 mm one
UPPER_WEB_THICKNESS = 10.6
WINDOW_Y = -5.0  # the bulkhead's lightening window, Ø28
WINDOW_Z = 55.0
# The deck's coolant slots are kidneys, not rectangles: each sits on a 48 mm
# circle about its own bore, 24.2 degrees either side of the transverse axis,
# which is where the raster found all sixteen of them.
SLOT_RADIUS = 48.0
SLOT_HALF_WIDTH = 5.2
SLOT_HALF_ANGLE = 13.0
SLOT_AZIMUTH = 24.2

CAST_FILLET = 4.0  # smooth-union radius where webs meet walls
CORE_RADIUS = 1.0  # smooth-difference radius where a core leaves a corner
# The machined cuts are SHARP, deliberately. `smooth_max` lifts a field by as
# much as its own k, once per tool; eight tools at a 1.2 mm blend lifted the
# middle of a 10.2 mm deck plate clean outside the solid and the deck
# vanished. A bore and a bolt hole are cut with a tool, not cast, so zero is
# also the right answer.
#
# The same trap has a second face, and it cost the jacket floor: where two
# fields are EQUAL a smooth difference adds its whole k, and along a
# coincident plane they are equal everywhere. The upper block therefore
# reaches down past the crankcase ceiling rather than sitting on it.

# ── design parameters ────────────────────────────────────────────────────────
# The four free ones are handed straight to the feature they dimension, and
# nothing else is derived from any of them — which is the property that makes
# a parameter safe to hand an optimizer here (see the docstring).
bore_radius = Scalar(37.75 * MM, free=True, name="bore_radius")
jacket_depth = Scalar((DECK_Z - DECK_PLATE - JACKET_FLOOR_Z) * MM, free=True, name="jacket_depth")
main_saddle_radius = Scalar(26.9 * MM, free=True, name="main_saddle_radius")
bulkhead_thickness = Scalar(20.0 * MM, free=True, name="bulkhead_thickness")

bore_pitch = Scalar(86.0 * MM, name="bore_pitch")
block_height = Scalar((DECK_Z - RAIL_Z) * MM, name="block_height")
deck_plate_thickness = Scalar(DECK_PLATE * MM, name="deck_plate_thickness")
barrel_radius = Scalar(BARREL_RADIUS * MM, name="barrel_radius")
head_bolt_radius = Scalar(6.0 * MM, name="head_bolt_radius")
main_bolt_radius = Scalar(5.0 * MM, name="main_bolt_radius")
saddle_pad_width = Scalar(204.0 * MM, name="saddle_pad_width")

BORE_PITCH = float(bore_pitch.value) / MM  # 86.0, in millimetres
# Copy 0 of a LinearPattern is its seed and the rest march in +x, so every
# pattern in this file starts at the -x end of its own row.
FIRST_BORE_X = -1.5 * BORE_PITCH  # -129
FIRST_MAIN_X = -2.0 * BORE_PITCH  # -172
JACKET_MID_Z = JACKET_FLOOR_Z + (DECK_Z - DECK_PLATE - JACKET_FLOOR_Z) / 2.0
BAND_MID_Z = (UPPER_BOTTOM_Z + DECK_Z - DECK_PLATE) / 2.0
BAND_HEIGHT = DECK_Z - DECK_PLATE - UPPER_BOTTOM_Z


def rounded(width, height, radius, **kwargs):
    """`PolygonProfile.rounded_rect`, with the degenerate case refused.

    At a corner radius of exactly half the shorter side that side's straight
    run has zero length, so the two arcs meeting there begin and end on the
    same point. The polygon distance divides by every edge length, so one
    repeated vertex is a divide by zero that propagates through the whole
    reduction and leaves the field zero *everywhere* — not merely wrong near
    the repeat. The symptom is silent: no exception, no NaN, just a union that
    swallows the entire scene and a part that renders as a slab. It cost this
    file three separate debugging sessions.

    **This is fixed in cadjoint** — ``rounded_rect`` now drops consecutive
    duplicate vertices before building the profile, so a square with a radius
    of half its side is a clean disc. The fix is not on this branch yet, so
    the guard stays; delete this function and call ``rounded_rect`` directly
    once it lands.
    """
    if radius >= 0.5 * min(width, height) - 1e-9:
        raise ValueError(
            f"a corner radius of {radius} collapses on a {width} x {height} rectangle; "
            "keep it under half the shorter side"
        )
    return PolygonProfile.rounded_rect(width, height, radius, **kwargs)


def pinned(value):
    """A number this file does not want an optimizer to touch."""
    return Scalar(value, free=False)


def placed(x, y, z):
    """A pinned position, in millimetres."""
    return Vector([x * MM, y * MM, z * MM], free=False)


ACROSS = [pinned(0.0), pinned(math.pi / 2.0), pinned(0.0)]  # a cylinder turned onto +x

cast_iron = Material(
    name="cast iron",
    color=[0.36, 0.36, 0.39],
    roughness=0.72,
    metallic=0.55,
    density=7200.0,
    conductivity=52.0,
    specific_heat=460.0,
    youngs_modulus=110e9,
    poisson_ratio=0.26,
    yield_strength=250e6,
    thermal_expansion=11.0e-6,
)
nodular_iron = Material(
    name="nodular iron",
    color=[0.24, 0.25, 0.28],
    roughness=0.55,
    metallic=0.7,
    density=7100.0,
    conductivity=33.0,
    specific_heat=460.0,
    youngs_modulus=169e9,
    poisson_ratio=0.28,
    yield_strength=420e6,
)

# ── 1. the deck plate: the top of a CLOSED deck ──────────────────────────────
# The single most important thing the measurement changed. The casting has a
# 10.2 mm plate over the whole jacket with sixteen slots through it; the
# guessed version of this file had an open deck and therefore the wrong part.
# The plate is modelled first because it carries the deck face, and the bores,
# the head bolts and the slots are all sunk from that face.
deck_profile = rounded(
    BLOCK_LENGTH * MM,
    DECK_WIDTH * MM,
    22.0 * MM,
    center=(0.0, DECK_CENTRE_Y * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (DECK_Z - DECK_PLATE / 2.0) * MM]),
    name="deck plate",
)
deck_plate = extrude(deck_profile, depth=deck_plate_thickness, material=cast_iron)
deck = deck_plate.cap("+")

deck_lug_profile = rounded(
    DECK_LUG_LENGTH * MM,
    DECK_LUG_WIDTH * MM,
    9.0 * MM,
    center=(FIRST_MAIN_X * MM, DECK_LUG_Y * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (DECK_Z - DECK_PLATE / 2.0) * MM]),
    name="deck lug",
)
deck_lugs = LinearPattern(
    extrude(deck_lug_profile, depth=deck_plate_thickness, material=cast_iron),
    direction=[1.0, 0.0, 0.0],
    count=MAIN_COUNT,
    spacing=bore_pitch,
)
deck_end_profile = rounded(
    DECK_END_LENGTH * MM,
    DECK_END_WIDTH * MM,
    9.0 * MM,
    center=(-(BLOCK_LENGTH - DECK_END_LENGTH) / 2.0 * MM, DECK_END_Y * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (DECK_Z - DECK_PLATE / 2.0) * MM]),
    name="deck end pad",
)
deck_end = extrude(deck_end_profile, depth=deck_plate_thickness, material=cast_iron)
deck_ends = Union(deck_end, Mirror(deck_end, "x"), smoothness=0.0)


# ── 2. the bore bank: a LOBED flank, not a slab ──────────────────────────────
# One cylinder per bore, blended into a slab that carries the waist. This is
# what makes the plan view read as a cylinder block rather than as a box: the
# outer wall belled out over each barrel, pinched between them. The blend is a
# real casting radius, and it is 4 mm — the same 4 mm that appears at every
# other cast corner in this file.
def bank_band(lobe_radius, waist_half, low, high, tag):
    """One band of the bore bank: four lobes blended into a waist slab."""
    lobes = LinearPattern(
        Solid.cylinder(
            radius=pinned(lobe_radius * MM),
            height=pinned((high - low) / 2.0 * MM),
            position=placed(FIRST_BORE_X, 0.0, (low + high) / 2.0),
            material=cast_iron,
            name=f"bank_lobe_{tag}",
        ),
        direction=[1.0, 0.0, 0.0],
        count=BORE_COUNT,
        spacing=bore_pitch,
    )
    waist = extrude(
        rounded(
            BLOCK_LENGTH * MM,
            2.0 * waist_half * MM,
            18.0 * MM,
            segments=2,
            plane=SketchPlane(origin=[0.0, 0.0, (low + high) / 2.0 * MM]),
            name=f"bank waist {tag}",
        ),
        depth=(high - low) * MM,
        material=cast_iron,
    )
    return Union(lobes, waist, smoothness=CAST_FILLET * MM)


upper_block = Union(
    bank_band(LOBE_RADIUS_LOW, WAIST_HALF_LOW, UPPER_BOTTOM_Z, BAND_SPLIT_Z + 3.0, "low"),
    bank_band(LOBE_RADIUS_HIGH, WAIST_HALF_HIGH, BAND_SPLIT_Z, DECK_Z - DECK_PLATE, "high"),
    smoothness=CAST_FILLET * MM,
)

# ── 3. the crankcase: a taper from the pan rail up to the bore bay ───────────
# 232 mm across at the rail, 140 where it meets the bank. `deck` faces up, so
# a plane derived from it is placed by pushing down; a loft's profile A sits
# at -height/2 along the plane normal, which here is the lower, wider outline.
crankcase_foot = rounded(
    BLOCK_LENGTH * MM,
    RAIL_WIDTH * MM,
    26.0 * MM,
    center=(0.0, 0.0),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (RAIL_Z + BORE_BOTTOM_Z) / 2.0 * MM]),
    name="crankcase foot",
)
crankcase_top = rounded(
    BLOCK_LENGTH * MM,
    CRANKCASE_TOP_WIDTH * MM,
    20.0 * MM,
    center=(0.0, 0.0),
    segments=2,
    name="crankcase top",
)
crankcase_wall = loft(
    crankcase_foot, crankcase_top, height=(BORE_BOTTOM_Z - RAIL_Z) * MM, material=cast_iron
)

# ── 4. end flanges, flank rails, bolt pads and the oil gallery ───────────────
# 349 mm across, but only at the two ends — the bellhousing face and the drive
# end. 8 mm corners, not 12: `rounded_rect` clamps the radius to half the
# shorter side, and at exactly half it collapses each corner arc onto a single
# point. Two coincident vertices are a zero-length edge, which is a divide by
# zero in the polygon distance — the profile then evaluates to 0 at every
# point in space and the union it feeds swallows the whole scene.
flange_profile = rounded(
    FLANGE_LENGTH * MM,
    FLANGE_WIDTH * MM,
    8.0 * MM,
    center=(-(BLOCK_LENGTH - FLANGE_LENGTH) / 2.0 * MM, -2.0 * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (RAIL_Z + FLANGE_TOP_Z) / 2.0 * MM]),
    name="end flange",
)
end_flange = extrude(flange_profile, depth=(FLANGE_TOP_Z - RAIL_Z) * MM, material=cast_iron)
end_flanges = Union(end_flange, Mirror(end_flange, "x"), smoothness=0.0)

# A continuous rail down each flank under the deck. One profile, mirrored
# across the block's own longitudinal midplane.
deck_rail_profile = rounded(
    BLOCK_LENGTH * MM,
    DECK_RAIL_WIDTH * MM,
    5.0 * MM,
    center=(0.0, -DECK_RAIL_Y * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (DECK_RAIL_TOP_Z - DECK_RAIL_DEPTH / 2.0) * MM]),
    name="deck rail",
)
deck_rail = extrude(deck_rail_profile, depth=DECK_RAIL_DEPTH * MM, material=cast_iron)
deck_rails = Union(deck_rail, Mirror(deck_rail, "y"), smoothness=0.0)

# A raised bolt pad on every main-bearing station, both flanks: one pad,
# mirrored, then walked along the bore pitch.
flank_pad_profile = rounded(
    FLANK_PAD_LENGTH * MM,
    FLANK_PAD_WIDTH * MM,
    6.0 * MM,
    center=(FIRST_MAIN_X * MM, -FLANK_PAD_Y * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, FLANK_PAD_Z * MM]),
    name="flank pad",
)
flank_pad = extrude(flank_pad_profile, depth=FLANK_PAD_HEIGHT * MM, material=cast_iron)
flank_pads = LinearPattern(
    Union(flank_pad, Mirror(flank_pad, "y"), smoothness=0.0),
    direction=[1.0, 0.0, 0.0],
    count=MAIN_COUNT,
    spacing=bore_pitch,
)

# The oil gallery: not a drilling through a wall but an integral tube standing
# proud of the flank, which is how it reads in a photograph. The boss is
# unioned into the casting and the Ø20 gallery is bored through it below.
gallery_boss = Solid.cylinder(
    radius=pinned(GALLERY_BOSS_RADIUS * MM),
    height=pinned((BLOCK_LENGTH + 4.0) / 2.0 * MM),
    position=placed(0.0, GALLERY_Y, GALLERY_Z),
    rotation=ACROSS,
    material=cast_iron,
    name="gallery_boss",
)

blank = Union(
    deck_plate,
    deck_lugs,
    deck_ends,
    upper_block,
    crankcase_wall,
    end_flanges,
    deck_rails,
    flank_pads,
    gallery_boss,
    smoothness=0.0,
)

# ── 5. the barrels: one hole from the deck, patterned four ways ──────────────
# `Face.hole` returns the TOOL rather than a cut, which is what lets this one
# column be subtracted from the water jacket — where it leaves the barrel
# standing — without ever being subtracted from the block. At 86 mm pitch and
# 86 mm over the walls the four barrels touch: this bank is siamesed, and the
# tool's own union is what says so.
barrel = deck.hole(
    barrel_radius,
    depth=(DECK_Z - BORE_BOTTOM_Z) * MM,
    at=(FIRST_BORE_X * MM, 0.0),
    through=12.0 * MM,
)
barrels = LinearPattern(barrel, direction=[1.0, 0.0, 0.0], count=BORE_COUNT, spacing=bore_pitch)

# ── 6. the crankcase cavity ──────────────────────────────────────────────────
# Open below and open to the bores above: measured, the bores end at z = 73
# and nothing hangs below that, so this cavity needs no coring around the
# barrels. Its ceiling is the underside of the jacket floor, and the upper
# block reaches down past it (see the note on coincident faces above).
cavity_foot = rounded(
    (BLOCK_LENGTH - 2 * 12.0) * MM,
    CAVITY_FOOT_WIDTH * MM,
    20.0 * MM,
    center=(0.0, CAVITY_FOOT_Y * MM),
    segments=2,
    plane=SketchPlane(origin=[0.0, 0.0, (RAIL_Z - 5.0 + BORE_BOTTOM_Z) / 2.0 * MM]),
    name="crankcase cavity",
)
cavity_top = rounded(
    (BLOCK_LENGTH - 2 * 12.0) * MM,
    CAVITY_TOP_WIDTH * MM,
    16.0 * MM,
    center=(0.0, CAVITY_TOP_Y * MM),
    segments=2,
    name="crankcase cavity top",
)
crankcase = loft(cavity_foot, cavity_top, height=(BORE_BOTTOM_Z - RAIL_Z + 5.0) * MM)
hollow = Difference(blank, crankcase, smoothness=CORE_RADIUS * MM)

# ── 7. main bearing bulkheads: the one sketch with design freedom ────────────
# Sketch plane normal +X gives in-plane axes u = -Z, v = +Y, so profile x is
# world height measured DOWNWARD and profile y is world y. The pads at profile
# x = 0 sit on the crank axis, which is where this casting's main-cap parting
# face is — measured, there is no block iron at all below z = 0 between the
# rails, so the arch is a half saddle and the caps carry the other half.
bulkhead_pad_near = Vector2(value=[0.0, -114.0 * MM], free=True, name="bulkhead_pad_near")
bulkhead_pad_far = Vector2(value=[0.0, 90.0 * MM], free=True, name="bulkhead_pad_far")
bulkhead_crown_far = Vector2(value=[-32.0 * MM, 80.0 * MM], free=True, name="bulkhead_crown_far")
bulkhead_crown_near = Vector2(
    value=[-32.0 * MM, -100.0 * MM], free=True, name="bulkhead_crown_near"
)
bulkhead_profile = PolygonProfile(
    [bulkhead_pad_near, bulkhead_pad_far, bulkhead_crown_far, bulkhead_crown_near],
    plane=SketchPlane(origin=[FIRST_MAIN_X * MM, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
    name="main bulkhead",
)
bulkhead = extrude(bulkhead_profile, depth=bulkhead_thickness, material=cast_iron)

# The constraints say what the bulkhead IS: a parting pad lying flat on the
# crank axis, a named width across it, and a flat crown above. How far the
# crown pulls in from the pad is left free — the freedom a web really has once
# the saddle is fixed.
FixedConstraint(bulkhead_pad_near, [0.0, -114.0 * MM])
VerticalConstraint(bulkhead_pad_near, bulkhead_pad_far)
DistanceConstraint(bulkhead_pad_near, bulkhead_pad_far, saddle_pad_width)
VerticalConstraint(bulkhead_crown_near, bulkhead_crown_far)

bulkheads = LinearPattern(bulkhead, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch)

# The web does not stay 20 mm the whole way up. Measured along the crank on
# the bore centreline it is 20.0 mm at the saddle and 10.6 mm where it meets
# the bore bay, so the upper half is a second, thinner plate on the same
# stations. Drawing one 20 mm web the full height was the largest single
# error this model had — a hundred and sixty square centimetres in every
# slice through the crankcase.
upper_web_profile = PolygonProfile(
    [
        [-(BULKHEAD_STEP_Z - 2.0) * MM, -86.0 * MM],
        [-(BULKHEAD_STEP_Z - 2.0) * MM, 64.0 * MM],
        [-(BORE_BOTTOM_Z + 3.0) * MM, 56.0 * MM],
        [-(BORE_BOTTOM_Z + 3.0) * MM, -76.0 * MM],
    ],
    plane=SketchPlane(origin=[FIRST_MAIN_X * MM, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
    name="upper web",
    free=False,
)
upper_web = extrude(upper_web_profile, depth=UPPER_WEB_THICKNESS * MM, material=cast_iron)
upper_webs = LinearPattern(
    upper_web, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch
)

# 4 mm against a 20 mm web is a fifth of its thickness — a foundry radius, not
# decoration, in the corner where a bulkhead meets the crankcase wall.
structure = Union(hollow, bulkheads, upper_webs, smoothness=CAST_FILLET * MM / 2.0)

# ── 8. the water jacket: a void defined by what it is not ────────────────────
# The pocket inside the outer wall, minus the barrel bank, minus the ten
# head-bolt columns. Nothing here models a boss: the bosses are the material
# the cut steps around, which is how a core box works. The pocket is lobed for
# the same reason the outside is — it is the same core.
jacket_lobes = LinearPattern(
    Solid.cylinder(
        radius=pinned((LOBE_RADIUS_LOW - WALL) * MM),
        height=pinned((BAND_SPLIT_Z - JACKET_FLOOR_Z) / 2.0 * MM),
        position=placed(FIRST_BORE_X, 0.0, (JACKET_FLOOR_Z + BAND_SPLIT_Z) / 2.0),
        name="jacket_lobe_low",
    ),
    direction=[1.0, 0.0, 0.0],
    count=BORE_COUNT,
    spacing=bore_pitch,
)
jacket_lobes_high = LinearPattern(
    Solid.cylinder(
        radius=pinned((LOBE_RADIUS_HIGH - WALL) * MM),
        height=pinned((DECK_Z - DECK_PLATE - BAND_SPLIT_Z) / 2.0 * MM),
        position=placed(FIRST_BORE_X, 0.0, (BAND_SPLIT_Z + DECK_Z - DECK_PLATE) / 2.0),
        name="jacket_lobe_high",
    ),
    direction=[1.0, 0.0, 0.0],
    count=BORE_COUNT,
    spacing=bore_pitch,
)
jacket_waist_profile = rounded(
    (BLOCK_LENGTH - 2 * 12.0) * MM,
    2.0 * (WAIST_HALF_HIGH - WALL) * MM,
    14.0 * MM,
    segments=2,
    plane=deck.plane(offset=-(DECK_Z - JACKET_MID_Z) * MM),
    name="water jacket",
)
jacket_pocket = Union(
    jacket_lobes,
    jacket_lobes_high,
    extrude(jacket_waist_profile, depth=jacket_depth),
    smoothness=0.0,
)

bolt_column = deck.hole(
    11.0 * MM,
    depth=(DECK_Z - JACKET_FLOOR_Z + 4.0) * MM,
    at=(FIRST_MAIN_X * MM, -HEAD_BOLT_Y * MM),
    through=8.0 * MM,
)
bolt_columns = LinearPattern(
    Union(bolt_column, Mirror(bolt_column, "y"), smoothness=0.0),
    direction=[1.0, 0.0, 0.0],
    count=MAIN_COUNT,
    spacing=bore_pitch,
)
jacket = Difference(jacket_pocket, barrels, bolt_columns, smoothness=0.0)

# ── 9. the cuts ──────────────────────────────────────────────────────────────
# Opening the bore eats into the 5.25 mm the fixed `barrel_radius` core left.
bore = deck.hole(
    bore_radius,
    depth=(DECK_Z - BORE_BOTTOM_Z + 6.0) * MM,
    at=(FIRST_BORE_X * MM, 0.0),
    through=8.0 * MM,
)
bores = LinearPattern(bore, direction=[1.0, 0.0, 0.0], count=BORE_COUNT, spacing=bore_pitch)

# Ten head bolts, on the bulkhead pitch rather than the bore pitch: they land
# on the webs between the bores, which on a siamesed bank is the only iron
# there is to put them in.
head_bolt = deck.hole(
    head_bolt_radius,
    depth=(DECK_Z - JACKET_FLOOR_Z) * MM,
    at=(FIRST_MAIN_X * MM, -HEAD_BOLT_Y * MM),
    through=6.0 * MM,
)
head_bolts = LinearPattern(
    Union(head_bolt, Mirror(head_bolt, "y"), smoothness=0.0),
    direction=[1.0, 0.0, 0.0],
    count=MAIN_COUNT,
    spacing=bore_pitch,
)


def kidney(sign: float) -> list[list[float]]:
    """A kidney slot on the deck, in face coordinates about bore 1.

    An arc band between two radii about the bore's own axis: the shape a
    coolant slot in a deck actually is, and the shape the deck raster found
    sixteen of. ``sign`` picks which side of the transverse axis it sits on.
    """
    centre = math.radians(90.0 - sign * SLOT_AZIMUTH)
    half = math.radians(SLOT_HALF_ANGLE)
    outer = SLOT_RADIUS + SLOT_HALF_WIDTH
    inner = SLOT_RADIUS - SLOT_HALF_WIDTH
    steps = 4
    points = []
    for i in range(steps + 1):
        angle = centre - half + 2.0 * half * i / steps
        points.append([outer * math.cos(angle) * MM, outer * math.sin(angle) * MM])
    for i in range(steps + 1):
        angle = centre + half - 2.0 * half * i / steps
        points.append([inner * math.cos(angle) * MM, inner * math.sin(angle) * MM])
    return points


slot_row = LinearPattern(
    Union(
        deck.pocket(
            kidney(+1.0),
            depth=(DECK_PLATE + 6.0) * MM,
            at=(FIRST_BORE_X * MM, 0.0),
            through=6.0 * MM,
        ),
        deck.pocket(
            kidney(-1.0),
            depth=(DECK_PLATE + 6.0) * MM,
            at=(FIRST_BORE_X * MM, 0.0),
            through=6.0 * MM,
        ),
        smoothness=0.0,
    ),
    direction=[1.0, 0.0, 0.0],
    count=BORE_COUNT,
    spacing=bore_pitch,
)
deck_slots = Union(slot_row, Mirror(slot_row, "y"), smoothness=0.0)

# One line bore opens all five saddles at once, which is how it is cut.
crank_tunnel = Solid.cylinder(
    radius=main_saddle_radius,
    height=pinned((BLOCK_LENGTH + 60.0) / 2.0 * MM),
    position=placed(0.0, 0.0, 0.0),
    rotation=ACROSS,
    name="crank_tunnel",
)

main_bolt = Solid.cylinder(
    radius=main_bolt_radius,
    height=pinned(28.0 * MM),
    position=placed(FIRST_MAIN_X, -MAIN_BOLT_Y, 22.0),
    name="main_bolt",
)
main_bolts = LinearPattern(
    Union(main_bolt, Mirror(main_bolt, "y"), smoothness=0.0),
    direction=[1.0, 0.0, 0.0],
    count=MAIN_COUNT,
    spacing=bore_pitch,
)

# The Ø28 lightening window each web carries above its saddle, measured at
# (y = -5, z = 55). A short cylinder across the block, patterned onto the webs
# rather than bored the length of the casting.
window = Solid.cylinder(
    radius=pinned(14.0 * MM),
    height=pinned(18.0 * MM),
    position=placed(FIRST_MAIN_X, WINDOW_Y, WINDOW_Z),
    rotation=ACROSS,
    name="web_window",
)
windows = LinearPattern(window, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch)

# The gallery itself, bored down the boss unioned on above.
gallery_bore = Solid.cylinder(
    radius=pinned(GALLERY_BORE_RADIUS * MM),
    height=pinned((BLOCK_LENGTH + 40.0) / 2.0 * MM),
    position=placed(0.0, GALLERY_Y, GALLERY_Z),
    rotation=ACROSS,
    name="gallery_bore",
)

# Confined to the end flange. The casting's opening runs from x = 164 to 179
# and the barrel of cylinder 4 reaches 172, so a bore drawn at the measured
# depth eats the fourth barrel wall — in the casting the two clear each other
# in three dimensions, in a straight cylinder they do not. It is drawn 10 mm
# deep into the flange instead, which is the face it has to appear on.
end_bore = Solid.cylinder(
    radius=pinned(END_BORE_RADIUS * MM),
    height=pinned(10.0 * MM),
    position=placed(BLOCK_LENGTH / 2.0 - 1.0, END_BORE_Y, END_BORE_Z),
    rotation=ACROSS,
    name="end_bore",
)

# A bolt into each flank pad, which is what the pads are there for.
flank_bolt = Solid.cylinder(
    radius=pinned(FLANK_BOLT_RADIUS * MM),
    height=pinned(26.0 * MM),
    position=placed(FIRST_MAIN_X, -(FLANK_PAD_Y + 6.0), FLANK_PAD_Z),
    rotation=[pinned(math.pi / 2.0), pinned(0.0), pinned(0.0)],
    name="flank_bolt",
)
flank_bolts = LinearPattern(
    Union(flank_bolt, Mirror(flank_bolt, "y"), smoothness=0.0),
    direction=[1.0, 0.0, 0.0],
    count=MAIN_COUNT,
    spacing=bore_pitch,
)

block = Difference(
    structure,
    jacket,
    bores,
    head_bolts,
    deck_slots,
    main_bolts,
    windows,
    gallery_bore,
    flank_bolts,
    end_bore,
    crank_tunnel,
    smoothness=0.0,
)
block.name = "block"

# ── 10. main bearing caps: rendered, not part of the casting ─────────────────
# Context, and the reason the crank axis reads as an axis in a screenshot.
# Bored by the SAME tunnel the block is — one subtree, emitted once — which is
# what makes cap bore and saddle concentric by construction.
cap_profile = PolygonProfile(
    [
        [2.0 * MM, -62.0 * MM],
        [46.0 * MM, -62.0 * MM],
        [46.0 * MM, 62.0 * MM],
        [2.0 * MM, 62.0 * MM],
    ],
    plane=SketchPlane(origin=[FIRST_MAIN_X * MM, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
    name="main cap",
    free=False,
)
cap = extrude(cap_profile, depth=26.0 * MM, material=nodular_iron)
caps = Difference(
    LinearPattern(cap, direction=[1.0, 0.0, 0.0], count=MAIN_COUNT, spacing=bore_pitch),
    crank_tunnel,
    smoothness=0.0,
)

scene = Union(block, caps, smoothness=0.0)
satisfy_constraints(scene, steps=2)

# ── traceability: the iron volume as a function of the parameters ────────────
# Not an optimization — a proof that the casting is still differentiable end
# to end after a dozen booleans. The measured volume is 3776 cm³; this lattice
# is far too coarse to reproduce that and is not trying to, which is what
# research/reverse/overlay.py is for.
block_parameters, block_fixed, _ = extract_parameters(block)
block_sdf = functionalize(block)

_axes = [
    jnp.linspace(-0.98, 0.98, 25),
    jnp.linspace(-0.92, 0.92, 17),
    jnp.linspace(-0.10, 1.06, 19),
]
volume_cells = jnp.stack(jnp.meshgrid(*_axes, indexing="ij"), axis=-1).reshape(-1, 3)
volume_cell = float((1.96 / 24) * (1.84 / 16) * (1.16 / 18))


def block_volume(parameters):
    """Smoothed iron volume of the casting, differentiable in `parameters`."""
    sdf = block_sdf(parameters, block_fixed)
    return volume_cell * jnp.sum(jax.nn.sigmoid(-sdf(volume_cells) / 0.02))
