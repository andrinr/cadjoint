"""Parametric heat sink: finned extrusion, copper slug, press-fit bushings.

A compact power-module heat sink and the default tour of the toolchain: the
fin comb is one parameter-backed sketch profile extruded through a named
depth, the copper heat slug under the die is a revolved section, and two
steel bushings carry the mounting screws down to a green FR4 board that
carries the die and its drive electronics — context geometry the physics
never sees. A named SimMesh discretizes the
sink, the declared thermal study conducts the die's heat flux up into the
fins on it, and the single declared optimization at the bottom descends that
SAME simulation — peak temperature against a material-volume penalty —
differentiably, straight through the geometry the viewport renders, with
the mesher and the solver each crossing a Tesseract boundary.

Named design parameters:
  - ``fin_depth``: extrusion depth of the fin comb (along y)
  - ``base_width``: driving dimension across the base deck
  - ``bushing_spacing``: distance between the two mounting bushings

The comb sketch keeps twelve meaningful design freedoms under its
constraints — fin depth, and per fin its tip height, root width, and
tip width (taper and lean emerge; side walls carry no verticality),
plus symmetric outer-fin spacing and deck thickness; everything else is
a relation (horizontal/mirror) or pinned (base span, the slug's die
interface).
"""

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import (
    DistanceConstraint,
    EqualLengthConstraint,
    FixedConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    satisfy_constraints,
)
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, revolve
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh, ThermalStudy
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.optimize import Optimization
from cadjoint.render import Material
from cadjoint.sdf import Box, Cylinder, Translate
from cadjoint.sdf.boolean import Union

# ── design parameters ────────────────────────────────────────────────────────
# fin_depth is genuinely free (the optimizer moves it); the named scalars
# are driving dimensions: constraints hold the sketch to them, so editing a
# value here re-dimensions the part without freeing it to drift.
fin_depth = Scalar(1.2, free=True, name="fin_depth")
base_width = Scalar(1.8, name="base_width")
bushing_spacing = Scalar(1.56, name="bushing_spacing")

# Materials carry their physics as well as their look (SI: kg/m³, W/(m·K),
# J/(kg·K), Pa, 1/K). The scene below is drawn at unit scale rather than in
# metres, so the thermal study states its conductivity explicitly instead of
# taking it from the materials; the properties still feed mass, elastic
# studies and the safety factor, and become the default the moment a scene
# is authored in metres.
aluminum = Material(
    name="aluminum",
    color=[0.8, 0.82, 0.85],
    roughness=0.3,
    metallic=0.9,
    density=2700.0,
    conductivity=167.0,
    specific_heat=896.0,
    youngs_modulus=68.9e9,
    poisson_ratio=0.33,
    thermal_expansion=23.6e-6,
    yield_strength=276e6,
)
copper = Material(
    name="copper",
    color=[0.9, 0.45, 0.22],
    roughness=0.18,
    metallic=0.95,
    density=8940.0,
    conductivity=391.0,
    specific_heat=385.0,
    youngs_modulus=117e9,
    poisson_ratio=0.34,
    thermal_expansion=17.0e-6,
    yield_strength=69e6,
)
steel = Material(
    name="steel",
    color=[0.55, 0.57, 0.6],
    roughness=0.4,
    metallic=0.85,
    density=7870.0,
    conductivity=51.9,
    specific_heat=486.0,
    youngs_modulus=205e9,
    poisson_ratio=0.29,
    thermal_expansion=11.5e-6,
    yield_strength=370e6,
)
# Board-level context: rendered for orientation, excluded from the thermal
# domain below (see ``thermal_body``), so the physics never sees them.
fr4 = Material(name="fr4", color=[0.10, 0.36, 0.22], roughness=0.85, metallic=0.0)
silicon = Material(name="silicon", color=[0.07, 0.08, 0.10], roughness=0.15, metallic=0.3)
black_oxide = Material(name="black oxide", color=[0.11, 0.11, 0.12], roughness=0.45, metallic=0.85)
electrolytic = Material(
    name="electrolytic", color=[0.10, 0.14, 0.32], roughness=0.55, metallic=0.05
)

# ── fin comb: base deck + three fins as one sketch profile ───────────────────
# Sketch plane normal +Y gives in-plane axes u = -X, v = +Z: profile y is
# world height. Extrusion spans ±fin_depth/2 around y = 0. Every vertex is a
# live sketch point — drag a fin tip in the viewport and rerun.
# The initial comb is DELIBERATELY overbuilt — brick-thick fins wasting
# material — so running cool-sink visibly slims and tapers it into a
# well-proportioned sink instead of nudging an already-decent one.
base_l = Vector2(value=[-0.9, 0.0], free=True, name="base_l")
base_r = Vector2(value=[0.9, 0.0], free=True, name="base_r")
deck_r = Vector2(value=[0.9, 0.18], free=True, name="deck_r")
fin1_root_r = Vector2(value=[0.75, 0.18], free=True, name="fin1_root_r")
fin1_tip_r = Vector2(value=[0.75, 0.85], free=True, name="fin1_tip_r")
fin1_tip_l = Vector2(value=[0.45, 0.85], free=True, name="fin1_tip_l")
fin1_root_l = Vector2(value=[0.45, 0.18], free=True, name="fin1_root_l")
fin2_root_r = Vector2(value=[0.15, 0.18], free=True, name="fin2_root_r")
fin2_tip_r = Vector2(value=[0.15, 0.85], free=True, name="fin2_tip_r")
fin2_tip_l = Vector2(value=[-0.15, 0.85], free=True, name="fin2_tip_l")
fin2_root_l = Vector2(value=[-0.15, 0.18], free=True, name="fin2_root_l")
fin3_root_r = Vector2(value=[-0.45, 0.18], free=True, name="fin3_root_r")
fin3_tip_r = Vector2(value=[-0.45, 0.85], free=True, name="fin3_tip_r")
fin3_tip_l = Vector2(value=[-0.75, 0.85], free=True, name="fin3_tip_l")
fin3_root_l = Vector2(value=[-0.75, 0.18], free=True, name="fin3_root_l")
deck_l = Vector2(value=[-0.9, 0.18], free=True, name="deck_l")
comb_profile = PolygonProfile(
    [
        base_l,
        base_r,
        deck_r,
        fin1_root_r,
        fin1_tip_r,
        fin1_tip_l,
        fin1_root_l,
        fin2_root_r,
        fin2_tip_r,
        fin2_tip_l,
        fin2_root_l,
        fin3_root_r,
        fin3_tip_r,
        fin3_tip_l,
        fin3_root_l,
        deck_l,
    ],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="fin comb",
)
sink = extrude(comb_profile, depth=fin_depth, material=aluminum)

# The comb is a production-style constrained sketch that still keeps real
# design freedom. Relations — horizontal/vertical squaring, symmetric
# margins about the base — shape HOW the comb may move; the pinned base
# span anchors it. Each fin keeps its OWN tip height and root width (the
# optimizer sizes all three individually), and fin side walls carry no
# verticality constraint — root and tip widths move independently, so
# TAPER and lean are free to emerge per fin, alongside deck thickness,
# symmetric outer-fin spacing, and fin_depth.
# Every descent step is projected back onto this system (see
# cadjoint.optimize).
FixedConstraint(base_l, [-0.9, 0.0])
DistanceConstraint(base_l, base_r, base_width)
HorizontalConstraint(base_l, base_r)
VerticalConstraint(base_r, deck_r)
VerticalConstraint(base_l, deck_l)
HorizontalConstraint(deck_l, deck_r)
HorizontalConstraint(fin1_root_r, deck_r)
HorizontalConstraint(fin1_root_l, deck_r)
HorizontalConstraint(fin2_root_r, deck_r)
HorizontalConstraint(fin2_root_l, deck_r)
HorizontalConstraint(fin3_root_r, deck_r)
HorizontalConstraint(fin3_root_l, deck_r)
HorizontalConstraint(fin1_tip_r, fin1_tip_l)
HorizontalConstraint(fin2_tip_r, fin2_tip_l)
HorizontalConstraint(fin3_tip_r, fin3_tip_l)
EqualLengthConstraint(base_l, fin3_root_l, base_r, fin1_root_r)
EqualLengthConstraint(base_l, fin2_root_l, base_r, fin2_root_r)

# ── copper heat slug: revolved section under the die, screw bore on axis ─────
# Revolve spins the profile around the plane's local Y axis (world z here):
# profile x is radius, profile y runs along the axis. The slug presses into
# the deck from below; the die contacts its bottom face.
#
# Design rule: boundary-condition regions sit on pinned geometry. The slug
# bottom carries the die's heat-flux BC, so its outline is NOT free — the
# optimizer shapes fins and depth, never the chip interface. (The fin-top
# Dirichlet region below is anchored generously instead: its threshold sits
# far under the tips, so the height freedom cannot move the fins out of it.)
slug_bore_low = Vector2(value=[0.05, -0.18], name="slug_bore_low")
slug_rim_low = Vector2(value=[0.26, -0.18], name="slug_rim_low")
slug_rim_high = Vector2(value=[0.26, 0.04], name="slug_rim_high")
slug_bore_high = Vector2(value=[0.05, 0.04], name="slug_bore_high")
slug_profile = PolygonProfile(
    [slug_bore_low, slug_rim_low, slug_rim_high, slug_bore_high],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="slug section",
)
slug = revolve(slug_profile, material=copper)

# ── mounting bushings: fixed pattern, spacing tied by a constraint ───────────
# Standard press-fit parts: their radius/height are pinned Scalars (not
# free), so optimization never resizes catalog hardware.
bushing_a = Vector([0.78, 0.0, 0.1], free=True, name="bushing_a")
bushing_b = Vector([-0.78, 0.0, 0.1], free=True, name="bushing_b")
FixedConstraint(bushing_a, [0.78, 0.0, 0.1])
DistanceConstraint(bushing_a, bushing_b, bushing_spacing)
bush_a = Solid.cylinder(
    radius=Scalar(0.07), height=Scalar(0.12), position=bushing_a, material=steel, name="bush_a"
)
bush_b = Solid.cylinder(
    radius=Scalar(0.07), height=Scalar(0.12), position=bushing_b, material=steel, name="bush_b"
)

# The thermal body is what the study meshes and the optimizer moves: the
# sink, the slug pressed into it, and the two bushings, blended at 0.03 so
# the press-fit seams read as fillets rather than cracks.
thermal_body = Union(sink, slug, bush_a, bush_b, smoothness=0.03)

# ── board-level context: the module the sink is bolted to ───────────────────
# Rendered so the part reads as a power module rather than a lone comb, and
# kept OUT of the simulation via ``domain=thermal_body`` on the mesh below:
# the flux enters through the slug bottom exactly as before, and every
# mesh, solve and gradient is identical to the thermal body alone. Plain
# primitives (no construction mirror) — context, not design intent.
board = Translate(Box(size=[1.2, 0.78, 0.015], material=fr4), [0.0, 0.0, -0.245])
die = Translate(Box(size=[0.17, 0.17, 0.025], material=silicon), [0.0, 0.0, -0.205])
head_a = Translate(Cylinder(0.062, 0.03, material=black_oxide), [0.78, 0.0, 0.25])
head_b = Translate(Cylinder(0.062, 0.03, material=black_oxide), [-0.78, 0.0, 0.25])
cap_a = Translate(Cylinder(0.07, 0.09, material=electrolytic), [1.05, 0.38, -0.14])
cap_b = Translate(Cylinder(0.07, 0.09, material=electrolytic), [1.05, -0.38, -0.14])

# A 5 mm blend rather than a hard union: invisible at this scale, and the
# feature-edge extractor's Newton steps converge in a third of the time on
# a smooth field (measured: 32.7 s hard, 8.8 s at 0.005).
scene = Union(thermal_body, board, die, head_a, head_b, cap_a, cap_b, smoothness=0.005)
satisfy_constraints(scene, steps=2)

# ── simulation mesh: the sink volume on a named grid ─────────────────────────
# First-class meshing intent: the study below solves on it, the viewer
# inspects it (counts, bounds, element quality), and the optimization
# refreezes it as the design moves. method="tet10" is the quality path —
# boundary-conforming quadratic tets from the dual-contoured surface
# (method="hex" is the fast voxelize+snap alternative).
sink_mesh = SimMesh(
    name="sink-mesh",
    resolution=(18, 13, 11),
    bounds=(-1.05, -0.8, -0.3),
    size=(2.1, 1.6, 1.4),
    method="tet10",
    domain=thermal_body,
)

# ── thermal study: die flux on the slug bottom, ambient at the fin field ─────
# Node selections are programmatic: the flux enters through the boundary
# faces of the slug's bottom disc; the upper fin field is held at ambient
# (an idealized convection sink).
heat_study = ThermalStudy(
    name="sink-conduction",
    conductivity=2.0,
    bcs=[
        HeatFlux(
            Nodes.halfspace([0.0, 0.0, -0.12], [0.0, 0.0, -1.0])
            & Nodes.sphere([0.0, 0.0, -0.18], 0.4),
            6.0,
        ),
        Dirichlet(Nodes.halfspace([0.0, 0.0, 0.45], [0.0, 0.0, 1.0]), 0.0),
    ],
    mesh=sink_mesh,
)

# The regularizer is a real reverse-mode derivative path through sketch
# points -> extrusion -> final SDF evaluation: the (smoothed) aluminum
# volume of the fin comb as a function of the free parameters above.
sink_parameters, sink_fixed, _ = extract_parameters(sink)
sink_sdf = functionalize(sink)

axes = [jnp.linspace(-1.0, 1.0, 15), jnp.linspace(-0.7, 0.7, 15), jnp.linspace(-0.05, 0.95, 15)]
cells = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
cell_volume = float((2.0 / 14) * (1.4 / 14) * (1.0 / 14))


def material_volume(parameters):
    sdf = sink_sdf(parameters, sink_fixed)
    return cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(cells) / 0.03))


# The declared optimization closes the loop end to end: the thermal study
# above becomes the objective. Per step, the frozen simulation mesh follows
# the design differentiably (node positions re-projected through the traced
# SDF), the study solves on it, and the PEAK temperature descends against
# the material-volume regularizer while every update projects back onto the
# sketch constraints — run it from the viewer and the optimized part
# arrives with its temperature field attached, values written back here.
# (Peak, not mean: the die is the hot spot, and a mean-temperature objective
# degenerately rewards deleting hot material instead of cooling the chip.)
cool_sink = Optimization(
    name="cool-sink",
    study="sink-conduction",
    metric="max",
    regularizer=material_volume,
    regularizer_weight=0.4,
    steps=12,
    learning_rate=0.004,
    # Run the loop through two Tesseracts: the tetfill mesher (TetGen behind
    # an exact pass-through VJP) feeding the jax-fem solver tesseract.  The
    # dual contouring upstream stays differentiable in JAX against the true
    # SDF, which is what keeps the fin creases sharp.  "direct" runs the
    # same objective fully in-process.
    gradient_path="tesseract-dc",
)
