"""Parametric heat sink: finned extrusion, copper slug, press-fit bushings.

A compact power-module heat sink and the default tour of the toolchain: the
fin comb is one parameter-backed sketch profile extruded through a named
depth, the copper heat slug under the die is a revolved section, and two
steel bushings carry the mounting screws. A declared thermal study conducts
the die's heat flux up into the fins, and the block at the bottom takes a
real engineering gradient — material volume w.r.t. the named dimensions —
straight through the same geometry the viewport renders.

Named design parameters:
  - ``fin_depth``: extrusion depth of the fin comb (along y)
  - ``fin_height``: driving dimension from a fin root to its tip
  - ``bushing_spacing``: distance between the two mounting bushings
"""

import jax
import jax.numpy as jnp

from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import DistanceConstraint, FixedConstraint, satisfy_constraints
from cadjoint.construction import PolygonProfile, SketchPlane, Solid, extrude, revolve
from cadjoint.fem import Dirichlet, HeatFlux, Nodes, ThermalStudy
from cadjoint.geometry import Scalar, Vector, Vector2
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union

# ── design parameters ────────────────────────────────────────────────────────
fin_depth = Scalar(1.2, free=True, name="fin_depth")
fin_height = Scalar(0.67, name="fin_height")
bushing_spacing = Scalar(1.56, name="bushing_spacing")

aluminum = Material(name="aluminum", color=[0.8, 0.82, 0.85], roughness=0.3, metallic=0.9)
copper = Material(name="copper", color=[0.9, 0.45, 0.22], roughness=0.18, metallic=0.95)
steel = Material(name="steel", color=[0.55, 0.57, 0.6], roughness=0.4, metallic=0.85)

# ── fin comb: base deck + three fins as one sketch profile ───────────────────
# Sketch plane normal +Y gives in-plane axes u = -X, v = +Z: profile y is
# world height. Extrusion spans ±fin_depth/2 around y = 0. Every vertex is a
# live sketch point — drag a fin tip in the viewport and rerun.
base_l = Vector2(value=[-0.9, 0.0], free=True, name="base_l")
base_r = Vector2(value=[0.9, 0.0], free=True, name="base_r")
deck_r = Vector2(value=[0.9, 0.18], free=True, name="deck_r")
fin1_root_r = Vector2(value=[0.68, 0.18], free=True, name="fin1_root_r")
fin1_tip_r = Vector2(value=[0.68, 0.85], free=True, name="fin1_tip_r")
fin1_tip_l = Vector2(value=[0.52, 0.85], free=True, name="fin1_tip_l")
fin1_root_l = Vector2(value=[0.52, 0.18], free=True, name="fin1_root_l")
fin2_root_r = Vector2(value=[0.08, 0.18], free=True, name="fin2_root_r")
fin2_tip_r = Vector2(value=[0.08, 0.85], free=True, name="fin2_tip_r")
fin2_tip_l = Vector2(value=[-0.08, 0.85], free=True, name="fin2_tip_l")
fin2_root_l = Vector2(value=[-0.08, 0.18], free=True, name="fin2_root_l")
fin3_root_r = Vector2(value=[-0.52, 0.18], free=True, name="fin3_root_r")
fin3_tip_r = Vector2(value=[-0.52, 0.85], free=True, name="fin3_tip_r")
fin3_tip_l = Vector2(value=[-0.68, 0.85], free=True, name="fin3_tip_l")
fin3_root_l = Vector2(value=[-0.68, 0.18], free=True, name="fin3_root_l")
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

# The fin height is a named dimension: the center fin's tip must sit
# fin_height above its root. Constraints own this part of the sketch —
# move a point, then Satisfy projects it back onto the system.
FixedConstraint(base_l, [-0.9, 0.0])
DistanceConstraint(fin2_root_r, fin2_tip_r, fin_height)

# ── copper heat slug: revolved section under the die, screw bore on axis ─────
# Revolve spins the profile around the plane's local Y axis (world z here):
# profile x is radius, profile y runs along the axis. The slug presses into
# the deck from below; the die contacts its bottom face.
slug_bore_low = Vector2(value=[0.05, -0.18], free=True, name="slug_bore_low")
slug_rim_low = Vector2(value=[0.26, -0.18], free=True, name="slug_rim_low")
slug_rim_high = Vector2(value=[0.26, 0.04], free=True, name="slug_rim_high")
slug_bore_high = Vector2(value=[0.05, 0.04], free=True, name="slug_bore_high")
slug_profile = PolygonProfile(
    [slug_bore_low, slug_rim_low, slug_rim_high, slug_bore_high],
    plane=SketchPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0]),
    name="slug section",
)
slug = revolve(slug_profile, material=copper)

# ── mounting bushings: fixed pattern, spacing tied by a constraint ───────────
bushing_a = Vector([0.78, 0.0, 0.1], free=True, name="bushing_a")
bushing_b = Vector([-0.78, 0.0, 0.1], free=True, name="bushing_b")
FixedConstraint(bushing_a, [0.78, 0.0, 0.1])
DistanceConstraint(bushing_a, bushing_b, bushing_spacing)
bush_a = Solid.cylinder(radius=0.07, height=0.12, position=bushing_a, material=steel, name="bush_a")
bush_b = Solid.cylinder(radius=0.07, height=0.12, position=bushing_b, material=steel, name="bush_b")

scene = Union(sink, slug, bush_a, bush_b, smoothness=0.03)
satisfy_constraints(scene, steps=2)

# ── thermal study: die flux on the slug bottom, ambient at the fin field ─────
# Node selections are programmatic: the flux enters through the boundary
# faces of the slug's bottom disc; the upper fin field is held at ambient
# (an idealized convection sink).
heat_study = ThermalStudy(
    name="sink-conduction",
    resolution=(20, 14, 12),
    conductivity=2.0,
    bcs=[
        HeatFlux(
            Nodes.halfspace([0.0, 0.0, -0.12], [0.0, 0.0, -1.0])
            & Nodes.sphere([0.0, 0.0, -0.18], 0.4),
            6.0,
        ),
        Dirichlet(Nodes.halfspace([0.0, 0.0, 0.6], [0.0, 0.0, 1.0]), 0.0),
    ],
    bounds=(-1.05, -0.8, -0.3),
    size=(2.1, 1.6, 1.4),
)

# This is a real reverse-mode derivative through sketch points -> extrusion ->
# final SDF evaluation: the aluminum volume of the fin comb, and how it moves
# with the extrusion depth and the center fin's tip. Drag a vertex or edit
# fin_depth and rerun: the sensitivities update in the AD panel above.
sink_parameters, sink_fixed, _ = extract_parameters(sink)
sink_sdf = functionalize(sink)

axes = [jnp.linspace(-1.0, 1.0, 15), jnp.linspace(-0.7, 0.7, 15), jnp.linspace(-0.05, 0.95, 15)]
cells = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
cell_volume = float((2.0 / 14) * (1.4 / 14) * (1.0 / 14))


def material_volume(parameters):
    sdf = sink_sdf(parameters, sink_fixed)
    return cell_volume * jnp.sum(jax.nn.sigmoid(-sdf(cells) / 0.03))


volume, volume_gradient = jax.value_and_grad(material_volume)(sink_parameters)
differentiability_demo = {
    "pipeline": "Profile -> Extrude -> SDF",
    "metric": "aluminum volume (smoothed)",
    "value": float(volume),
    "parameter_count": len(sink_parameters),
    "sensitivities": [
        {"parameter": "fin_depth", "value": float(volume_gradient["fin_depth"])},
        {"parameter": "fin2_tip_l.y", "value": float(volume_gradient["fin2_tip_l"][1])},
    ],
}
