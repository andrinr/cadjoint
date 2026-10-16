"""Reference values for the jax-js spike, from Python cadjoint.

Builds the text-to-cad space-frame node (six tubes, smooth union k=4,
Shell 2, cut by a 40 mm ball), evaluates it at 2000 fixed points, and
takes a central difference of sum(field) with respect to the hub radius.
"""
import json
import math
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from cadjoint.construction import Solid
from cadjoint.sdf import Rotate, Shell
from cadjoint.sdf.boolean import Intersection, Union

DIRECTIONS = [(0, 0), (115, 0), (115, 120), (115, 240), (70, 60), (70, 300)]


def node(hub_radius=15.0, blend=4.0, wall=2.0):
    hub = Solid.sphere(radius=hub_radius, name="hub")
    tubes = []
    for i, (tilt, turn) in enumerate(DIRECTIONS):
        tube = Solid.cylinder(radius=10.0, height=22.0, position=(0.0, 0.0, 22.0), name=f"tube_{i}")
        tubes.append(Rotate(Rotate(tube, "y", math.radians(tilt)), "z", math.radians(turn)))
    solid = Union(hub, *tubes, smoothness=blend)
    return Intersection(Shell(solid, wall), Solid.sphere(radius=40.0, name="reach"), smoothness=0.0)


rng = np.random.default_rng(0)
points = rng.uniform(-45, 45, size=(2000, 3)).astype(np.float32)
p64 = jnp.asarray(points, dtype=jnp.float64)
values = np.asarray(node()(p64), dtype=np.float64)
h = 1e-4
d_hub = (float(np.asarray(node(15.0 + h)(p64)).sum())
         - float(np.asarray(node(15.0 - h)(p64)).sum())) / (2 * h)
json.dump({"points": points.ravel().tolist(), "values": values.tolist(),
           "d_sum_d_hub_radius": d_hub}, open(sys.argv[1], "w"))
print("values", values.min(), values.max(), "d/d hub", d_hub)
