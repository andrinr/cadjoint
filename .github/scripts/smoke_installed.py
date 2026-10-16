"""Smoke-test an installed cadjoint distribution, not the checkout.

Run from outside the repository against a venv holding only the built wheel:

    python smoke_installed.py 0.1.0

Checks that the import resolves to site-packages, that the version matches
the expected one, that the README's Python example runs and differentiates,
and that the package data the wheel must carry (the playground build and
the Tesseract configs) is present.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

import cadjoint
from cadjoint import extract_parameters, functionalize
from cadjoint.constraints import DistanceConstraint, satisfy_constraints
from cadjoint.construction import PolygonProfile, SketchPlane, extrude
from cadjoint.geometry import Scalar, Vector2
from cadjoint.meshing import GridSpec, extract_mesh
from cadjoint.plugins.registry import BUILTIN_PACKAGES, builtin_specs
from cadjoint.viewer._http import STATIC_ROOT

location = Path(cadjoint.__file__).resolve()
assert "site-packages" in location.parts, f"imported from a checkout: {location}"
if len(sys.argv) > 1:
    assert cadjoint.__version__ == sys.argv[1], (cadjoint.__version__, sys.argv[1])

a = Vector2([0.0, 0.0], free=True, name="a")
b = Vector2([1.0, 0.0], free=True, name="b")
c = Vector2([1.0, 1.0], free=True, name="c")
d = Vector2([0.0, 1.0], free=True, name="d")
DistanceConstraint(a, b, 1.2)
depth = Scalar(0.5, free=True, name="depth")
block = extrude(PolygonProfile([a, b, c, d], plane=SketchPlane()), depth=depth)
satisfy_constraints(block)

grid = GridSpec.from_bounds((-0.5, -0.5, -0.5), (2.0, 2.0, 1.5), 32)
mesh = extract_mesh(block, grid)

free, fixed, _ = extract_parameters(block)
sdf_fn = functionalize(block)
point = jnp.array([0.5, 0.5, 2.0])
grads = jax.grad(lambda p: sdf_fn(p, fixed)(point))(free)
assert jnp.isfinite(grads["depth"]) and grads["depth"] != 0.0, grads

assert (STATIC_ROOT / "index.html").is_file(), STATIC_ROOT
specs = builtin_specs()
assert set(specs) == set(BUILTIN_PACKAGES), sorted(specs)
missing = [name for name, spec in specs.items() if spec.version is None]
assert not missing, f"tesseract_config.yaml missing for {missing}"

print(f"cadjoint {cadjoint.__version__} from {location.parent}: ok")
