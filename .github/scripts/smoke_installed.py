"""Smoke-test an installed cadjoint wheel: ``python -I smoke_installed.py VERSION``.

Run outside the checkout. Checks the import resolves to site-packages at the
expected version, that a gradient flows through a scene, and that the
package data the wheel must carry is there.
"""

import sys
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp

import cadjoint
from cadjoint import extract_parameters, functionalize
from cadjoint.construction import PolygonProfile, SketchPlane, extrude
from cadjoint.geometry import Scalar, Vector2
from cadjoint.plugins.registry import builtin_specs
from cadjoint.viewer._http import STATIC_ROOT

assert "site-packages" in Path(cadjoint.__file__).parts, cadjoint.__file__
assert version("cadjoint") == sys.argv[1], (version("cadjoint"), sys.argv[1])

corners = [
    Vector2(xy, free=True, name=f"v{i}") for i, xy in enumerate([[0, 0], [1, 0], [1, 1], [0, 1]])
]
depth = Scalar(0.5, free=True, name="depth")
block = extrude(PolygonProfile(corners, plane=SketchPlane()), depth=depth)
free, fixed, _ = extract_parameters(block)
grads = jax.grad(lambda p: functionalize(block)(p, fixed)(jnp.array([0.5, 0.5, 2.0])))(free)
assert jnp.isfinite(grads["depth"]) and grads["depth"] != 0.0, grads

assert (STATIC_ROOT / "index.html").is_file(), "playground build missing"
missing = [name for name, spec in builtin_specs().items() if spec.version is None]
assert not missing, f"tesseract_config.yaml missing for {missing}"
print(f"cadjoint {version('cadjoint')}: ok")
