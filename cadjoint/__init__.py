"""cadjoint: Differentiable CAD with SDFs and CSG."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from cadjoint.constraints.solve import solve_constraints
from cadjoint.extraction import apply_parameters, extract_parameters
from cadjoint.functionalize import functionalize, functionalize_scene
from cadjoint.parametrization import (
    compute_param_scales,
    from_normalized,
    normalize,
    to_constrained,
    to_normalized,
    to_unconstrained,
    unnormalize,
)
from cadjoint.render.material import Material
from cadjoint.sdf import SDF, boolean, primitives, transforms

# The version lives in pyproject.toml alone; an uninstalled checkout has none.
try:
    __version__ = _version("cadjoint")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "SDF",
    "Material",
    "primitives",
    "boolean",
    "transforms",
    "extract_parameters",
    "apply_parameters",
    "functionalize",
    "functionalize_scene",
    "solve_constraints",
    "to_unconstrained",
    "to_constrained",
    "compute_param_scales",
    "normalize",
    "unnormalize",
    "to_normalized",
    "from_normalized",
]
