"""jaxCAD: Differentiable CAD with SDFs and CSG."""

from jaxcad.constraints.solve import solve_constraints
from jaxcad.extraction import extract_parameters
from jaxcad.functionalize import functionalize, functionalize_scene
from jaxcad.parametrization import (
    compute_param_scales,
    from_normalized,
    normalize,
    to_constrained,
    to_normalized,
    to_unconstrained,
    unnormalize,
)
from jaxcad.render.functionalize import functionalize_render
from jaxcad.render.material import Material
from jaxcad.sdf import SDF, boolean, primitives, transforms

__all__ = [
    "SDF",
    "Material",
    "primitives",
    "boolean",
    "transforms",
    "extract_parameters",
    "functionalize",
    "functionalize_scene",
    "functionalize_render",
    "solve_constraints",
    "to_unconstrained",
    "to_constrained",
    "compute_param_scales",
    "normalize",
    "unnormalize",
    "to_normalized",
    "from_normalized",
]
