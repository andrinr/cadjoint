"""Sphere-tracing renderer.

References:
    https://google-research.github.io/self-organising-systems/2022/jax-raycast/
    https://iquilezles.org/articles/rmshadows/
"""

from jaxcad.render.raymarch.camera import _camera_rays, _normalize
from jaxcad.render.raymarch.env import make_gradient_sky
from jaxcad.render.raymarch.render import (
    _render_image,
    _render_pixel,
    raymarch,
    render_raymarched,
)
from jaxcad.render.raymarch.shade import (
    _cast_shadow,
    _compute_normal,
    _normal_fd,
    _shade_surface,
)
from jaxcad.render.raymarch.trace import (
    TraceMode,
    _bisection_refine,
    _fresnel_schlick,
    _refract,
    _sphere_trace,
    _sphere_trace_with_bracket,
    _trace,
    _trace_through_glass,
)

__all__ = [
    # Public API
    "raymarch",
    "render_raymarched",
    "make_gradient_sky",
    "TraceMode",
    # Semi-public (used by render/functionalize.py and tests)
    "_camera_rays",
    "_render_image",
    "_render_pixel",
    "_sphere_trace",
    "_sphere_trace_with_bracket",
    "_bisection_refine",
    "_trace",
    "_cast_shadow",
    "_normal_fd",
    "_compute_normal",
    "_normalize",
    "_shade_surface",
    "_refract",
    "_fresnel_schlick",
    "_trace_through_glass",
]
