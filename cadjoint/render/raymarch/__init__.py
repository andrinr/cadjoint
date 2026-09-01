"""Forward sphere-tracing renderer.

References:
    https://google-research.github.io/self-organising-systems/2022/jax-raycast/
    https://iquilezles.org/articles/rmshadows/
"""

from cadjoint.render.raymarch.camera import _camera_rays, _normalize
from cadjoint.render.raymarch.env import make_gradient_sky
from cadjoint.render.raymarch.render import (
    _render_image,
    _render_pixel,
    raymarch,
    render_raymarched,
    render_scene,
)
from cadjoint.render.raymarch.shade import (
    _cast_shadow,
    _compute_normal,
    _normal_fd,
    _shade_surface,
)
from cadjoint.render.raymarch.trace import (
    TraceResult,
    _fresnel_schlick,
    _refract,
    _sphere_trace,
    _trace_through_glass,
)

__all__ = [
    # Public API
    "raymarch",
    "render_scene",
    "render_raymarched",
    "make_gradient_sky",
    "TraceResult",
    # Internal building blocks used by tests and advanced integrations
    "_camera_rays",
    "_render_image",
    "_render_pixel",
    "_sphere_trace",
    "_cast_shadow",
    "_normal_fd",
    "_compute_normal",
    "_normalize",
    "_shade_surface",
    "_refract",
    "_fresnel_schlick",
    "_trace_through_glass",
]
