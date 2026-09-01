"""Camera basis and primary-ray generation shared by the viewer shaders.

The preview and path-tracing templates previously each carried their own copy of
this code, which is exactly the kind of duplication that drifts: the overlay
projection on the host side has to agree with it pixel for pixel, so there needs
to be one definition.

Keep in sync with ``frontend/src/viewer/math.ts``, which reimplements the same
formulas on the CPU for picking and overlay placement.
"""

from __future__ import annotations

CAMERA_MARKER = "__CADJOINT_CAMERA__"

CAMERA_WGSL = r"""
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  return v * inverseSqrt(max(dot(v, v), 1e-12));
}

struct CameraBasis {
  forward : vec3<f32>,
  right   : vec3<f32>,
  up      : vec3<f32>,
};

// Orthonormal camera frame. World up is +Y, except when the view direction is
// almost parallel to it — looking straight down is exactly what the Top preset
// asks for, and cross(forward, +Y) is degenerate there.
fn camera_basis(camera: vec3<f32>, look_at: vec3<f32>) -> CameraBasis {
  var basis: CameraBasis;
  basis.forward = safe_normalize(look_at - camera);
  var reference = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(basis.forward.y) > 0.999) {
    reference = vec3<f32>(0.0, 0.0, 1.0);
  }
  basis.right = safe_normalize(cross(basis.forward, reference));
  basis.up = cross(basis.right, basis.forward);
  return basis;
}

struct PrimaryRay {
  origin    : vec3<f32>,
  direction : vec3<f32>,
};

// Ray through a pixel, for either projection.
//
// `uv` is the screen coordinate in the viewer's convention:
//   uv = (frag.xy / resolution - 0.5) * vec2(aspect, -1)
//
// Perspective rays fan out from the camera with a half-height of
// FOV_SCALE / 2 per unit of depth; orthographic rays are parallel and offset
// across a viewport `ortho_height` world units tall, so a scene keeps its
// framing when `ortho_height` is the perspective frustum height at the orbit
// distance.
fn primary_ray(
  uv: vec2<f32>,
  camera: vec3<f32>,
  look_at: vec3<f32>,
  projection: f32,
  ortho_height: f32,
) -> PrimaryRay {
  let basis = camera_basis(camera, look_at);
  var ray: PrimaryRay;
  if (projection > 0.5) {
    ray.origin = camera + (uv.x * basis.right + uv.y * basis.up) * ortho_height;
    ray.direction = basis.forward;
  } else {
    ray.origin = camera;
    ray.direction = safe_normalize(
      basis.forward + 1.5 * (uv.x * basis.right + uv.y * basis.up),
    );
  }
  return ray;
}
"""


def inject_camera(template: str) -> str:
    """Substitute the shared camera code into a shader template."""
    if CAMERA_MARKER not in template:
        raise ValueError(f"Shader template is missing the {CAMERA_MARKER!r} marker")
    return template.replace(CAMERA_MARKER, CAMERA_WGSL, 1)
