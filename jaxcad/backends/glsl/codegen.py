"""Public API for compiling JAX SDF functions to GLSL."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .._stablehlo_emitter import StableHLOToGLSL

_RAYMARCHER = """\
#version 430

// ── compiled SDF ─────────────────────────────────────────────────────────────
{sdf_code}

// ── raymarching ───────────────────────────────────────────────────────────────
uniform vec2  iResolution;
uniform vec3  uCameraPos;
uniform vec3  uCameraTarget;
uniform vec3  uLightDir;
uniform vec3  uBgColor;
out vec4 fragColor;

#define MAX_STEPS {max_steps}
#define MAX_DIST  {max_dist:.1f}
#define SURF_EPS  {surf_eps:.6f}
#define NORM_EPS  0.001

vec3 sdf_normal(vec3 p) {{
    vec2 e = vec2(NORM_EPS, 0.0);
    return normalize(vec3(
        sdf(p + e.xyy) - sdf(p - e.xyy),
        sdf(p + e.yxy) - sdf(p - e.yxy),
        sdf(p + e.yyx) - sdf(p - e.yyx)
    ));
}}

float trace(vec3 ro, vec3 rd) {{
    float t = 0.01;
    for (int i = 0; i < MAX_STEPS; i++) {{
        float d = sdf(ro + rd * t);
        if (d < SURF_EPS) return t;
        if (t > MAX_DIST) return -1.0;
        t += d;
    }}
    return -1.0;
}}

float soft_shadow(vec3 ro, vec3 rd, float k) {{
    float res = 1.0, t = 0.02;
    for (int i = 0; i < 24; i++) {{
        float h = sdf(ro + rd * t);
        if (h < 0.001) return 0.0;
        res = min(res, k * h / t);
        t += h;
        if (t > 20.0) break;
    }}
    return clamp(res, 0.0, 1.0);
}}

void main() {{
    vec2 uv = (gl_FragCoord.xy / iResolution - 0.5)
              * vec2(iResolution.x / iResolution.y, 1.0);

    vec3 fwd   = normalize(uCameraTarget - uCameraPos);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up    = cross(right, fwd);

    vec3 ro = uCameraPos;
    vec3 rd = normalize(fwd + 1.5 * (uv.x * right + uv.y * up));

    float t   = trace(ro, rd);
    vec3  col = uBgColor;

    if (t >= 0.0) {{
        vec3  pos  = ro + rd * t;
        vec3  nor  = sdf_normal(pos);
        vec3  ldir = normalize(uLightDir);
        float diff = max(dot(nor, ldir), 0.0);
        float sha  = soft_shadow(pos + 0.02 * nor, ldir, 8.0);
        float spec = pow(max(dot(reflect(-ldir, nor), -rd), 0.0), 32.0);
        col = vec3(0.18) * 0.15 + vec3(0.85) * diff * sha + vec3(0.4) * spec;
    }}

    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    fragColor = vec4(col, 1.0);
}}
"""


def compile_sdf_to_glsl(
    fn: Callable,
    example_point: jnp.ndarray | None = None,
) -> str:
    """Compile a JAX SDF function to a GLSL function string.

    Uses ``jax.export`` to get StableHLO, then converts each op to GLSL.
    The emitted entry-point is always named ``sdf``.

    Args:
        fn: Callable ``(p: f32[3]) -> f32[]`` traceable by JAX.
        example_point: Example input for shape inference.  Defaults to zeros.

    Returns:
        GLSL source for the SDF function(s) — no surrounding shader boilerplate.
    """
    if example_point is None:
        example_point = jnp.zeros(3)
    return StableHLOToGLSL().compile(fn, example_point)


def build_fragment_shader(
    fn: Callable,
    example_point: jnp.ndarray | None = None,
    max_steps: int = 64,
    max_dist: float = 100.0,
    surf_eps: float = 1e-3,
) -> str:
    """Return a complete GLSL 4.30 fragment shader with the compiled SDF.

    The shader includes sphere tracing, normal estimation, soft shadows,
    and Blinn-Phong lighting.  Camera and light are set via uniforms.
    """
    sdf_code = compile_sdf_to_glsl(fn, example_point)
    return _RAYMARCHER.format(
        sdf_code=sdf_code,
        max_steps=max_steps,
        max_dist=max_dist,
        surf_eps=surf_eps,
    )
