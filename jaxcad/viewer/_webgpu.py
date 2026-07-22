"""Shared WebGPU shader used by notebook and standalone viewers."""

from __future__ import annotations

_SDF_MARKER = "__JAXCAD_SDF_CODE__"

WGSL_VIEWER_TEMPLATE = r"""__JAXCAD_SDF_CODE__

struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.001;
  return normalize(vec3<f32>(
    sdf(p + vec3<f32>( e, 0.0, 0.0)) - sdf(p + vec3<f32>(-e, 0.0, 0.0)),
    sdf(p + vec3<f32>(0.0,  e, 0.0)) - sdf(p + vec3<f32>(0.0, -e, 0.0)),
    sdf(p + vec3<f32>(0.0, 0.0,  e)) - sdf(p + vec3<f32>(0.0, 0.0, -e)),
  ));
}

fn trace(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.01;
  for (var i = 0; i < 96; i++) {
    let d = sdf(ro + rd * t);
    if (d < 0.001) { return t; }
    if (t > 100.0) { return -1.0; }
    t += d;
  }
  return -1.0;
}

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, k: f32) -> f32 {
  var res = 1.0;
  var t = 0.02;
  for (var i = 0; i < 24; i++) {
    let h = sdf(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    res = min(res, k * h / t);
    t += h;
    if (t > 20.0) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let res = u.resolution.xy;
  let uv = (frag.xy / res - 0.5) * vec2<f32>(res.x / res.y, -1.0);

  let cam = u.camera_pos.xyz;
  let tgt = u.camera_target.xyz;
  let fwd = normalize(tgt - cam);
  let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
  let up = cross(right, fwd);

  let ro = cam;
  let rd = normalize(fwd + 1.5 * (uv.x * right + uv.y * up));

  let t = trace(ro, rd);
  var col = u.bg_color.xyz;

  if (t >= 0.0) {
    let pos = ro + rd * t;
    let nor = sdf_normal(pos);
    let ldir = normalize(u.light_dir.xyz);
    let diff = max(dot(nor, ldir), 0.0);
    let sha = soft_shadow(pos + 0.02 * nor, ldir, 8.0);
    let spec = pow(max(dot(reflect(-ldir, nor), -rd), 0.0), 32.0);
    col = vec3<f32>(0.027) + vec3<f32>(0.85) * diff * sha + vec3<f32>(0.4) * spec;
  }

  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
  return vec4<f32>(col, 1.0);
}
"""


def build_viewer_shader(sdf_code: str) -> str:
    """Embed a compiled ``sdf`` function in the complete viewer shader."""
    if _SDF_MARKER in sdf_code:
        raise ValueError(f"SDF source cannot contain the reserved marker {_SDF_MARKER!r}")
    return WGSL_VIEWER_TEMPLATE.replace(_SDF_MARKER, sdf_code, 1)
