"""Shared WebGPU preview shader used by notebook and standalone viewers."""

from __future__ import annotations

_SDF_MARKER = "__JAXCAD_SDF_CODE__"

DEFAULT_MATERIAL_WGSL = r"""
fn material_base(_p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.85, 0.85, 0.85, 0.45);
}

fn material_optics(_p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.0, 1.0, 1.5, 0.0);
}
"""

WGSL_VIEWER_TEMPLATE = r"""__JAXCAD_SDF_CODE__

const PI: f32 = 3.141592653589793;

struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

// Only referenced by fs_main_depth, so pipelines built from vs_main/fs_main
// (the notebook widget) never see this binding in their derived layout.
struct ViewUniforms {
  view_proj : mat4x4<f32>,
};
@group(0) @binding(2) var<uniform> view: ViewUniforms;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  return v * inverseSqrt(max(dot(v, v), 1e-12));
}

fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.001;
  return safe_normalize(vec3<f32>(
    sdf(p + vec3<f32>( e, 0.0, 0.0)) - sdf(p + vec3<f32>(-e, 0.0, 0.0)),
    sdf(p + vec3<f32>(0.0,  e, 0.0)) - sdf(p + vec3<f32>(0.0, -e, 0.0)),
    sdf(p + vec3<f32>(0.0, 0.0,  e)) - sdf(p + vec3<f32>(0.0, 0.0, -e)),
  ));
}

fn trace(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.01;
  for (var i = 0; i < 96; i++) {
    let d = sdf(ro + rd * t);
    if (abs(d) < 0.001) { return t; }
    if (t > 100.0) { return -1.0; }
    t += max(abs(d) * 0.9, 0.0005);
  }
  return -1.0;
}

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, k: f32) -> f32 {
  var visibility = 1.0;
  var t = 0.02;
  for (var i = 0; i < 32; i++) {
    let h = sdf(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    visibility = min(visibility, k * h / t);
    t += max(h * 0.9, 0.0005);
    if (t > 30.0) { break; }
  }
  return clamp(visibility, 0.0, 1.0);
}

fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
  let horizon = max(u.bg_color.xyz, vec3<f32>(0.01));
  let ground = horizon * 0.45;
  let sky = mix(horizon * 1.8, vec3<f32>(0.58, 0.70, 0.92), 0.55);
  return mix(ground, sky, smoothstep(-0.2, 0.7, direction.y));
}

fn fresnel_schlick(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) *
    pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn shade_material(
  position: vec3<f32>,
  normal: vec3<f32>,
  ray_direction: vec3<f32>,
) -> vec3<f32> {
  let base = material_base(position);
  let optics = material_optics(position);
  let base_color = clamp(base.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  let roughness = clamp(base.w, 0.04, 1.0);
  let metallic = clamp(optics.x, 0.0, 1.0);
  let opacity = clamp(optics.y, 0.0, 1.0);
  let reflectivity = clamp(optics.w, 0.0, 1.0);
  let view_direction = -ray_direction;
  let light_direction = safe_normalize(u.light_dir.xyz);
  let half_vector = safe_normalize(view_direction + light_direction);
  let normal_dot_light = max(dot(normal, light_direction), 0.0);
  let normal_dot_view = max(dot(normal, view_direction), 0.0);
  let normal_dot_half = max(dot(normal, half_vector), 0.0);
  let view_dot_half = max(dot(view_direction, half_vector), 0.0);

  let alpha = roughness * roughness;
  let alpha_squared = alpha * alpha;
  let distribution_denominator =
    normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
  let distribution = alpha_squared /
    max(PI * distribution_denominator * distribution_denominator, 1e-6);
  let geometry_k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  let geometry_light = normal_dot_light /
    max(normal_dot_light * (1.0 - geometry_k) + geometry_k, 1e-6);
  let geometry_view = normal_dot_view /
    max(normal_dot_view * (1.0 - geometry_k) + geometry_k, 1e-6);
  let f0 = mix(vec3<f32>(0.04), base_color, metallic);
  let fresnel = fresnel_schlick(view_dot_half, f0);
  let specular = distribution * geometry_light * geometry_view * fresnel /
    max(4.0 * normal_dot_view * normal_dot_light, 1e-6);
  let diffuse =
    (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * base_color / PI;
  let visibility = soft_shadow(
    position + normal * 0.004,
    light_direction,
    16.0,
  );
  let light_radiance =
    vec3<f32>(1.0, 0.92, 0.82) * max(u.light_dir.w, 1.0);
  let direct =
    (diffuse + specular) * light_radiance * normal_dot_light * visibility;
  let ambient = base_color * environment_radiance(normal) * 0.18;
  let reflected = environment_radiance(reflect(ray_direction, normal));
  let opaque = mix(direct + ambient, reflected, reflectivity);
  let glass_fresnel = fresnel_schlick(
    max(dot(view_direction, normal), 0.0),
    vec3<f32>(0.04),
  );
  let transparent = mix(
    environment_radiance(ray_direction) * base_color,
    reflected,
    glass_fresnel,
  );
  return mix(transparent, opaque, opacity);
}

fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp(
    (color * (a * color + vec3<f32>(b))) /
      (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
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

struct TraceResult {
  color    : vec3<f32>,
  position : vec3<f32>,
  hit      : bool,
};

fn trace_pixel(frag_xy: vec2<f32>) -> TraceResult {
  let res = u.resolution.xy;
  let uv = (frag_xy / res - 0.5) * vec2<f32>(res.x / res.y, -1.0);
  let camera = u.camera_pos.xyz;
  let camera_target = u.camera_target.xyz;
  let forward = safe_normalize(camera_target - camera);
  let right = safe_normalize(cross(forward, vec3<f32>(0.0, 1.0, 0.0)));
  let up = cross(right, forward);
  let ray_direction = safe_normalize(
    forward + 1.5 * (uv.x * right + uv.y * up),
  );
  let distance = trace(camera, ray_direction);

  var result: TraceResult;
  result.color = environment_radiance(ray_direction);
  result.position = camera;
  result.hit = distance >= 0.0;

  if (result.hit) {
    let position = camera + ray_direction * distance;
    let normal = sdf_normal(position);
    result.color = shade_material(position, normal, ray_direction);
    result.position = position;
  }
  return result;
}

fn display_color(color: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(pow(aces_tone_map(color), vec3<f32>(1.0 / 2.2)), 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  return display_color(trace_pixel(frag.xy).color);
}

struct DepthFragment {
  @location(0) color            : vec4<f32>,
  @builtin(frag_depth) depth    : f32,
};

// Same image as fs_main, plus scene depth so construction-tree overlays can be
// depth-tested against the rendered solid instead of floating on top of it.
@fragment
fn fs_main_depth(@builtin(position) frag: vec4<f32>) -> DepthFragment {
  let result = trace_pixel(frag.xy);
  var fragment: DepthFragment;
  fragment.color = display_color(result.color);
  fragment.depth = 1.0;
  if (result.hit) {
    let clip = view.view_proj * vec4<f32>(result.position, 1.0);
    fragment.depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);
  }
  return fragment;
}
"""


def ensure_material_wgsl(scene_code: str) -> str:
    """Append the viewer's default material functions when only an SDF is supplied."""
    has_base = "fn material_base(" in scene_code
    has_optics = "fn material_optics(" in scene_code
    if has_base != has_optics:
        raise ValueError("Scene source must define both material WGSL functions or neither")
    if has_base:
        return scene_code
    return f"{scene_code}\n\n{DEFAULT_MATERIAL_WGSL}"


def build_viewer_shader(scene_code: str) -> str:
    """Embed compiled distance/material functions in the complete preview shader."""
    if _SDF_MARKER in scene_code:
        raise ValueError(f"SDF source cannot contain the reserved marker {_SDF_MARKER!r}")
    return WGSL_VIEWER_TEMPLATE.replace(_SDF_MARKER, ensure_material_wgsl(scene_code), 1)
