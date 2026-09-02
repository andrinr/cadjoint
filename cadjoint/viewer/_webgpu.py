"""Shared WebGPU preview shader used by notebook and standalone viewers."""

from __future__ import annotations

from cadjoint.viewer._camera_wgsl import inject_camera

_SDF_MARKER = "__CADJOINT_SDF_CODE__"

DEFAULT_MATERIAL_WGSL = r"""
fn material_base(_p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.85, 0.85, 0.85, 0.45);
}

fn material_optics(_p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.0, 1.0, 1.5, 0.0);
}
"""

_VIEWER_TEMPLATE = r"""__CADJOINT_SDF_CODE__

const PI: f32 = 3.141592653589793;

// display.x  projection: 0 perspective, 1 orthographic
// display.y  orthographic viewport height in world units
// display.z  DISPLAY_* flag bits, packed as a float
// display.w  x-ray strength, 0 disables
struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
  path_settings: vec4<f32>,
  display      : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

const DISPLAY_SHADOWS: u32 = 1u;
const DISPLAY_REFLECTIONS: u32 = 2u;
const DISPLAY_FLAT: u32 = 4u;
const DISPLAY_HIDE_SOLID: u32 = 8u;
const DISPLAY_HARD_SHADOWS: u32 = 16u;

// Hard shadows stay lifted rather than black: a single occlusion test has no
// penumbra to soften the edge, and fully dark cores hide the geometry inside.
const HARD_SHADOW_FLOOR: f32 = 0.45;

fn display_flag(flag: u32) -> bool {
  return (u32(max(u.display.z, 0.0)) & flag) != 0u;
}

// Only referenced by fs_main_depth, so pipelines built from vs_main/fs_main
// (the notebook widget) never see this binding in their derived layout.
struct ViewUniforms {
  view_proj : mat4x4<f32>,
};
@group(0) @binding(2) var<uniform> view: ViewUniforms;

__CADJOINT_CAMERA__

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

// One occlusion ray, no penumbra: cheaper than soft_shadow and gives the crisp
// edge a drafting view wants.
fn hard_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.02;
  for (var i = 0; i < 48; i++) {
    let h = sdf(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    t += max(h, 0.002);
    if (t > 30.0) { break; }
  }
  return 1.0;
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

// The viewport ground.
//
// `u.bg_color` is the ground colour in *linear radiance*: the caller has
// already inverted this shader's tone map and gamma, so a ray that hits
// nothing lands on exactly the colour the viewport declares.
//
// It is flat, and the flatness is the point. The old version was a gradient —
// ground at 0.45x, sky brightened to 1.8x and mixed toward a hard-coded blue
// — which a dark viewport needs, because there the environment is the only
// thing lighting a miss. Two things rule it out here. A light ground washes
// every face toward its own value, so the gradient's bright half erases the
// model; and the default projection is orthographic, where every primary ray
// shares one direction, so a direction-dependent environment collapses to a
// single colour that *changes as the camera orbits*. A ground you measure a
// field against cannot be a function of the camera.
fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
  return max(u.bg_color.xyz, vec3<f32>(0.001));
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
  let shadow_origin = position + normal * 0.004;
  var visibility = 1.0;
  if (display_flag(DISPLAY_SHADOWS)) {
    if (display_flag(DISPLAY_HARD_SHADOWS)) {
      visibility = mix(HARD_SHADOW_FLOOR, 1.0, hard_shadow(shadow_origin, light_direction));
    } else {
      visibility = soft_shadow(shadow_origin, light_direction, 16.0);
    }
  }

  // Silhouette contour.
  //
  // On a dark ground a part reads because it is the bright thing; on a light
  // one it has to read by its edge, because the tone map pushes any lit face
  // and the background into the same shoulder. This darkens only the sliver
  // where the surface turns away from the eye — a drawn outline that follows
  // the geometry, costing no value anywhere on the interior.
  let facing = abs(dot(normal, view_direction));
  let contour = mix(0.16, 1.0, smoothstep(0.0, 0.30, facing));

  if (display_flag(DISPLAY_FLAT)) {
    // Flat shading: albedo lit only by incidence and shadowing — no specular
    // and no environment. The floor was 0.35 against a black viewport, where
    // the only risk was an unlit face disappearing; on paper the risk is the
    // opposite, a lit face washing into the ground. The range is quoted in
    // linear radiance and the tone map's shoulder eats most of the top of it:
    // 0.30..0.92 of a 0.85 albedo lands at sRGB 166..224 against a 230 ground,
    // which is the pale blob. 0.13..0.48 lands at 105..193 — a solid grey part
    // — and the contour closes the silhouette.
    return base_color * mix(0.13, 0.48, normal_dot_light * visibility) * contour;
  }
  let light_radiance =
    vec3<f32>(1.0, 0.92, 0.82) * max(u.light_dir.w, 1.0);
  let direct =
    (diffuse + specular) * light_radiance * normal_dot_light * visibility;
  // The dome is now roughly the background's own radiance rather than a dark
  // gradient, so the same 0.18 coefficient delivers about five times the
  // ambient it used to. 0.07 keeps the unlit side of a part where it was in
  // absolute terms, which is what stops the whole model flattening to the
  // ground's value.
  let ambient = base_color * environment_radiance(normal) * 0.07;
  let reflected = environment_radiance(reflect(ray_direction, normal));
  let mirror = select(0.0, reflectivity, display_flag(DISPLAY_REFLECTIONS));
  let opaque = mix(direct + ambient, reflected, mirror) * contour;
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
  occludes : bool,
};

fn trace_pixel(frag_xy: vec2<f32>) -> TraceResult {
  let res = u.resolution.xy;
  let uv = (frag_xy / res - 0.5) * vec2<f32>(res.x / res.y, -1.0);
  let ray = primary_ray(
    uv,
    u.camera_pos.xyz,
    u.camera_target.xyz,
    u.display.x,
    u.display.y,
  );
  let hide_solid = display_flag(DISPLAY_HIDE_SOLID);
  let distance = select(trace(ray.origin, ray.direction), -1.0, hide_solid);

  var result: TraceResult;
  result.color = environment_radiance(ray.direction);
  result.position = ray.origin;
  result.hit = distance >= 0.0;
  result.occludes = result.hit;

  if (result.hit) {
    let position = ray.origin + ray.direction * distance;
    let normal = sdf_normal(position);
    result.color = shade_material(position, normal, ray.direction);
    result.position = position;

    let xray = clamp(u.display.w, 0.0, 1.0);
    if (xray > 0.0) {
      // Fade faces that point at the viewer and keep grazing angles solid, so
      // silhouettes and creases stay legible while interiors turn translucent.
      let facing = 1.0 - abs(dot(normal, ray.direction));
      let alpha = mix(0.62, 0.98, facing * facing);
      // On paper a face cannot fade toward the background: the ground is the
      // brightest thing on screen, so "transparent" and "absent" would be the
      // same pixel and an x-rayed solid disappears entirely (measured: sRGB
      // 222 of geometry against a 225 ground). It fades toward a veil instead
      // — the ground knocked down a fifth — and it fades less far, so the
      // ghost is a pale tint at sRGB ~208 with its grazing edges running down
      // to ~104. That is the same reading the dark viewport got from the
      // opposite direction: faint where you look through the part, strong
      // where the surface turns away.
      result.color = mix(
        environment_radiance(ray.direction) * 0.42,
        result.color,
        mix(1.0, alpha, xray),
      );
      // Construction geometry must stay visible through an x-rayed solid.
      result.occludes = false;
    }
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
  if (result.occludes) {
    let clip = view.view_proj * vec4<f32>(result.position, 1.0);
    fragment.depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);
  }
  return fragment;
}
"""

WGSL_VIEWER_TEMPLATE = inject_camera(_VIEWER_TEMPLATE)


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
