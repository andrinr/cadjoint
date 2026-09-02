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
//
// The SDF views ride in the six scalars this struct already carried and never
// used, rather than in an eighth vec4. The struct is a contract with two
// writers — the playground renderer and the notebook widget in `widget.py` —
// and the widget allocates exactly 112 bytes, so growing it would fault the
// widget rather than merely ignore the new fields. Every one of these six is
// written as 0 by the widget, and 0 is the inert value of each:
//
// resolution.z     SDF view: 0 solid, 1 signed slice, 2 gradient magnitude
// resolution.w     slice plane axis: 0 X, 1 Y, 2 Z
// camera_pos.w     slice plane coordinate, world units
// camera_target.w  the level set traced: f = c rather than f = 0
// bg_color.w       isoline spacing on the slice, world units; 0 draws none
// path_settings.w  sphere-tracing step budget; 0 means the default
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

const SDF_VIEW_SOLID: f32 = 0.0;
const SDF_VIEW_SLICE: f32 = 1.0;
const SDF_VIEW_GRADIENT: f32 = 2.0;
const SDF_VIEW_NORMAL: f32 = 3.0;
const SDF_VIEW_DEPTH: f32 = 4.0;

/// True for the two views that replace the solid's shading rather than cutting
/// a plane through the field.
fn sdf_view_shades_surface(mode: f32) -> bool {
  return mode == SDF_VIEW_NORMAL || mode == SDF_VIEW_DEPTH;
}

/// The default march, kept identical to what this shader used before the
/// budget became a uniform, so a caller that writes 0 gets the old picture.
const DEFAULT_TRACE_STEPS: i32 = 96;

fn trace_steps() -> i32 {
  return select(i32(u.path_settings.w), DEFAULT_TRACE_STEPS, u.path_settings.w < 1.0);
}

/**
 * The field the whole shader traces, shades and shadows against.
 *
 * Every read of the scene goes through here rather than through `sdf`
 * directly, so the isosurface offset is not a shell drawn over the real
 * surface: the primary ray, the normal and both shadow marches all agree on
 * where the surface is, and `f = c` lights and occludes like the solid it is.
 * Subtracting a constant leaves the field's gradient untouched, so the march
 * stays as conservative as it was.
 */
fn scene_field(p: vec3<f32>) -> f32 {
  return sdf(p) - u.camera_target.w;
}

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
    scene_field(p + vec3<f32>( e, 0.0, 0.0)) - scene_field(p + vec3<f32>(-e, 0.0, 0.0)),
    scene_field(p + vec3<f32>(0.0,  e, 0.0)) - scene_field(p + vec3<f32>(0.0, -e, 0.0)),
    scene_field(p + vec3<f32>(0.0, 0.0,  e)) - scene_field(p + vec3<f32>(0.0, 0.0, -e)),
  ));
}

// |∇f| by the same central differences, at a step scaled to what is on screen.
//
// This is the number that says whether the field is still a *distance*. An
// exact SDF has |∇f| = 1 everywhere it is differentiable; a smooth union, an
// offset or a scaled primitive breaks that, and where it is broken the
// mesher's Hermite data and the seam projections are being fed a lie. Drawn
// as a deviation from 1, the non-metric regions are exactly the coloured ones.
fn sdf_gradient_magnitude(p: vec3<f32>, e: f32) -> f32 {
  return length(vec3<f32>(
    scene_field(p + vec3<f32>( e, 0.0, 0.0)) - scene_field(p + vec3<f32>(-e, 0.0, 0.0)),
    scene_field(p + vec3<f32>(0.0,  e, 0.0)) - scene_field(p + vec3<f32>(0.0, -e, 0.0)),
    scene_field(p + vec3<f32>(0.0, 0.0,  e)) - scene_field(p + vec3<f32>(0.0, 0.0, -e)),
  )) / (2.0 * e);
}

fn trace(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.01;
  let budget = trace_steps();
  for (var i = 0; i < budget; i++) {
    let d = scene_field(ro + rd * t);
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
  let budget = trace_steps() / 2;
  for (var i = 0; i < budget; i++) {
    let h = scene_field(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    t += max(h, 0.002);
    if (t > 30.0) { break; }
  }
  return 1.0;
}

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, k: f32) -> f32 {
  var visibility = 1.0;
  var t = 0.02;
  let budget = trace_steps() / 3;
  for (var i = 0; i < budget; i++) {
    let h = scene_field(ro + rd * t);
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
  /// Set on a pixel the slice owns: its colour is already display-encoded.
  sdf_view : bool,
};

// ── the SDF views ──────────────────────────────────────────────────────────
//
// A diverging ramp centred on zero: violet inside the solid, ochre outside it,
// the viewport's own paper at the crossing. Mirrored from
// `frontend/src/viewer/sdfRamp.ts`, which carries the measurements that chose
// the two hues and how far they sit from viridis and magma.
//
// These are *display* values, and everything else this shader produces is
// linear radiance that then goes through ACES and gamma. The slice is data,
// not light — a tone map applied to a ramp is a ramp you can no longer read a
// value off — so `sdf_view_color` returns the display value directly and
// `trace_pixel` marks the pixel so the tone map is skipped.
const SDF_INSIDE: vec3<f32> = vec3<f32>(0.3924, 0.0015, 0.9925);
const SDF_OUTSIDE: vec3<f32> = vec3<f32>(0.5398, 0.3384, 0.0033);
const SDF_CENTRE: vec3<f32> = vec3<f32>(0.902, 0.902, 0.914);
/// The achromatic ink both isoline families are drawn in.
const SDF_ISOLINE: vec3<f32> = vec3<f32>(0.12, 0.12, 0.13);

/// The ramp at a normalized signed value, t in [-1, 1]. See `sdfRamp.ts` for
/// why the interpolation is in sqrt(|t|) rather than in |t|.
fn sdf_ramp(t: f32) -> vec3<f32> {
  let clamped = clamp(t, -1.0, 1.0);
  let end = select(SDF_OUTSIDE, SDF_INSIDE, clamped < 0.0);
  return mix(SDF_CENTRE, end, sqrt(abs(clamped)));
}

/**
 * How dark this pixel's isoline ink is, 0…1.
 *
 * Screen-space derivatives turn "within half a line width of a multiple of
 * `spacing`" into a line that is one pixel wide whatever the zoom — the same
 * contract the floor grid holds itself to, which is why the spacing handed in
 * is the floor grid's own. A contour that thickened as you zoomed out would
 * stop being a contour and start being a band.
 */
fn sdf_isoline(value: f32, spacing: f32, width: f32) -> f32 {
  if (spacing <= 0.0) { return 0.0; }
  let scaled = value / spacing;
  let gradient = max(fwidth(scaled), 1e-6);
  let distance = abs(fract(scaled - 0.5) - 0.5) / gradient;
  // Screen-space level of detail, and the whole far field depends on it.
  //
  // `gradient` is how many intervals this pixel spans, so its reciprocal is
  // how many pixels one interval is worth. Out where the field runs fast —
  // away from the part, or at a shallow grazing angle — that falls below a
  // pixel, and a contour test sampled once per pixel then answers yes or no
  // essentially at random: the plane turns to black-and-white speckle. There
  // is no amount of anti-aliasing that fixes an interval finer than the
  // sampling; the only correct answer is to stop drawing that tier. It fades
  // out between two and six pixels per interval and the diverging ramp is
  // left to carry the value on its own, which it does without aliasing
  // because it is continuous.
  let pixels_per_interval = 1.0 / gradient;
  let lod = smoothstep(2.0, 6.0, pixels_per_interval);
  return (1.0 - smoothstep(0.0, width, distance)) * lod;
}

/// The single, heavier contour at one stated value — the zero level set on a
/// signed slice, |grad f| = 1 on a gradient slice.
fn sdf_level_line(value: f32, width: f32) -> f32 {
  let gradient = max(fwidth(value), 1e-6);
  return 1.0 - smoothstep(0.0, width, abs(value) / gradient);
}

/**
 * Linear camera depth, mapped bright-near to dark-far.
 *
 * The range is the *framed* depth rather than the clip planes: the near end
 * is half a frame height in front of the orbit target and the far end half a
 * frame behind it, so what fills the ramp is what fills the picture. Clip
 * planes would spend almost the whole ramp on empty space either side of the
 * part and leave the part itself a flat mid grey — the usual reason a z-pass
 * is unreadable. It follows the zoom, like the grid gain and the slice's own
 * contour interval, and the two ends are printed in the viewport readout.
 */
fn depth_tone(along_ray: f32) -> f32 {
  let orbit = length(u.camera_pos.xyz - u.camera_target.xyz);
  let frame = max(u.display.y, 1e-4);
  let near = max(orbit - 0.5 * frame, 1e-3);
  let far = orbit + 0.5 * frame;
  return 1.0 - clamp((along_ray - near) / max(far - near, 1e-6), 0.0, 1.0);
}

struct SliceHit {
  distance : f32,
  hit      : bool,
};

/**
 * Half-width of the cutting card, in world units.
 *
 * The plane is mathematically infinite; the thing drawn is not. A card of
 * stated size leaves the rest of the viewport for the faded solid, so the
 * slice sits *in* the scene and can be seen in relation to it — an infinite
 * plane fills the frame and there is nothing left to relate it to.
 *
 * Mirrors `SDF_SLICE_RANGE` in `frontend/src/viewer/display.ts`, which is also
 * the travel of the slice control, so the card sweeps a cube of its own size.
 */
const SDF_CARD_HALF: f32 = 2.0;

/// Where the primary ray meets the cutting card. A ray running along the plane,
/// arriving from behind the camera, or landing off the card's edge is a miss.
fn slice_intersection(ro: vec3<f32>, rd: vec3<f32>) -> SliceHit {
  let axis = i32(round(clamp(u.resolution.w, 0.0, 2.0)));
  var normal = vec3<f32>(0.0, 0.0, 0.0);
  normal[axis] = 1.0;
  let denominator = dot(rd, normal);
  if (abs(denominator) < 1e-5) { return SliceHit(0.0, false); }
  let distance = (u.camera_pos.w - dot(ro, normal)) / denominator;
  let point = ro + rd * distance;
  var on_card = distance > 0.0;
  for (var i = 0; i < 3; i++) {
    if (i != axis && abs(point[i]) > SDF_CARD_HALF) { on_card = false; }
  }
  return SliceHit(distance, on_card);
}

/**
 * The slice's colour at a world point.
 *
 * `mode` picks what is being drawn: the signed field itself, or |∇f|. Both are
 * normalized against something the viewport already states rather than against
 * a hidden constant — the field against half the framed height, so a full
 * ramp side is exactly half a screen of distance at the current gain, and the
 * gradient against ±0.5 about 1.0, so an exact SDF is the paper it is drawn on
 * and every departure from a metric field is coloured.
 */
fn sdf_view_color(point: vec3<f32>, mode: f32) -> vec3<f32> {
  let spacing = u.bg_color.w;
  let gradient = mode == SDF_VIEW_GRADIENT;
  // One scalar, two meanings, so the derivatives below are taken once and in
  // straight-line code: the signed field, or its own gradient magnitude.
  // Neither is clamped or folded — a negative distance is drawn exactly as a
  // positive one is, in the other hue, so the interior of a part is a graded
  // field running to its medial axis and not a flat fill.
  let step = max(spacing * 0.02, 1e-4);
  let value = select(scene_field(point), sdf_gradient_magnitude(point, step) - 1.0, gradient);

  // The ramp saturates two major divisions out, which is also where the minor
  // contours stop. One number governs both, so the hue and the line density
  // are telling the reader the same thing.
  let major = select(spacing, 0.1, gradient);
  let minor = major * 0.2;
  let scale = select(2.0 * spacing, 0.5, gradient);
  var color = sdf_ramp(value / scale);

  // Two tiers of contour.
  //
  // Uniform spacing at the grid rung is right out where the field is smooth
  // and far too coarse where it is interesting: everything worth reading off
  // a distance field happens within a division or two of the surface. So the
  // minor tier is a fifth of the major — the floor grid's own subdivision, so
  // a contour interval is always a stateable number — and it is faded out
  // across the band it covers rather than stopped at its edge, which would
  // draw a ring around nothing. The gradient view keeps one uniform family:
  // its interesting region is a *value*, 1.0, not a neighbourhood of a
  // surface, so a taper would have nothing to taper toward.
  let taper = select(
    1.0 - smoothstep(0.0, 2.0 * major, abs(value)),
    0.0,
    gradient,
  );
  color = mix(color, SDF_ISOLINE, sdf_isoline(value, minor, 0.75) * taper * 0.3);
  color = mix(
    color,
    SDF_ISOLINE,
    sdf_isoline(value, major, 0.9) * select(0.5, 0.35, gradient),
  );
  // And the one heavier line at the value that matters: the zero level set on
  // a signed slice — which is the part's own section outline — or |∇f| = 1 on
  // a gradient slice.
  return mix(color, SDF_ISOLINE, sdf_level_line(value, select(1.5, 1.2, gradient)));
}

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
  result.sdf_view = false;

  let view_mode = u.resolution.z;
  if (result.hit) {
    let position = ray.origin + ray.direction * distance;
    let normal = sdf_normal(position);
    result.color = shade_material(position, normal, ray.direction);
    result.position = position;

    // The two surface views replace the shading outright. Both are data, not
    // light, so they skip the tone map — see `resolve_color`.
    if (view_mode == SDF_VIEW_NORMAL) {
      // The standard encoding, n * 0.5 + 0.5, in *world* axes: +X red, +Y
      // green, +Z blue, so a face pointing at the sky is the blue everyone
      // expects a normal map to be.
      result.color = normal * 0.5 + vec3<f32>(0.5);
      result.sdf_view = true;
    } else if (view_mode == SDF_VIEW_DEPTH) {
      result.color = vec3<f32>(depth_tone(distance));
      result.sdf_view = true;
    }

    let xray = clamp(u.display.w, 0.0, 1.0);
    if (xray > 0.0 && !sdf_view_shades_surface(view_mode)) {
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
      // An x-rayed solid still *is* somewhere, and it still writes its depth.
      //
      // This used to clear `occludes`, which made the depth buffer say the ray
      // missed — and the floor grid, which draws behind whatever the scene has
      // written, then came through the part. On the default (X-Ray) preset the
      // ground grid was visible straight through the heat sink, so the drawing
      // read as a grid laid over the model rather than a model standing on the
      // ground. Depth is the answer to "what is in front of what" and the
      // solid may not lie about it to buy an overlay a see-through.
      //
      // Construction geometry gets its see-through from its own end instead:
      // while the solid is x-rayed the overlay pipelines are built with
      // `depthCompare: "always"` (`createOverlayPipelines` in
      // `viewer/pipelines.ts`), which is what the gizmos have always used. The
      // grid is not construction geometry — it is the ground the part stands
      // on — so it keeps testing depth and is occluded by every silhouette.
    }
  }

  // The slice, composited last so it can read the solid it stands in front of.
  //
  // Depth order is respected rather than assumed: the plane is drawn where it
  // is nearer than the surface, and the solid occludes it where it is not, so
  // the picture is a cut through the scene and not an overlay floating on it.
  // Everywhere the solid survives it is stepped back toward the ground, which
  // leaves the part legible as a silhouette without letting it compete with a
  // ramp the reader is about to take a value off.
  //
  // The shape of this block is dictated by WGSL's uniformity analysis, and it
  // is worth saying why. `sdf_view_color` takes screen-space derivatives — the
  // isolines are one pixel wide at every zoom, which is a `fwidth` — and a
  // derivative may only be taken where control flow is *uniform* across the
  // quad. `u.resolution.z` comes out of a uniform buffer, so branching on it
  // is uniform and the outer `if` is legal. Whether this pixel's ray reaches
  // the plane before it reaches the solid is not: it varies pixel to pixel.
  // So the colour is computed unconditionally inside the uniform branch and
  // only *chosen* inside the non-uniform one. wgpu's naga accepts the naive
  // nesting; Tint, which is what a browser actually compiles with, rejects it.
  // Written as one uniform if/else chain, and deliberately without an early
  // return: `view_mode` comes from a uniform buffer, so every arm here is
  // reached under uniform control flow, which is what lets the slice arm take
  // the screen-space derivatives its contours need. A `return` out of the
  // middle of it would end that, whatever the condition was.
  if (sdf_view_shades_surface(view_mode)) {
    // Nothing was hit, so there is no normal and no depth to report. Both
    // views state that rather than falling back on the lit background, which
    // would read as a surface: an absent normal is the encoding's own zero
    // vector, and an absent depth is past the far end of the ramp.
    if (!result.hit) {
      result.color = select(
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(depth_tone(1e6)),
        view_mode == SDF_VIEW_DEPTH,
      );
      result.sdf_view = true;
    }
  } else if (view_mode != SDF_VIEW_SOLID) {
    result.color = mix(environment_radiance(ray.direction), result.color, 0.28);
    let plane = slice_intersection(ray.origin, ray.direction);
    let point = ray.origin + ray.direction * max(plane.distance, 0.0);
    let slice_color = sdf_view_color(point, view_mode);
    // On the card, the field wins outright — the solid does not occlude it,
    // in front of the plane or behind it. That is what a section view is, and
    // it is the only way the *interior* of a part is visible at all: a plane
    // inside a solid is always behind that solid's own front surface, so depth
    // ordering hid exactly the half of the field the view exists to show.
    // Everywhere off the card the faded solid stays, which is what the card is
    // read against.
    if (plane.hit) {
      result.color = slice_color;
      result.position = point;
      result.hit = true;
      result.occludes = true;
      result.sdf_view = true;
    }
  }
  return result;
}

fn display_color(color: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(pow(aces_tone_map(color), vec3<f32>(1.0 / 2.2)), 1.0);
}

/// Tone-map light, pass data through. See `sdf_view_color`.
fn resolve_color(result: TraceResult) -> vec4<f32> {
  if (result.sdf_view) { return vec4<f32>(result.color, 1.0); }
  return display_color(result.color);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  return resolve_color(trace_pixel(frag.xy));
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
  fragment.color = resolve_color(result);
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
