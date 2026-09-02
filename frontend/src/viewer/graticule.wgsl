// WGSL for the viewport's ground grid — where the floor is.
//
// One fullscreen triangle, emitted at the far plane (z = 1) and depth-tested
// `less-equal` against the depth the SDF pass writes. A ray miss writes exactly
// 1.0, so the grid paints on the background and is rejected wherever any
// geometry, FEM surface or path-traced silhouette wrote a nearer depth. That is
// what "drawn behind the geometry so the part occludes it" means here, and it
// costs no sorting, no readback and no second pass over the scene.
//
// What the fragment does is raycast the ground plane. The camera basis arrives
// as uniforms, each fragment reconstructs the same ray the preview shader would
// have used, intersects it with z = 0 — the world is Z-up, so the floor is the
// XY plane a sketch lies on — and rules a square grid in *world* coordinates on
// it. The grid therefore recedes with perspective and slides under the part as
// you orbit, which is the whole reason it tells you anything about where things
// are.
//
// Two things keep the horizon from turning into moiré. Coverage is measured in
// pixels through `fwidth`, so a line is one pixel wide however oblique the
// plane is; and the pattern fades out both with distance from the orbit target
// and as the cell size falls below a few pixels, so the far field dissolves
// instead of aliasing.
//
// It is deliberately faint — a minor line is about 1.35:1 against the paper,
// below the band structure is held to. It is a spatial cue, not structure, and
// it has to disappear the moment attention lands on the geometry over it.
//
// The colours arrive as uniforms from the token layer; there are no literals.

struct Graticule {
  // width, height (framebuffer px), line width (px), 1 when orthographic
  frame   : vec4<f32>,
  // camera position xyz, orthographic frame height in world units
  eye     : vec4<f32>,
  // camera right xyz, viewport aspect
  right   : vec4<f32>,
  // camera up xyz, grid spacing in world units
  up      : vec4<f32>,
  // camera forward xyz, the projection's field scale
  forward : vec4<f32>,
  // minor grid lines: rgb + alpha
  line    : vec4<f32>,
  // every nth line, drawn a step firmer: rgb + alpha
  major   : vec4<f32>,
  // the two ground axes (X at y = 0, Y at x = 0): rgb + alpha
  axis    : vec4<f32>,
  // fade start, fade end (world radius), axis width multiple, lines per major
  fade    : vec4<f32>,
  // orbit centre xyz — the fade is measured from it, unused w
  centre  : vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Graticule;

@vertex
fn vs_graticule(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
  // One oversized triangle covering the clip rectangle, at the far plane.
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  return vec4<f32>(corners[vertex_index], 1.0, 1.0);
}

/// Coverage of a line of half-width `half_width` at distance `distance` px.
fn hairline(distance: f32, half_width: f32) -> f32 {
  return clamp(half_width - distance + 0.5, 0.0, 1.0);
}

/// Straight "over" compositing, non-premultiplied, so the pass can blend once.
fn over(dst: vec4<f32>, src_rgb: vec3<f32>, src_a: f32) -> vec4<f32> {
  let alpha = src_a + dst.a * (1.0 - src_a);
  if (alpha <= 1e-6) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  let rgb = (src_rgb * src_a + dst.rgb * dst.a * (1.0 - src_a)) / alpha;
  return vec4<f32>(rgb, alpha);
}

@fragment
fn fs_graticule(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
  let size = g.frame.xy;
  let half_width = g.frame.z * 0.5;
  let orthographic = g.frame.w > 0.5;
  let aspect = g.right.w;
  let spacing = max(g.up.w, 1e-6);

  // The same field coordinates the projection in `viewer/math.ts` uses:
  // px = (u / aspect + 0.5) * width, py = (0.5 - v) * height.
  let u = (frag.x / size.x - 0.5) * aspect;
  let v = 0.5 - frag.y / size.y;
  let lateral = g.right.xyz * u + g.up.xyz * v;

  // Perspective rays fan out from the eye; orthographic rays are parallel and
  // the fan becomes an offset of the origin. One `select` rather than a branch,
  // because the derivatives below need uniform control flow.
  let ray_origin = select(g.eye.xyz, g.eye.xyz + lateral * g.eye.w, orthographic);
  let ray_dir = select(
    normalize(g.forward.xyz + lateral * g.forward.w),
    g.forward.xyz,
    orthographic,
  );

  // Intersect the floor. `t` is the distance along a unit ray, so it doubles as
  // the depth the fade is measured in.
  let denominator = ray_dir.z;
  let t = -ray_origin.z / select(denominator, 1e-6, abs(denominator) < 1e-6);
  let hit = select(0.0, 1.0, abs(denominator) >= 1e-6 && t > 0.0);
  let ground = ray_origin + ray_dir * t;
  let plane = ground.xy;

  // Derivatives are taken unconditionally: WGSL requires uniform control flow
  // for them, and a quad that straddles the horizon simply reports an enormous
  // width, which the coverage below turns into nothing. That is the correct
  // answer there anyway.
  let cell = plane / spacing;
  let cell_width = fwidth(cell);
  let plane_width = fwidth(plane);

  // Distance to the nearest ruled line, converted from cells to pixels, so a
  // line is one pixel wide no matter how oblique the plane is.
  let to_line = abs(cell - round(cell));
  let line_px = to_line / max(cell_width, vec2<f32>(1e-8));
  let minor = hairline(min(line_px.x, line_px.y), half_width);

  // Every nth line again, a step firmer — the only hierarchy the plane has,
  // and what lets you count cells without reading each one.
  let major_cell = cell / max(g.fade.w, 1.0);
  let major_width = fwidth(major_cell);
  let major_to_line = abs(major_cell - round(major_cell));
  let major_px = major_to_line / max(major_width, vec2<f32>(1e-8));
  let major = hairline(min(major_px.x, major_px.y), half_width);

  // Fade outward from the orbit target, not from the eye: what you are looking
  // at is the part of the floor worth ruling, and everything past it dissolves
  // rather than aliasing into moiré at the horizon. The second fade catches the
  // same failure arriving from the other direction — a steep view close up,
  // where a cell falls below a couple of pixels.
  let radius = length(plane - g.centre.xy);
  let distance_fade = 1.0 - smoothstep(g.fade.x, g.fade.y, radius);
  let density_fade = 1.0 - smoothstep(0.125, 0.5, max(cell_width.x, cell_width.y));
  let major_density = 1.0 - smoothstep(0.125, 0.5, max(major_width.x, major_width.y));
  let visibility = hit * distance_fade;

  var paint = over(paint_zero(), g.line.rgb, minor * g.line.a * visibility * density_fade);
  paint = over(paint, g.major.rgb, major * g.major.a * visibility * major_density);

  // The two ground axes: the X axis is the line y = 0, the Y axis is x = 0.
  // A step above a major line and not subject to the density fade — they are
  // two marks, not a pattern, so they cannot moiré, and they are what says
  // which way the world is pointing once the grid itself has faded out.
  let axis_px = abs(plane) / max(plane_width, vec2<f32>(1e-8));
  let axis_half = half_width * g.fade.z;
  let axis = max(hairline(axis_px.y, axis_half), hairline(axis_px.x, axis_half));
  return over(paint, g.axis.rgb, axis * g.axis.a * visibility);
}

/// The empty accumulator. A function so the compositing chain reads as one
/// expression from nothing to the final colour.
fn paint_zero() -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
