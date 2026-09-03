// WGSL for the viewport's construction grid — where the world's planes are.
//
// One fullscreen triangle, emitted at the far plane (z = 1) and depth-tested
// `less-equal` against the depth the SDF pass writes. A ray miss writes exactly
// 1.0, so the grid paints on the background and is rejected wherever any
// geometry, FEM surface or path-traced silhouette wrote a nearer depth. That is
// what "drawn behind the geometry so the part occludes it" means here, and it
// costs no sorting, no readback and no second pass over the scene.
//
// What the fragment does is raycast a world coordinate plane. The camera basis
// arrives as uniforms, each fragment reconstructs the same ray the preview
// shader would have used, intersects it with the plane and rules a square grid
// in *world* coordinates on it. The grid therefore recedes with perspective and
// slides under the part as you orbit, which is the whole reason it tells you
// anything about where things are.
//
// Which plane is not fixed. On z = 0 the grid is right in Top and Bottom and
// useless in Front, Back, Left and Right, where it is exactly edge-on and rules
// as one line — the four views someone reaches for to measure something square.
// `plane_weights` therefore scores all three world planes by how face-on they
// are, biased toward the floor because the floor is the model's ground and the
// rest of the viewport references it, and rules the winner. The two crossovers
// are a few degrees wide and dissolve, so the plane changes without a jump; the
// readout in `components/viewer/Graticule.tsx` names whichever one is up. The
// arithmetic is mirrored on the CPU in `viewer/graticule.ts` — one of them
// draws the grid, the other names it, and they must agree.
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
  // the two in-plane axes: rgb + alpha
  axis    : vec4<f32>,
  // fade start, fade end (world radius), axis width multiple, lines per major
  fade    : vec4<f32>,
  // orbit centre xyz — the fade is measured from it, unused w
  centre  : vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Graticule;

/// How much the floor is preferred over a wall. See `GRID_FLOOR_PREFERENCE`.
const FLOOR_PREFERENCE: f32 = 2.0;

/// Half-width of a crossover, in score units. See `GRID_PLANE_BAND`.
const PLANE_BAND: f32 = 0.06;

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

/// The two world axes a plane is spanned by, and the one it is normal to.
struct PlaneAxes {
  u : vec3<f32>,
  v : vec3<f32>,
  n : vec3<f32>,
};

/// 0 = XY (the floor), 1 = XZ, 2 = YZ — the order `GRID_PLANES` uses.
fn plane_axes(kind: i32) -> PlaneAxes {
  let x = vec3<f32>(1.0, 0.0, 0.0);
  let y = vec3<f32>(0.0, 1.0, 0.0);
  let z = vec3<f32>(0.0, 0.0, 1.0);
  var axes: PlaneAxes;
  axes.u = x;
  axes.v = y;
  axes.n = z;
  if (kind == 1) {
    axes.v = z;
    axes.n = y;
  } else if (kind == 2) {
    axes.u = y;
    axes.v = z;
    axes.n = x;
  }
  return axes;
}

/// How strongly each world plane is drawn, in `plane_axes` order.
///
/// A plane is worth ruling in proportion to how square it is to the camera,
/// |forward · n|; the floor carries `FLOOR_PREFERENCE` on top of that. The
/// smoothstep against the best rival is what makes the change a dissolve: it
/// is 1 for the leader and 0 for the rest except within `PLANE_BAND` of a
/// crossover, where two of them share the paint. Normalised, so the grid never
/// gets fainter or firmer for changing plane.
fn plane_weights(forward: vec3<f32>) -> vec3<f32> {
  let face = abs(forward);
  let score = vec3<f32>(face.z * FLOOR_PREFERENCE, face.y, face.x);
  let rival = vec3<f32>(
    max(score.y, score.z),
    max(score.x, score.z),
    max(score.x, score.y),
  );
  let raw = smoothstep(
    vec3<f32>(-PLANE_BAND),
    vec3<f32>(PLANE_BAND),
    score - rival,
  );
  let total = raw.x + raw.y + raw.z;
  if (total <= 1e-6) {
    return vec3<f32>(1.0, 0.0, 0.0);
  }
  return raw / total;
}

/// Rule one world plane into the accumulating paint.
///
/// Everything below is in the plane's own (u, v) coordinates, so the floor and
/// the two walls are the same code: the grid, the every-fifth major line, and
/// the two axes where this plane meets the other two.
fn rule_plane(
  dst: vec4<f32>,
  kind: i32,
  weight: f32,
  ray_origin: vec3<f32>,
  ray_dir: vec3<f32>,
) -> vec4<f32> {
  let axes = plane_axes(kind);
  let orthographic = g.frame.w > 0.5;
  let half_width = g.frame.z * 0.5;
  let spacing = max(g.up.w, 1e-6);

  // Intersect the plane. `t` is the distance along a unit ray, so under
  // perspective it doubles as the depth in front of the eye — and a hit behind
  // the eye is the sky, not the floor. Under orthographic there is no eye: the
  // origin is a station on the camera plane, the ray is one line through the
  // world, and the plane point on it is on screen whichever side of the
  // station it falls. Testing `t > 0` there is what cut the near third of the
  // ground off along a hard horizontal line.
  let denominator = dot(ray_dir, axes.n);
  let along = -dot(ray_origin, axes.n)
    / select(denominator, 1e-6, abs(denominator) < 1e-6);
  let facing = abs(denominator) >= 1e-6;
  let hit = select(0.0, 1.0, facing && (orthographic || along > 0.0));
  let ground = ray_origin + ray_dir * along;
  let plane = vec2<f32>(dot(ground, axes.u), dot(ground, axes.v));
  let centre = vec2<f32>(dot(g.centre.xyz, axes.u), dot(g.centre.xyz, axes.v));

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
  // at is the part of the plane worth ruling, and everything past it dissolves
  // rather than aliasing into moiré at the horizon. The second fade catches the
  // same failure arriving from the other direction — a steep view close up,
  // where a cell falls below a couple of pixels.
  let radius = length(plane - centre);
  let distance_fade = 1.0 - smoothstep(g.fade.x, g.fade.y, radius);
  let density_fade = 1.0 - smoothstep(0.125, 0.5, max(cell_width.x, cell_width.y));
  let major_density = 1.0 - smoothstep(0.125, 0.5, max(major_width.x, major_width.y));
  let visibility = hit * distance_fade * weight;

  var paint = over(dst, g.line.rgb, minor * g.line.a * visibility * density_fade);
  paint = over(paint, g.major.rgb, major * g.major.a * visibility * major_density);

  // The two axes this plane carries: the lines where it meets the other two
  // coordinate planes, so the floor shows X and Y and a wall shows the vertical
  // one and its own horizontal. A step above a major line and not subject to
  // the density fade — they are two marks, not a pattern, so they cannot moiré,
  // and they are what says which way the world is pointing once the grid itself
  // has faded out.
  let axis_px = abs(plane) / max(plane_width, vec2<f32>(1e-8));
  let axis_half = half_width * g.fade.z;
  let axis = max(hairline(axis_px.y, axis_half), hairline(axis_px.x, axis_half));
  return over(paint, g.axis.rgb, axis * g.axis.a * visibility);
}

@fragment
fn fs_graticule(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
  let size = g.frame.xy;
  let orthographic = g.frame.w > 0.5;
  let aspect = g.right.w;

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

  // All three planes are ruled unconditionally and weighted, rather than
  // branched on: the weights are a function of a uniform, so two of them are
  // zero everywhere except in a crossover, and unrolling keeps every `fwidth`
  // below in flat, uniform control flow.
  let weights = plane_weights(g.forward.xyz);
  var paint = paint_zero();
  paint = rule_plane(paint, 0, weights.x, ray_origin, ray_dir);
  paint = rule_plane(paint, 1, weights.y, ray_origin, ray_dir);
  paint = rule_plane(paint, 2, weights.z, ray_origin, ray_dir);
  return paint;
}

/// The empty accumulator. A function so the compositing chain reads as one
/// expression from nothing to the final colour.
fn paint_zero() -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
