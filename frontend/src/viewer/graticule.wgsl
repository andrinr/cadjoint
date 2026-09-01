// WGSL for the viewport graticule — the instrument faceplate behind the scene.
//
// One fullscreen triangle, emitted at the far plane (z = 1) and depth-tested
// `less-equal` against the depth the SDF pass writes. A ray miss writes exactly
// 1.0, so the graticule paints on the background and is rejected wherever any
// geometry, FEM surface or path-traced silhouette wrote a nearer depth. That is
// what "drawn behind the geometry so the part occludes it" means here, and it
// costs no sorting, no readback, and no second coordinate system.
//
// Everything is computed from `@builtin(position)`, i.e. framebuffer pixels, so
// the pattern is registered with the frame by construction. The colours arrive
// as uniforms from the token layer; there are no literals in here.

struct Graticule {
  // width, height (framebuffer px), division size (px), line width (px)
  frame   : vec4<f32>,
  // division lines: rgb + alpha
  line    : vec4<f32>,
  // the two centre axes and their subdivision ticks: rgb + alpha
  axis    : vec4<f32>,
  // corner brackets: rgb + alpha
  bracket : vec4<f32>,
  // subdivisions per division, tick arm (px), fifth-tick arm (px), bracket arm (px)
  marks   : vec4<f32>,
  // bracket weight (px), bracket inset (px), interior line limit, unused
  edge    : vec4<f32>,
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

/// A 1px-feathered box over [lo, hi]: 1 inside, 0 outside, ramped at the edge.
fn band(x: f32, lo: f32, hi: f32) -> f32 {
  return clamp(min(x - lo, hi - x) + 0.5, 0.0, 1.0);
}

/// Coverage of a line of half-width `half_width` at distance `distance`.
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
  let division = max(g.frame.z, 1.0);
  let half_width = g.frame.w * 0.5;
  // Signed pixel offset from the faceplate centre, which is the viewport
  // centre — the point the orbit target always projects to.
  //
  // Landed on a pixel *centre* rather than on `size * 0.5`. `@builtin(position)`
  // samples at x.5, so an even framebuffer puts the exact centre on a pixel
  // boundary and the two centre axes come out as a pair of half-covered
  // pixels — measurably weaker than the ordinary division lines they are
  // supposed to lead. The half-pixel shift is 0.07% of the frame, far below
  // anything the gain claims.
  let centre = floor(size * 0.5) + 0.5;
  let offset = frag.xy - centre;

  // Each line is snapped to the nearest pixel centre, so every hairline lands
  // on one whole pixel at its full token weight instead of splitting across
  // two at half strength — which is what made the three weights (line, axis,
  // bracket) indistinguishable in a screenshot. Lines are snapped
  // *individually*, so the error is bounded at half a pixel and never
  // accumulates: a measurement across four divisions is out by under 0.15%.
  let index = round(offset / division);
  let distance = abs(offset - round(index * division));
  // The outermost horizontal pair coincides with the viewport's top and bottom
  // edges; the corner brackets state the frame instead of a drawn box (§16.3).
  let limit = g.edge.z;

  var paint = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  // Division lines, everything but the two centre axes.
  var grid = 0.0;
  if (abs(index.x) > 0.5) {
    grid = max(grid, hairline(distance.x, half_width));
  }
  if (abs(index.y) > 0.5 && abs(index.y) < limit + 0.5) {
    grid = max(grid, hairline(distance.y, half_width));
  }
  paint = over(paint, g.line.rgb, grid * g.line.a);

  // The two centre axes, and — only on them — five subdivisions per division.
  // Verbatim 475A: "horizontal and vertical centerlines further marked in
  // 0.2 cm increments"; nothing else carries minor ticks.
  let subdivision = division / max(g.marks.x, 1.0);
  let tick_index = round(offset / subdivision);
  let tick_distance = abs(offset - round(tick_index * subdivision));
  let fifth = abs(tick_index - round(tick_index / 5.0) * 5.0) < vec2<f32>(0.5);
  let arm = select(vec2<f32>(g.marks.y), vec2<f32>(g.marks.z), fifth);

  var axis = 0.0;
  if (abs(index.x) < 0.5) {
    axis = max(axis, hairline(distance.x, half_width));
  }
  if (abs(index.y) < 0.5) {
    axis = max(axis, hairline(distance.y, half_width));
  }
  // Ticks on the horizontal axis stand off it vertically, and vice versa.
  axis = max(axis, hairline(tick_distance.x, half_width) * band(abs(offset.y), -1.0, arm.x));
  axis = max(axis, hairline(tick_distance.y, half_width) * band(abs(offset.x), -1.0, arm.y));
  paint = over(paint, g.axis.rgb, axis * g.axis.a);

  // Four corner brackets: the frame stated four times rather than drawn.
  let inset = g.edge.y;
  let weight = g.edge.x;
  let reach = g.marks.w;
  let near = min(frag.xy, size - frag.xy);
  let bracket = max(
    band(near.x, inset, inset + weight) * band(near.y, inset - 1.0, inset + reach),
    band(near.y, inset, inset + weight) * band(near.x, inset - 1.0, inset + reach),
  );
  return over(paint, g.bracket.rgb, bracket * g.bracket.a);
}
