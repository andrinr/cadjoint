// WGSL for the construction-tree overlay.
// 
// Edges and vertex handles are drawn as screen-space expanded quads so their
// width is in pixels rather than world units, and both are depth-tested against
// the depth the SDF pass writes — an edge behind the solid is correctly hidden.
// 
// Sketch edges usually lie exactly on the extruded solid's side faces, so every
// vertex is nudged a fixed fraction of the way toward the camera before
// projection. That is a scale-free bias: it resolves the coplanar z-fight at
// any zoom without a depth-bias constant tuned per scene.

struct Overlay {
  view_proj  : mat4x4<f32>,
  camera_pos : vec4<f32>,
  // width, height, line width (px), handle radius (px)
  viewport   : vec4<f32>,
  // depth nudge fraction, unused, unused, unused
  params     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> o: Overlay;

fn half_viewport() -> vec2<f32> {
  return max(o.viewport.xy * 0.5, vec2<f32>(1.0));
}

// Pull a point slightly toward the camera so overlay geometry wins the depth
// test against surfaces it is coplanar with.
fn nudge(p: vec3<f32>) -> vec3<f32> {
  return p + (o.camera_pos.xyz - p) * o.params.x;
}

fn project(p: vec3<f32>) -> vec4<f32> {
  var clip = o.view_proj * vec4<f32>(nudge(p), 1.0);
  // Keep points behind the eye from wrapping to the far side of the screen.
  clip.w = max(clip.w, 1e-4);
  return clip;
}

struct EdgeVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) side           : f32,
  @location(1) color          : vec4<f32>,
};

@vertex
fn vs_edge(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) start : vec3<f32>,
  @location(1) end   : vec3<f32>,
  @location(2) color : vec4<f32>,
) -> EdgeVertex {
  // Two triangles spanning the segment: x picks the endpoint, y the side.
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let corner = corners[vertex_index];

  let clip_start = project(start);
  let clip_end = project(end);
  let half_size = half_viewport();
  let screen_start = clip_start.xy / clip_start.w * half_size;
  let screen_end = clip_end.xy / clip_end.w * half_size;

  var direction = screen_end - screen_start;
  let span = length(direction);
  direction = select(vec2<f32>(1.0, 0.0), direction / span, span > 1e-6);
  let normal = vec2<f32>(-direction.y, direction.x);

  let base = select(clip_start, clip_end, corner.x > 0.5);
  let offset = normal * (o.viewport.z * 0.5) * corner.y / half_size;

  var out: EdgeVertex;
  out.position = vec4<f32>(base.xy + offset * base.w, base.z, base.w);
  out.side = corner.y;
  out.color = color;
  return out;
}

@fragment
fn fs_edge(input: EdgeVertex) -> @location(0) vec4<f32> {
  // Feather the outer pixel so lines do not alias into stair steps.
  let coverage = smoothstep(1.0, 0.55, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}

// Translation axes get their own screen-space shape instead of being assembled
// from wireframe segments. The result is a compact filled arrow whose weight is
// stable at every zoom level.
struct GizmoVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) color          : vec4<f32>,
  @location(1) side           : f32,
  @location(2) barycentric    : vec3<f32>,
  @location(3) head           : f32,
};

@vertex
fn vs_gizmo_arrow(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) start    : vec3<f32>,
  @location(1) end      : vec3<f32>,
  @location(2) color    : vec4<f32>,
  @location(3) emphasis : f32,
) -> GizmoVertex {
  let clip_start = project(start);
  let clip_end = project(end);
  let half_size = half_viewport();
  let screen_start = clip_start.xy / clip_start.w * half_size;
  let screen_end = clip_end.xy / clip_end.w * half_size;

  var direction = screen_end - screen_start;
  let span = max(length(direction), 1e-4);
  direction = select(vec2<f32>(1.0, 0.0), direction / span, span > 1e-4);
  let normal = vec2<f32>(-direction.y, direction.x);

  let head_length = min(14.0 + emphasis * 2.0, span * 0.42);
  let head_width = 6.5 + emphasis * 1.5;
  let shaft_width = 2.35 + emphasis * 0.65;
  let start_gap = min(5.0, span * 0.12);
  let head_base = screen_end - direction * head_length;
  let shaft_start = screen_start + direction * start_gap;
  // A tiny overlap prevents a hairline seam between the shaft and head.
  let shaft_end = head_base + direction;

  var point: vec2<f32>;
  var along = 0.0;
  var side = 0.0;
  var barycentric = vec3<f32>(-1.0);
  var head = 0.0;

  if (vertex_index < 6u) {
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vertex_index];
    point = mix(shaft_start, shaft_end, corner.x) + normal * shaft_width * corner.y;
    along = (start_gap + corner.x * max(span - start_gap - head_length + 1.0, 0.0)) / span;
    side = corner.y;
  } else {
    let head_vertex = vertex_index - 6u;
    var points = array<vec2<f32>, 3>(
      screen_end,
      head_base + normal * head_width,
      head_base - normal * head_width,
    );
    var barycentrics = array<vec3<f32>, 3>(
      vec3<f32>(1.0, 0.0, 0.0),
      vec3<f32>(0.0, 1.0, 0.0),
      vec3<f32>(0.0, 0.0, 1.0),
    );
    point = points[head_vertex];
    along = select(1.0 - head_length / span, 1.0, head_vertex == 0u);
    barycentric = barycentrics[head_vertex];
    head = 1.0;
  }

  let clip = mix(clip_start, clip_end, clamp(along, 0.0, 1.0));
  var out: GizmoVertex;
  out.position = vec4<f32>(point / half_size * clip.w, clip.z, clip.w);
  out.color = color;
  out.side = side;
  out.barycentric = barycentric;
  out.head = head;
  return out;
}

@vertex
fn vs_gizmo_scale(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) start    : vec3<f32>,
  @location(1) end      : vec3<f32>,
  @location(2) color    : vec4<f32>,
  @location(3) emphasis : f32,
) -> GizmoVertex {
  let clip_start = project(start);
  let clip_end = project(end);
  let half_size = half_viewport();
  let screen_start = clip_start.xy / clip_start.w * half_size;
  let screen_end = clip_end.xy / clip_end.w * half_size;

  var direction = screen_end - screen_start;
  let span = max(length(direction), 1e-4);
  direction = select(vec2<f32>(1.0, 0.0), direction / span, span > 1e-4);
  let normal = vec2<f32>(-direction.y, direction.x);
  let block_radius = 5.2 + emphasis;
  let shaft_width = 2.2 + emphasis * 0.65;
  let start_gap = min(5.0, span * 0.12);
  let shaft_start = screen_start + direction * start_gap;
  let shaft_end = screen_end - direction * (block_radius - 0.8);

  var point: vec2<f32>;
  var along = 0.0;
  var side = 0.0;
  var local = vec2<f32>(-1.0);
  var head = 0.0;

  if (vertex_index < 6u) {
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vertex_index];
    point = mix(shaft_start, shaft_end, corner.x) + normal * shaft_width * corner.y;
    along = (start_gap + corner.x * max(span - start_gap - block_radius + 0.8, 0.0)) / span;
    side = corner.y;
  } else {
    let block_vertex = vertex_index - 6u;
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    local = corners[block_vertex];
    point = screen_end + (direction * local.x + normal * local.y) * block_radius;
    along = 1.0;
    head = 2.0;
  }

  let clip = mix(clip_start, clip_end, clamp(along, 0.0, 1.0));
  var out: GizmoVertex;
  out.position = vec4<f32>(point / half_size * clip.w, clip.z, clip.w);
  out.color = color;
  out.side = side;
  out.barycentric = vec3<f32>(local, 0.0);
  out.head = head;
  return out;
}

@fragment
fn fs_gizmo_arrow(input: GizmoVertex) -> @location(0) vec4<f32> {
  var border = smoothstep(0.62, 0.86, abs(input.side));
  var coverage = smoothstep(1.0, 0.82, abs(input.side));
  if (input.head > 1.5) {
    let edge = max(abs(input.barycentric.x), abs(input.barycentric.y));
    border = smoothstep(0.64, 0.86, edge);
    coverage = smoothstep(1.0, 0.88, edge);
  } else if (input.head > 0.5) {
    let nearest_edge = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
    border = 1.0 - smoothstep(0.055, 0.14, nearest_edge);
    coverage = 1.0;
  }
  let outline = vec3<f32>(0.025, 0.035, 0.03);
  let shaded = mix(input.color.rgb, outline, border * 0.82);
  return vec4<f32>(shaded, input.color.a * coverage);
}

struct HandleVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) local          : vec2<f32>,
  @location(1) color          : vec4<f32>,
  @location(2) emphasis       : f32,
  // 1 = a free design parameter, drawn filled; 0 = a source constant, drawn
  // as a ring. Same disc, same colour, same radius — only the middle differs.
  @location(3) fill           : f32,
};

@vertex
fn vs_handle(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) center   : vec3<f32>,
  @location(1) color    : vec4<f32>,
  @location(2) emphasis : f32,
  @location(3) fill     : f32,
) -> HandleVertex {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );
  let corner = corners[vertex_index];
  let clip = project(center);
  let radius = o.viewport.w * (1.0 + emphasis * 0.35);
  let offset = corner * radius / half_viewport();

  var out: HandleVertex;
  out.position = vec4<f32>(clip.xy + offset * clip.w, clip.z, clip.w);
  out.local = corner;
  out.color = color;
  out.emphasis = emphasis;
  out.fill = fill;
  return out;
}

@fragment
fn fs_handle(input: HandleVertex) -> @location(0) vec4<f32> {
  let radius = length(input.local);
  if (radius > 1.0) {
    discard;
  }
  // Filled disc with a darker rim so handles read against bright surfaces.
  let edge = smoothstep(1.0, 0.82, radius);
  let rim = smoothstep(0.62, 0.92, radius);
  let shade = mix(input.color.rgb, input.color.rgb * 0.35, rim);
  // A point the shader holds as a constant keeps the ring and loses the
  // middle: the annulus is the rim the filled disc already draws, so the two
  // states are one shape at two weights rather than two marks. The band is
  // wide enough (0.42 of the radius) to survive at handle size.
  let hollow = smoothstep(0.46, 0.62, radius);
  let coverage = mix(hollow, 1.0, input.fill);
  return vec4<f32>(shade, input.color.a * edge * coverage);
}

// ── face highlight ──────────────────────────────────────────────────────────
//
// The one overlay drawn as an actual surface rather than as an expanded quad:
// the polygon of the face under the pointer, triangulated on the CPU, filled
// with a faint achromatic wash. It carries the same camera nudge as the edges,
// so it wins the depth test against the face it is painted on, and it does not
// write depth — the hairline outline drawn straight after it has to land on
// top, not z-fight with the wash it belongs to.

struct FaceVertex {
  @builtin(position) position : vec4<f32>,
  @location(0)       color    : vec4<f32>,
};

@vertex
fn vs_face(@location(0) point: vec3<f32>, @location(1) color: vec4<f32>) -> FaceVertex {
  var out: FaceVertex;
  out.position = project(point);
  out.color = color;
  return out;
}

@fragment
fn fs_face(input: FaceVertex) -> @location(0) vec4<f32> {
  return input.color;
}
