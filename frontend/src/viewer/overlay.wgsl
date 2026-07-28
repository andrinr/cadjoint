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

struct HandleVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) local          : vec2<f32>,
  @location(1) color          : vec4<f32>,
  @location(2) emphasis       : f32,
};

@vertex
fn vs_handle(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) center   : vec3<f32>,
  @location(1) color    : vec4<f32>,
  @location(2) emphasis : f32,
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
  return vec4<f32>(shade, input.color.a * edge);
}
