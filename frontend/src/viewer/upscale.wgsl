// The interaction preview, stretched back over the framebuffer.
//
// While a gesture is in flight a slow scene is raymarched into a smaller
// target (`Renderer.gestureScale`); this pass covers the framebuffer with it.
// Colour is filtered. Depth is not — a depth texture cannot be, and an
// averaged depth would be a surface nobody drew — so it is the nearest
// texel's, written back through `frag_depth` so the ground grid and the
// construction overlays that follow test against the solid as they always do.

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var scene_sampler: sampler;

struct UpscaleVertex {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_upscale(@builtin(vertex_index) index: u32) -> UpscaleVertex {
  // One triangle over the whole viewport; uv runs 0..1 across it, top-left
  // first, the way texture coordinates do.
  let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
  var out: UpscaleVertex;
  out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  out.uv = uv;
  return out;
}

struct UpscaleFragment {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_upscale(in: UpscaleVertex) -> UpscaleFragment {
  let size = textureDimensions(scene_depth);
  let texel = min(vec2<u32>(in.uv * vec2<f32>(size)), size - vec2<u32>(1u));
  var out: UpscaleFragment;
  out.color = textureSampleLevel(scene_color, scene_sampler, in.uv, 0.0);
  out.depth = textureLoad(scene_depth, texel, 0);
  return out;
}
