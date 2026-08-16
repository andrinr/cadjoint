// WGSL for the FEM simulation surface.
//
// Draws the boundary of the solved hex mesh as an indexed triangle list with
// one scalar per vertex (temperature or von Mises stress). The fragment stage
// maps the scalar through a baked viridis ramp, shades with a screen-space
// derivative normal (flat facets, no normal buffer needed), and supports a
// ParaView-style clip plane: fragments beyond the plane are discarded so the
// interior of the field is visible along any axis.

struct SimUniforms {
  view_proj : mat4x4<f32>,
  // Clip plane: fragments with dot(world, xyz) > w are discarded.
  clip      : vec4<f32>,
  // min scalar, 1 / (max - min), hover tint strength, clip enabled flag.
  params    : vec4<f32>,
};
@group(0) @binding(0) var<uniform> s: SimUniforms;

struct SimVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) world          : vec3<f32>,
  @location(1) scalar         : f32,
};

@vertex
fn vs_sim(
  @location(0) position : vec3<f32>,
  @location(1) scalar   : f32,
) -> SimVertex {
  var out: SimVertex;
  out.position = s.view_proj * vec4<f32>(position, 1.0);
  out.world = position;
  out.scalar = scalar;
  return out;
}

// Polynomial fit of the matplotlib viridis colormap (degree 6, per channel).
// The same coefficients drive the legend gradient in simulation.ts — keep the
// two lists in sync.
fn viridis(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.2744554245, 0.0057679624, 0.3326638811);
  let c1 = vec3<f32>(0.1077083262, 1.3964696839, 1.3867705979);
  let c2 = vec3<f32>(-0.3272410968, 0.2148135645, 0.0919768808);
  let c3 = vec3<f32>(-4.5999315182, -5.7582381893, -19.2918089503);
  let c4 = vec3<f32>(6.2037359013, 14.1539649474, 56.6562995652);
  let c5 = vec3<f32>(4.7517868889, -13.7494394044, -65.3209678276);
  let c6 = vec3<f32>(-5.4320771710, 4.6415713160, 26.2721076045);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

@fragment
fn fs_sim(input: SimVertex) -> @location(0) vec4<f32> {
  if (s.params.w > 0.5 && dot(input.world, s.clip.xyz) > s.clip.w) {
    discard;
  }
  let t = clamp((input.scalar - s.params.x) * s.params.y, 0.0, 1.0);
  var color = clamp(viridis(t), vec3<f32>(0.0), vec3<f32>(1.0));
  // Hover highlight: pull the group's faces toward a warm tint.
  color = mix(color, vec3<f32>(1.0, 0.62, 0.25), s.params.z);
  // Flat facet shading from screen-space derivatives; clipped cross sections
  // keep their facet normal too, which reads like a real cut.
  let normal = normalize(cross(dpdx(input.world), dpdy(input.world)));
  let light = normalize(vec3<f32>(0.55, 0.8, 0.35));
  let shade = 0.62 + 0.38 * abs(dot(normal, light));
  return vec4<f32>(color * shade, 1.0);
}
