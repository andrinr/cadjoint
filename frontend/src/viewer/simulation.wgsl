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
  // x: ramp selector (0 = field/viridis, 1 = quality/magma).
  // yzw: camera position in world space. The facet normal comes from
  // screen-space derivatives, whose sign depends on the derivative order, so
  // the eye vector is needed both to orient it and to find the silhouette.
  extra     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> s: SimUniforms;

struct SimVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) world          : vec3<f32>,
  @location(1) scalar         : f32,
  // Per-vertex highlight tint (BC previews): rgb hue + blend strength.
  @location(2) overlay        : vec4<f32>,
};

@vertex
fn vs_sim(
  @location(0) position : vec3<f32>,
  @location(1) scalar   : f32,
  @location(2) overlay  : vec4<f32>,
) -> SimVertex {
  var out: SimVertex;
  out.position = s.view_proj * vec4<f32>(position, 1.0);
  out.world = position;
  out.scalar = scalar;
  out.overlay = overlay;
  return out;
}

// Polynomial fits of the matplotlib colormaps (degree 6, per channel). The
// same coefficient tables drive the legends in src/simColors.ts — keep the
// lists in sync. viridis is the FIELD ramp, magma the QUALITY ramp.
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

fn magma(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(-0.0020666453, -0.0006875655, -0.0095482507);
  let c1 = vec3<f32>(0.2504864448, 0.6944550333, 2.4952869139);
  let c2 = vec3<f32>(8.3459009063, -3.5960313696, 0.3290570684);
  let c3 = vec3<f32>(-27.6669694889, 14.2538530831, -13.6465831585);
  let c4 = vec3<f32>(52.1706837385, -27.9445843529, 12.8810906346);
  let c5 = vec3<f32>(-50.7585722964, 29.0538803789, 4.2699357345);
  let c6 = vec3<f32>(18.6642528253, -11.4900266123, -5.5707689618);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

@fragment
fn fs_sim(input: SimVertex) -> @location(0) vec4<f32> {
  if (s.params.w > 0.5 && dot(input.world, s.clip.xyz) > s.clip.w) {
    discard;
  }
  let t = clamp((input.scalar - s.params.x) * s.params.y, 0.0, 1.0);
  var color: vec3<f32>;
  if (s.extra.x > 0.5) {
    color = clamp(magma(t), vec3<f32>(0.0), vec3<f32>(1.0));
  } else {
    color = clamp(viridis(t), vec3<f32>(0.0), vec3<f32>(1.0));
  }
  // BC-preview tint, blended over the field per vertex before highlights.
  color = mix(color, input.overlay.rgb, clamp(input.overlay.a, 0.0, 1.0));
  // Hover highlight: pull the group's faces toward a warm tint.
  color = mix(color, vec3<f32>(1.0, 0.62, 0.25), s.params.z);
  // Flat facet shading from screen-space derivatives; clipped cross sections
  // keep their facet normal too, which reads like a real cut.
  //
  // On the paper viewport the field can no longer be read by its own
  // brightness: viridis(1) is 1.02:1 against `#e6e6e9` and magma(1) is
  // 1.15:1, so a hot region and the ground are the same luminance. Two
  // changes carry the form instead.
  //
  // 1. A signed Lambert term rather than `abs(dot(...))`. `abs` was a
  //    dark-ground trick — it kept every facet emitting — but it folds the
  //    normal sphere in half, so a facet turned away from the key light was
  //    drawn as bright as one turned into it. With the eye vector in
  //    `extra.yzw` the derivative normal can be oriented, so the term can be
  //    signed and faces pointing away can go properly dark. The key light
  //    also moved off the diagonal: at (0.55, 0.8, 0.35) the three visible
  //    faces of an axis-aligned part all sat within 0.15 of each other, and
  //    on paper facet separation is most of what draws the form.
  // 2. A silhouette contour. The outermost sliver of the surface — where the
  //    facet turns away from the eye — is darkened hard. That is the only
  //    thing separating a yellow field from a white ground, and it costs no
  //    colour fidelity anywhere but the rim.
  let eye = normalize(s.extra.yzw - input.world);
  var normal = normalize(cross(dpdx(input.world), dpdy(input.world)));
  normal = normal * sign(dot(normal, eye) + 1e-6);
  let light = normalize(vec3<f32>(0.30, 0.86, 0.42));
  let shade = 0.42 + 0.58 * max(dot(normal, light), 0.0);
  let contour = mix(0.32, 1.0, smoothstep(0.0, 0.2, abs(dot(normal, eye))));
  return vec4<f32>(color * shade * contour, 1.0);
}

// ── element edges: the mesh's boundary-face edge lines ───────────────────
// Same vertex buffer as the surface (position at offset 0), drawn as a
// line list over its own index buffer. A small clip-space nudge keeps the
// hairlines in front of their own faces.

struct SimEdgeVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) world          : vec3<f32>,
  @location(1) scalar         : f32,
};

@vertex
fn vs_sim_edge(
  @location(0) position : vec3<f32>,
  @location(1) scalar   : f32,
) -> SimEdgeVertex {
  var out: SimEdgeVertex;
  out.position = s.view_proj * vec4<f32>(position, 1.0);
  // A larger nudge than the faces get: hairlines that z-fight with their
  // own surface read as dashed, which looks like missing elements.
  out.position.z = out.position.z - 0.004 * out.position.w;
  out.world = position;
  out.scalar = scalar;
  return out;
}

@fragment
fn fs_sim_edge(input: SimEdgeVertex) -> @location(0) vec4<f32> {
  if (s.params.w > 0.5 && dot(input.world, s.clip.xyz) > s.clip.w) {
    discard;
  }
  // A single fixed hairline colour cannot stay legible across a whole ramp:
  // charcoal vanishes in viridis' dark end, white vanishes in its bright
  // end. Evaluate the same ramp the surface uses and pick whichever of the
  // two edge tones contrasts with it.
  let t = clamp((input.scalar - s.params.x) * s.params.y, 0.0, 1.0);
  var under = viridis(t);
  if (s.extra.x > 0.5) {
    under = magma(t);
  }
  let luminance = dot(clamp(under, vec3<f32>(0.0), vec3<f32>(1.0)),
                      vec3<f32>(0.2126, 0.7152, 0.0722));
  // ELEMENT_EDGE_COLOR / ELEMENT_EDGE_COLOR_LIGHT — keep in sync with
  // src/simColors.ts.
  if (luminance > 0.32) {
    return vec4<f32>(0.05, 0.05, 0.06, 1.0);
  }
  return vec4<f32>(0.86, 0.90, 0.94, 1.0);
}
