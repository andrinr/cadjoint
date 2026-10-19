/**
 * The viewer's shaders for a scene graph, built in the browser: the fields
 * the compile payload carries today (`sdf`, `shader`, `scene_wgsl`,
 * `preview_shader`, `path_shader`, `present_shader`, `program`,
 * `shader_hash`), assembled the way `cadjoint/viewer/worker/main.py` does
 * with the direct backend in its uniform form.
 */
import {
  compileGraphDirect,
  DEFAULT_PARAMETER_BINDING,
  DEFAULT_PARAMETER_GROUP,
  PARAMETER_SLOT_BYTES,
  RESERVED_PARAMETER_SLOTS,
} from "./direct.ts";
import type { SceneGraph } from "./graph.ts";
import {
  DEFAULT_MATERIAL_WGSL,
  SCENE_MARKER,
  SDF_MARKER,
  WGSL_PATH_TRACER_TEMPLATE,
  WGSL_PRESENT_TEMPLATE,
  WGSL_VIEWER_TEMPLATE,
} from "./templates.generated.ts";

/** `_webgpu.ensure_material_wgsl`. */
export function ensureMaterialWgsl(sceneCode: string): string {
  const hasBase = sceneCode.includes("fn material_base(");
  const hasOptics = sceneCode.includes("fn material_optics(");
  if (hasBase !== hasOptics) throw new Error("Scene source must define both material WGSL functions or neither");
  return hasBase ? sceneCode : `${sceneCode}\n\n${DEFAULT_MATERIAL_WGSL}`;
}

// `String.replace` with a string pattern would read `$&` and friends in the
// replacement; scene code is data, so it goes in through a function.

/** `_webgpu.build_viewer_shader`. */
export function buildViewerShader(sceneCode: string): string {
  if (sceneCode.includes(SDF_MARKER)) throw new Error(`SDF source cannot contain the reserved marker '${SDF_MARKER}'`);
  const code = ensureMaterialWgsl(sceneCode);
  return WGSL_VIEWER_TEMPLATE.replace(SDF_MARKER, () => code);
}

/** `_pathtracer.build_path_tracer_shader`. */
export function buildPathTracerShader(sceneCode: string): string {
  if (sceneCode.includes(SCENE_MARKER)) throw new Error(`Scene source cannot contain the reserved marker '${SCENE_MARKER}'`);
  return WGSL_PATH_TRACER_TEMPLATE.replace(SCENE_MARKER, () => sceneCode);
}

/** One entry of the program's parameter list, as `ShaderParameter.as_dict` writes it. */
export type ProgramParameter = {
  name: string;
  offset: number;
  components: number;
  value: (number | null)[];
  free: boolean;
  depends?: string[];
};

/** `ShaderProgram.as_dict`. */
export type ProgramContract = {
  group: number;
  binding: number;
  buffer_bytes: number;
  nan_offset: number;
  cull_margin_offset: number | null;
  parameters: ProgramParameter[];
};

export type SceneShaders = {
  sdf: string;
  shader: string;
  scene_wgsl: string;
  preview_shader: string;
  path_shader: string;
  present_shader: string;
  program: ProgramContract;
  shader_hash: string;
};

/** The scene module and its buffer contract, as the worker's `_direct_shader` builds them. */
export function sceneProgram(graph: SceneGraph, culling = true): { wgsl: string; program: ProgramContract } {
  const direct = compileGraphDirect(graph, { uniforms: true, culling });
  const derived = new Set(direct.derived);
  const parameters = direct.parameters.map(([name, components, values], index): ProgramParameter => {
    const entry: ProgramParameter = {
      name,
      offset: index * PARAMETER_SLOT_BYTES,
      components,
      value: values.map((v) => (Number.isFinite(v) ? v : null)),
      free: !derived.has(name),
    };
    const depends = direct.depends[name];
    if (depends?.length) entry.depends = depends;
    return entry;
  });
  const slots = parameters.length;
  return {
    wgsl: direct.wgsl,
    program: {
      group: DEFAULT_PARAMETER_GROUP,
      binding: DEFAULT_PARAMETER_BINDING,
      buffer_bytes: (slots + RESERVED_PARAMETER_SLOTS) * PARAMETER_SLOT_BYTES,
      nan_offset: slots * PARAMETER_SLOT_BYTES,
      cull_margin_offset: (slots + 1) * PARAMETER_SLOT_BYTES,
      parameters,
    },
  };
}

async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Every shader field of the compile payload, from the graph alone. */
export async function sceneShaders(graph: SceneGraph, culling = true): Promise<SceneShaders> {
  const { wgsl, program } = sceneProgram(graph, culling);
  const preview = buildViewerShader(wgsl);
  const path = buildPathTracerShader(wgsl);
  return {
    sdf: wgsl,
    shader: preview,
    scene_wgsl: wgsl,
    preview_shader: preview,
    path_shader: path,
    present_shader: WGSL_PRESENT_TEMPLATE,
    program,
    shader_hash: await sha256Hex(`${preview}\0${path}`),
  };
}
