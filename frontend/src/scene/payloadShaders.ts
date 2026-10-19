/**
 * The shaders to install for a compile payload: the ones the server sent, or
 * ones built here from the payload's `scene_graph`.
 *
 * The server's stay the default and the reference. With the browser source
 * selected (`?shaders=browser`, or `localStorage["cadjoint.shaders"] =
 * "browser"`), the graph is turned into WGSL here, and the result is used
 * only when its `shader_hash` is the server's. A mismatch is logged and the
 * server's shaders are drawn, so a divergence shows up as a warning and never
 * as a wrong image.
 */
import type { CompilePayload } from "../types";
import type { Shaders } from "../viewer/display";
import { UnsupportedNode } from "./direct.ts";
import { sceneShaders } from "./shaders.ts";

export type ShaderOrigin = "server" | "browser";

const STORAGE_KEY = "cadjoint.shaders";

/** Where shaders come from in this page: `server` unless the browser source was asked for. */
export function preferredShaderOrigin(): ShaderOrigin {
  try {
    const query = new URLSearchParams(globalThis.location?.search ?? "").get("shaders");
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY);
    return (query ?? stored) === "browser" ? "browser" : "server";
  } catch {
    return "server";
  }
}

type Payload = Pick<
  CompilePayload,
  "preview_shader" | "path_shader" | "present_shader" | "program" | "shader_hash" | "scene_graph"
>;

export type PayloadShaders = Shaders & { origin: ShaderOrigin };

function fromServer(result: Payload): PayloadShaders {
  return {
    preview: result.preview_shader,
    path: result.path_shader,
    present: result.present_shader,
    program: result.program ?? null,
    origin: "server",
  };
}

/** The shaders for `result`, from `origin` when that source can produce the server's. */
export async function payloadShaders(
  result: Payload,
  origin: ShaderOrigin = preferredShaderOrigin(),
): Promise<PayloadShaders> {
  if (origin === "server" || !result.scene_graph) return fromServer(result);
  try {
    const built = await sceneShaders(result.scene_graph);
    const expected = result.shader_hash ?? "";
    if (built.shader_hash !== expected) {
      console.warn(
        `cadjoint: the browser-built shaders differ from the server's ` +
          `(${built.shader_hash.slice(0, 12)} vs ${expected.slice(0, 12)}); drawing the server's.`,
      );
      return fromServer(result);
    }
    return {
      preview: built.preview_shader,
      path: built.path_shader,
      present: built.present_shader,
      program: built.program,
      origin: "browser",
    };
  } catch (error) {
    if (!(error instanceof UnsupportedNode)) throw error;
    return fromServer(result);
  }
}
