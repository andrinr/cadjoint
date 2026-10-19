/**
 * Choosing between the server's shaders and ones built from the scene graph.
 * The server's are the reference: the browser's are drawn only when their
 * hash is the server's.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import type { SceneGraph } from "../src/scene/graph.ts";
import { payloadShaders } from "../src/scene/payloadShaders.ts";
import { sceneShaders } from "../src/scene/shaders.ts";
import type { CompilePayload } from "../src/types";

const fixture: { cases: { label: string; uniforms: boolean; graph: SceneGraph }[] } = JSON.parse(
  readFileSync(new URL("./fixtures/scene-direct.json", import.meta.url), "utf8"),
);
const graph = fixture.cases.find((c) => c.label === "covered" && c.uniforms)!.graph;

async function serverPayload() {
  const built = await sceneShaders(graph);
  return {
    preview_shader: built.preview_shader,
    path_shader: built.path_shader,
    present_shader: built.present_shader,
    program: built.program,
    shader_hash: built.shader_hash,
    scene_graph: graph as CompilePayload["scene_graph"],
  };
}

describe("payloadShaders", () => {
  it("uses the server's shaders by default", async () => {
    const result = await payloadShaders(await serverPayload());
    expect(result.origin).toBe("server");
  });

  it("uses the browser's when asked and they hash the same", async () => {
    const payload = await serverPayload();
    const result = await payloadShaders(payload, "browser");
    expect(result.origin).toBe("browser");
    expect(result.preview).toBe(payload.preview_shader);
  });

  it("falls back to the server's on a hash mismatch, and says so", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const payload = { ...(await serverPayload()), shader_hash: "0".repeat(64) };
    const result = await payloadShaders(payload, "browser");
    expect(result.origin).toBe("server");
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it("falls back to the server's without a graph or with a node it cannot emit", async () => {
    const payload = await serverPayload();
    expect((await payloadShaders({ ...payload, scene_graph: null }, "browser")).origin).toBe("server");
    const xor = { format: "cadjoint-scene", version: 2, root: { type: "xor", params: {}, children: [], id: "n0" } };
    const unsupported = { ...payload, scene_graph: xor as CompilePayload["scene_graph"] };
    expect((await payloadShaders(unsupported, "browser")).origin).toBe("server");
  });
});
