/**
 * `node frontend/src/scene/cli.ts < request.json`: the TypeScript scene
 * shaders, for the Python parity tests.
 *
 * The request is `{"graph": ..., "mode": "direct" | "shaders", ...}`.
 * `direct` takes `uniforms` and `culling` and answers the `DirectProgram`;
 * `shaders` takes `culling` and answers the compile payload's shader fields.
 * A graph with a node the emitter has no kernel for answers
 * `{"unsupported": message}`.
 */
import { readFileSync } from "node:fs";
import { compileGraphDirect, UnsupportedNode } from "./direct.ts";
import { sceneShaders } from "./shaders.ts";

const request = JSON.parse(readFileSync(0, "utf8"));
try {
  const reply =
    request.mode === "shaders"
      ? await sceneShaders(request.graph, request.culling ?? true)
      : compileGraphDirect(request.graph, { uniforms: request.uniforms, culling: request.culling });
  process.stdout.write(JSON.stringify(reply));
} catch (error) {
  if (!(error instanceof UnsupportedNode)) throw error;
  process.stdout.write(JSON.stringify({ unsupported: error.message }));
}
