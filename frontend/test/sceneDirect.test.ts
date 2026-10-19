/**
 * The scene-graph WGSL emitter against the Python module it ports.
 *
 * `test/fixtures/scene-direct.json` holds graphs and the module
 * `cadjoint.backends.wgsl.direct` writes for each, so this runs with no
 * Python; `tests/backends/test_wgsl_frontend.py` keeps the fixture current
 * and runs both modules on a device.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { compileGraphDirect, UnsupportedNode, type Slot } from "../src/scene/direct.ts";
import type { SceneGraph, SceneNode } from "../src/scene/graph.ts";
import { formatG9 } from "../src/scene/pyformat.ts";
import { sceneProgram } from "../src/scene/shaders.ts";

type Case = {
  label: string;
  graph: SceneGraph;
  uniforms: boolean;
  culling: boolean;
  wgsl: string;
  parameters: Slot[];
  derived: string[];
  depends: Record<string, string[]>;
};

const fixture: { cases: Case[] } = JSON.parse(
  readFileSync(new URL("./fixtures/scene-direct.json", import.meta.url), "utf8"),
);

describe("compileGraphDirect", () => {
  for (const c of fixture.cases) {
    it(`writes Python's module for ${c.label} (uniforms ${c.uniforms}, culling ${c.culling})`, () => {
      const program = compileGraphDirect(c.graph, { uniforms: c.uniforms, culling: c.culling });
      expect(program.wgsl).toBe(c.wgsl);
      expect(program.parameters.map(([n, k]) => [n, k])).toEqual(c.parameters.map(([n, k]) => [n, k]));
      const derived = new Set(c.derived);
      program.parameters.forEach(([name, , value], i) => {
        const expected = c.parameters[i][2];
        if (!derived.has(name)) expect(value).toEqual(expected);
        // Culling boxes: float32 arithmetic whose last ulp JAX may round differently.
        else value.forEach((v, j) => expect(Math.abs(v - expected[j])).toBeLessThanOrEqual(1e-6 * (1 + Math.abs(v))));
      });
      expect(program.derived).toEqual(c.derived);
      expect(program.depends).toEqual(c.depends);
    });
  }

  it("emits a shared subtree once", () => {
    const covered = fixture.cases.find((c) => c.label === "covered" && c.uniforms)!;
    const ids = new Set<string>();
    let repeats = 0;
    (function walk(node: SceneNode) {
      if (ids.has(node.id!)) repeats++;
      ids.add(node.id!);
      node.children.forEach(walk);
    })(covered.graph.root);
    expect(repeats).toBeGreaterThan(0);
    const functions = compileGraphDirect(covered.graph, { uniforms: true, culling: true }).wgsl.match(/^fn node\d+/gm)!;
    expect(functions.length).toBe(ids.size);
  });

  it("reads a version 1 graph as one without free parameters or materials", () => {
    const covered = fixture.cases.find((c) => c.label === "covered" && c.uniforms)!;
    const strip = (node: SceneNode): SceneNode => ({ type: node.type, params: node.params, children: node.children.map(strip) });
    const v1: SceneGraph = { format: "cadjoint-scene", version: 1, root: strip(covered.graph.root) };
    const program = compileGraphDirect(v1, { uniforms: true, culling: true });
    expect(program.parameters.filter(([name]) => !name.startsWith("__cull") && !name.startsWith("sketch"))).toEqual([]);
    expect(program.wgsl).toContain("mat4x2<f32>(vec2<f32>(1, 0), vec2<f32>(1, 1), vec2<f32>(1, 1), vec2<f32>(0.5, 0))");
  });

  it("refuses a node it has no kernel for", () => {
    const graph: SceneGraph = {
      format: "cadjoint-scene",
      version: 2,
      root: { type: "xor", params: {}, children: [], id: "n0" },
    };
    expect(() => compileGraphDirect(graph)).toThrow(UnsupportedNode);
  });

  it("refuses an unknown version", () => {
    const covered = fixture.cases[0];
    expect(() => compileGraphDirect({ ...covered.graph, version: 3 })).toThrow(/v3/);
  });
});

describe("sceneProgram", () => {
  it("lays the buffer out as the worker does", () => {
    const covered = fixture.cases.find((c) => c.label === "covered" && c.uniforms)!;
    const { program } = sceneProgram(covered.graph);
    const slots = covered.parameters.length;
    expect(program.buffer_bytes).toBe((slots + 2) * 16);
    expect(program.nan_offset).toBe(slots * 16);
    expect(program.cull_margin_offset).toBe((slots + 1) * 16);
    expect(program.parameters.map((p) => p.offset)).toEqual(covered.parameters.map((_, i) => i * 16));
    const boxes = program.parameters.filter((p) => !p.free);
    expect(boxes.map((p) => p.name)).toEqual(covered.derived);
    expect(program.parameters.find((p) => p.name === "radius")?.free).toBe(true);
  });
});

describe("formatG9", () => {
  it.each([
    [0, "0"],
    [-0, "-0"],
    [1, "1"],
    [0.1, "0.1"],
    [1e-5, "1e-05"],
    [1.5e-4, "0.00015"],
    [123456789, "123456789"],
    [1234567895, "1.2345679e+09"],
    [1234567885, "1.23456788e+09"],
    [2 / 3, "0.666666667"],
    [-6.283185307179586, "-6.28318531"],
    [1e30, "1e+30"],
    [NaN, "nan"],
    [Infinity, "inf"],
  ])("spells %s as Python does", (x, expected) => {
    expect(formatG9(x)).toBe(expected);
  });
});
