/**
 * The demo's bundle as the page loads it: every shipped scene parses, its
 * graph builds a shader, and the overlay data joins the parameter list.
 */
import { describe, expect, it } from "vitest";
import { sceneProgram } from "../src/scene/shaders.ts";
import { PUBLIC_TIER, SCENE_NAMES, loadScene, parseDemoScene } from "../src/static/bundle.ts";
import { meshEdges } from "../src/static/edges.ts";

describe("the bundle", () => {
  it("ships the scenes and the public tier", () => {
    expect(SCENE_NAMES).toEqual(expect.arrayContaining(["bracket", "motor_shield", "starter"]));
    expect(PUBLIC_TIER.installed).toBe(false);
    expect(Object.values(PUBLIC_TIER.kinds).every((kind) => !kind.available && kind.message)).toBe(true);
    expect(PUBLIC_TIER.kinds.brep.label).toBe("Dual contouring · public tier");
  });

  it("refuses a file of another format or version", () => {
    expect(() => parseDemoScene('{"format": "cadjoint-demo", "version": 2}')).toThrow(/v2/);
    expect(() => parseDemoScene('{"format": "cadjoint-scene", "version": 1}')).toThrow(/cadjoint-scene/);
  });

  for (const name of SCENE_NAMES) {
    it(`${name}: loads, builds its shader, and joins its handles to its parameters`, async () => {
      const scene = await loadScene(name);
      expect(scene.name).toBe(name);
      expect(scene.title.length).toBeGreaterThan(0);
      const { program } = sceneProgram(scene.scene_graph);
      const free = new Set(program.parameters.filter((s) => s.free).map((s) => s.name));
      const listed = new Map(scene.parameters.map((p) => [p.name, p]));
      for (const name of free) if (program.parameters.find((s) => s.name === name)!.value.every((v) => v !== null))
        expect(listed.has(name), `${name} is listed`).toBe(true);
      for (const parameter of scene.parameters) {
        expect(parameter.lower.length).toBe(parameter.components);
        parameter.value.forEach((v, i) => {
          expect(v).toBeGreaterThanOrEqual(parameter.lower[i]);
          expect(v).toBeLessThanOrEqual(parameter.upper[i]);
        });
        if (parameter.owner) expect(scene.construction.some((n) => n.id === parameter.owner)).toBe(true);
      }
      // Every bound vertex names a parameter the panel lists or one the shader folded away.
      for (const node of scene.construction)
        for (const vertex of node.vertices)
          if (vertex.binding && free.has(vertex.binding.name)) expect(listed.has(vertex.binding.name)).toBe(true);
    });
  }
});

describe("mesh edges", () => {
  // A unit cube as dual contouring emits it: six quads, each as two consecutive triangles.
  const vertices = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
  const quads = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 4, 7, 3],
  ];
  const triangles = new Uint32Array(quads.flatMap(([a, b, c, d]) => [a, b, c, a, c, d]));

  it("draws the quads without their diagonals, and every cube edge is a crease", () => {
    const edges = meshEdges(vertices, triangles, 8);
    expect(edges.wire).toHaveLength(12);
    expect(edges.sharp).toHaveLength(12);
    expect(edges.edges).toBe("lattice");
    expect(edges.resolution).toBe(8);
  });

  it("a flat fold below the threshold is no crease", () => {
    expect(meshEdges(vertices, triangles, 8, 95).sharp).toHaveLength(0);
  });
});
