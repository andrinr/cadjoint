import { describe, expect, it } from "vitest";
import {
  brepModel,
  colourFaces,
  edgeLegend,
  edgeRgba,
  edgeTone,
  exportDetail,
  exportLabel,
  packBrepEdges,
  packBrepFaces,
  packBrepTriangleOutlines,
  packBrepVertices,
  pickBrep,
  pickFace,
  type BrepModel,
  type BrepStyle,
} from "../src/brep";
import { BREP_FACE_PALETTE } from "../src/tokens";
import type { BRepEdgePayload, BRepInspectionPayload } from "../src/types";
import { projectPoint, type View } from "../src/viewer/math";

const VIEW: View = { position: [0, 0, 5], target: [0, 0, 0], width: 800, height: 600 };

function edge(id: number, kind: string, faces: [number, number], points: number[], extra: Partial<BRepEdgePayload> = {}): BRepEdgePayload {
  return {
    id,
    kind,
    state: kind === "traced" ? "crossing" : kind,
    faces,
    patches: [faces[0], faces[1]],
    vertices: [-1, -1],
    closed: false,
    analytic: true,
    residual: 0,
    crossing: 1,
    points,
    off_surface: 0,
    ...extra,
  };
}

/**
 * Two unit squares side by side in z = 0 (faces 0 and 1), a third square
 * behind them in z = -1 (face 2), the shared edge between 0 and 1, and a
 * vertex at its top.
 */
function payload(): BRepInspectionPayload {
  return {
    available: true,
    positions: [
      -1, -0.5, 0, 0, -0.5, 0, 0, 0.5, 0, -1, 0.5, 0,
      0, -0.5, 0, 1, -0.5, 0, 1, 0.5, 0, 0, 0.5, 0,
      -2, -2, -1, 2, -2, -1, 2, 2, -1, -2, 2, -1,
    ],
    triangles: [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11],
    triangle_face: [0, 0, 1, 1, 2, 2],
    faces: [0, 1, 2].map((id) => ({
      id,
      kind: "plane",
      surface: "plane",
      analytic: true,
      patch: id,
      leaf: 0,
      leaf_type: "Box",
      area: 1,
    })),
    edges: [
      edge(0, "traced", [0, 1], [0, -0.5, 0, 0, 0.5, 0]),
      edge(1, "blend", [1, 2], [1, -0.5, 0, 1, 0.5, 0], { off_surface: 0.5 }),
      edge(2, "a-new-state", [0, 2], [-1, -0.5, 0, -1, 0.5, 0]),
    ],
    vertices: [{ id: 0, kind: "corner", point: [0, 0.5, 0], faces: [0, 1, 2], residual: 0 }],
    diagnostics: {
      faces: 3,
      edges: 3,
      vertices: 1,
      source: "analytic",
      exact: true,
      fallback_reason: null,
      surfaces: { plane: 3 },
      edge_kinds: { traced: 1, blend: 1, "a-new-state": 1 },
      vertex_kinds: { corner: 1 },
      off_surface_tolerance: 0.01,
      off_surface_edges: 1,
      feature_curves: null,
      cell: 0.1,
      resolution: 64,
      stats: {},
      timings: {},
    },
  };
}

const model = (): BrepModel => brepModel(payload())!;
const style = (patch: Partial<BrepStyle> = {}): BrepStyle => ({
  hiddenKinds: new Set(),
  offSurfaceOnly: false,
  hover: null,
  ...patch,
});
const pixel = (point: [number, number, number]) => {
  const p = projectPoint(point, VIEW);
  return [p.x, p.y] as const;
};

describe("brepModel", () => {
  it("is null for a missing tier, a failed extraction, or a partial payload", () => {
    expect(brepModel(null)).toBeNull();
    expect(brepModel({ available: false, reason: "no diff-brep" })).toBeNull();
    expect(brepModel({ available: true, error: "IndexError" })).toBeNull();
    expect(brepModel({ ...payload(), faces: null })).toBeNull();
    expect(brepModel(payload())?.faces).toHaveLength(3);
  });
});

describe("colourFaces", () => {
  it("never gives two faces across an edge the same colour while the palette allows", () => {
    const colours = colourFaces(model());
    for (const e of model().edges) {
      expect(colours.get(e.faces[0])).not.toBe(colours.get(e.faces[1]));
    }
  });

  it("keeps corner neighbours apart too, and is deterministic", () => {
    const colours = colourFaces(model());
    expect(new Set(colours.values()).size).toBe(3);
    expect([...colourFaces(model())]).toEqual([...colours]);
  });

  it("separates every pair of a dense ring of faces with a big enough palette", () => {
    // Twelve faces, each edge-adjacent to its two ring neighbours and the hub.
    const faces = Array.from({ length: 13 }, (_, id) => ({ ...model().faces[0], id }));
    const edges = [] as BRepEdgePayload[];
    for (let i = 1; i <= 12; i++) {
      edges.push(edge(edges.length, "traced", [0, i], []));
      edges.push(edge(edges.length, "traced", [i, (i % 12) + 1], []));
    }
    const colours = colourFaces({ faces, edges, vertices: [] });
    for (const e of edges) expect(colours.get(e.faces[0])).not.toBe(colours.get(e.faces[1]));
    expect(Math.max(...colours.values())).toBeLessThan(BREP_FACE_PALETTE.length);
  });
});

describe("edge tones", () => {
  it("passes an unknown kind through under its own name, in the fallback tone", () => {
    const tone = edgeTone("a-new-state");
    expect(tone.known).toBe(false);
    expect(tone.label).toBe("a-new-state");
    expect(tone.rgb).toEqual(edgeTone("other").rgb);
  });

  it("lists known kinds first, in table order, then unknown ones, with counts", () => {
    expect(edgeLegend(model()).map((entry) => [entry.kind, entry.count])).toEqual([
      ["traced", 1],
      ["blend", 1],
      ["a-new-state", 1],
    ]);
  });

  it("hides a kind switched off in the legend, and dims all but off-surface edges", () => {
    const m = model();
    expect(edgeRgba(m.edges[0], m, style({ hiddenKinds: new Set(["traced"]) }))).toBeNull();
    const dimmed = edgeRgba(m.edges[0], m, style({ offSurfaceOnly: true }))!;
    const offending = edgeRgba(m.edges[1], m, style({ offSurfaceOnly: true }))!;
    expect(dimmed[3]).toBeLessThan(0.5);
    expect(offending[3]).toBe(1);
  });
});

describe("packing", () => {
  it("writes three unshared vertices per triangle, coloured by face", () => {
    const m = model();
    const colours = colourFaces(m);
    const faces = packBrepFaces(m, colours, null);
    expect(faces.length).toBe(6 * 3 * 10);
    const rgbOf = (triangle: number) => Array.from(faces.slice(triangle * 30 + 3, triangle * 30 + 6));
    expect(rgbOf(0)).toEqual(rgbOf(1));
    expect(rgbOf(0)).not.toEqual(rgbOf(2));
    const hovered = packBrepFaces(m, colours, { kind: "face", id: 0 });
    expect(hovered[3]).toBeGreaterThan(faces[3] - 1e-9);
  });

  it("carries each corner's normal, or zeros so the facet is shaded flat", () => {
    const flat = packBrepFaces(model(), new Map(), null);
    expect(Array.from(flat.slice(7, 10))).toEqual([0, 0, 0]);
    const triangles = payload().triangles!.length;
    const normals = Array.from({ length: triangles * 3 }, (_, i) => (i % 3 === 2 ? 1 : 0));
    const smooth = brepModel({ ...payload(), corner_normals: normals })!;
    expect(smooth.cornerNormals).toHaveLength(triangles * 3);
    const packed = packBrepFaces(smooth, new Map(), null);
    expect(Array.from(packed.slice(7, 10))).toEqual([0, 0, 1]);
    // A normal list of the wrong length is not trusted.
    expect(brepModel({ ...payload(), corner_normals: [0, 0, 1] })!.cornerNormals).toBeNull();
  });

  it("outlines each distinct triangle edge once, under the graph's edges", () => {
    const m = model();
    const triangles = m.triangles;
    const keys = new Set<string>();
    for (let t = 0; t < triangles.length; t += 3) {
      for (let k = 0; k < 3; k++) {
        const a = triangles[t + k];
        const b = triangles[t + ((k + 1) % 3)];
        if (a !== b) keys.add(`${Math.min(a, b)}-${Math.max(a, b)}`);
      }
    }
    expect(packBrepTriangleOutlines(m).length).toBe(keys.size * 10);
    const plain = packBrepEdges(m, style());
    const outlined = packBrepEdges(m, style({ showTriangles: true }));
    expect(outlined.length).toBe(plain.length + keys.size * 10);
    // The graph's edges come last, so they draw over the outlines.
    expect(Array.from(outlined.slice(-plain.length))).toEqual(Array.from(plain));
  });

  it("strokes each polyline as segments, and the hovered edge once more on top", () => {
    const m = model();
    expect(packBrepEdges(m, style()).length).toBe(3 * 10);
    expect(packBrepEdges(m, style({ hover: { kind: "edge", id: 1 } })).length).toBe(4 * 10);
    expect(packBrepEdges(m, style({ hiddenKinds: new Set(["blend"]) })).length).toBe(2 * 10);
    const ring = { ...m, edges: [edge(0, "traced", [0, 1], [0, 0, 0, 1, 0, 0, 0, 1, 0], { closed: true })] };
    expect(packBrepEdges(ring, style()).length).toBe(3 * 10);
  });

  it("packs every vertex as a small filled disc", () => {
    const vertices = packBrepVertices(model(), null);
    expect(vertices.length).toBe(9);
    expect(vertices[8]).toBe(2);
    expect(vertices[7]).toBeLessThan(0);
  });
});

describe("picking", () => {
  it("finds the nearest face under the pointer, not the one behind it", () => {
    const [x, y] = pixel([-0.5, -0.2, 0]);
    expect(pickFace(model(), x, y, VIEW)?.id).toBe(0);
    const [bx, by] = pixel([1.5, 1.5, -1]);
    expect(pickFace(model(), bx, by, VIEW)?.id).toBe(2);
  });

  it("prefers a vertex to an edge and an edge to a face", () => {
    const m = model();
    const [vx, vy] = pixel([0, 0.5, 0]);
    expect(pickBrep(m, vx, vy, VIEW)).toEqual({ kind: "vertex", id: 0 });
    const [ex, ey] = pixel([0, 0, 0]);
    expect(pickBrep(m, ex + 2, ey, VIEW)).toEqual({ kind: "edge", id: 0 });
    const [fx, fy] = pixel([0.5, -0.2, 0]);
    expect(pickBrep(m, fx, fy, VIEW)).toEqual({ kind: "face", id: 1 });
  });

  it("skips an edge whose kind is hidden", () => {
    const [ex, ey] = pixel([0, 0, 0]);
    expect(pickBrep(model(), ex + 2, ey, VIEW, new Set(["traced"]))?.kind).toBe("face");
  });
});

describe("the STEP export readout", () => {
  it("names a fitted face and prints its certificate", () => {
    const fitted = { surface: "bspline", fit_error: 2.3e-5, field_error: 4.1e-5, control_points: 3080, reason: "" } as const;
    expect(exportLabel(fitted)).toBe("B-spline fit");
    expect(exportDetail(fitted)).toBe("fit 2.3e−5 · |f| 4.1e−5 · 3,080 control points");
  });

  it("keeps an exact face's own name and says nothing more", () => {
    const exact = { surface: "cylinder" } as const;
    expect(exportLabel(exact)).toBe("cylinder");
    expect(exportDetail(exact)).toBe("");
  });

  it("calls a plane a flat blend was found to lie in a found plane", () => {
    expect(exportLabel({ surface: "plane", fit_error: 1e-6, field_error: 2e-6 })).toBe("plane (found)");
  });

  it("gives a faceted face's reason", () => {
    const faceted = { surface: "faceted", reason: "a crease inside the face" } as const;
    expect(exportLabel(faceted)).toBe("faceted");
    expect(exportDetail(faceted)).toBe("a crease inside the face");
  });
});
