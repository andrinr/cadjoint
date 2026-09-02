import { describe, expect, it } from "vitest";
import {
  addMeshRequest,
  deleteMeshRequest,
  meshArguments,
  qualityHistogram,
  setMeshValueRequest,
  setStudyDomainRequest,
  setStudyMeshRequest,
} from "../src/meshes";
import type { SimMeshPayload, StudyPayload } from "../src/types";

const mesh: SimMeshPayload = {
  kind: "mesh",
  stableId: null,
  index: 0,
  name: "coarse",
  resolution: [20, 14, 12],
  bounds: [-1, -1, -1],
  size: [2, 2, 2],
  padding: 0.1,
  domain: { name: "sink", type: "Extrusion" },
  line: 5,
  span: [40, 120],
  editable: true,
};

const study: StudyPayload = {
  index: 1,
  stableId: null,
  name: "heat",
  kind: "thermal",
  resolution: null,
  bounds: null,
  size: null,
  mesh: null,
  domain: null,
  material: { conductivity: 2 },
  line: 9,
  span: [200, 340],
  editable: true,
  bcs: [],
};

describe("mesh patch request builders", () => {
  it("builds add/delete/set bodies", () => {
    expect(addMeshRequest()).toEqual({ op: "add_mesh" });
    expect(addMeshRequest("fine")).toEqual({ op: "add_mesh", name: "fine" });
    expect(deleteMeshRequest(mesh)).toEqual({ op: "delete_mesh", mesh: 0 });
    expect(setMeshValueRequest(mesh, "resolution", [24, 16, 12])).toEqual({
      op: "set_mesh_value",
      mesh: 0,
      argument: "resolution",
      value: [24, 16, 12],
    });
    expect(setMeshValueRequest(mesh, "padding", 0.2)).toEqual({
      op: "set_mesh_value",
      mesh: 0,
      argument: "padding",
      value: 0.2,
    });
    expect(setMeshValueRequest(mesh, "domain", "slug")).toEqual({
      op: "set_mesh_value",
      mesh: 0,
      argument: "domain",
      value: "slug",
    });
    expect(setMeshValueRequest(mesh, "method", "tet10")).toEqual({
      op: "set_mesh_value",
      mesh: 0,
      argument: "method",
      value: "tet10",
    });
  });

  it("points studies at declared meshes and named domains", () => {
    expect(setStudyMeshRequest(study, "coarse")).toEqual({
      op: "set_study_value",
      study: 1,
      argument: "mesh",
      value: "coarse",
    });
    expect(setStudyDomainRequest(study, "sink")).toEqual({
      op: "set_study_value",
      study: 1,
      argument: "domain",
      value: "sink",
    });
  });

  it("lists editable numeric arguments, dropping absent bounds/size", () => {
    expect(meshArguments(mesh).map((row) => row.key)).toEqual([
      "resolution",
      "padding",
      "bounds",
      "size",
    ]);
    const automatic = { ...mesh, bounds: null, size: null };
    expect(meshArguments(automatic).map((row) => row.key)).toEqual([
      "resolution",
      "padding",
    ]);
  });
});

describe("qualityHistogram", () => {
  it("bins values across the observed range", () => {
    const histogram = qualityHistogram([0, 0.5, 1, 1, 1], 2);
    expect(histogram.min).toBe(0);
    expect(histogram.max).toBe(1);
    // [0, 0.5) → 2 values (0 and 0.5 falls in second half? 0.5/1*2 = 1 → bin 1)
    expect(histogram.counts).toEqual([1, 4]);
    expect(histogram.peak).toBe(4);
  });

  it("puts a constant field into one bin and survives empty input", () => {
    const flat = qualityHistogram([0.7, 0.7, 0.7], 4);
    expect(flat.counts).toEqual([3, 0, 0, 0]);
    expect(flat.min).toBe(0.7);
    expect(flat.max).toBe(0.7);
    expect(qualityHistogram([], 4)).toEqual({ counts: [], min: 0, max: 0, peak: 0 });
  });

  it("keeps the maximum value inside the last bin", () => {
    const histogram = qualityHistogram([0, 0.25, 0.5, 0.75, 1], 4);
    expect(histogram.counts.reduce((sum, count) => sum + count, 0)).toBe(5);
    expect(histogram.counts[3]).toBe(2);
  });
});
