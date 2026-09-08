import { describe, expect, it } from "vitest";
import { visiblePlaneFrames, type PlaneFrame, type PlaneVisibility } from "../src/planes";

const frame = (nodeId: string): PlaneFrame =>
  ({ nodeId, label: nodeId, derived: false }) as unknown as PlaneFrame;

const frames = [frame("pad"), frame("boss"), frame("rib")];
const sketching: PlaneVisibility = {
  sketching: true,
  selectedNodeId: null,
  hoveredNodeId: null,
  showOverlays: true,
  showSketches: true,
  showAllPlanes: false,
};
const ids = (list: PlaneFrame[]) => list.map((f) => f.nodeId);

describe("which sketch planes are drawn", () => {
  it("draws none outside sketch mode", () => {
    expect(visiblePlaneFrames(frames, { ...sketching, sketching: false, selectedNodeId: "pad" })).toEqual([]);
  });
  it("draws the selected and the hovered sketch's plane while sketching", () => {
    expect(ids(visiblePlaneFrames(frames, { ...sketching, selectedNodeId: "pad" }))).toEqual(["pad"]);
    expect(
      ids(visiblePlaneFrames(frames, { ...sketching, selectedNodeId: "pad", hoveredNodeId: "rib" })),
    ).toEqual(["pad", "rib"]);
    expect(visiblePlaneFrames(frames, sketching)).toEqual([]);
  });
  it("draws every plane, in any mode, with the switch on", () => {
    expect(ids(visiblePlaneFrames(frames, { ...sketching, sketching: false, showAllPlanes: true }))).toEqual([
      "pad",
      "boss",
      "rib",
    ]);
  });
  it("draws none with the overlay or the sketch geometry off, switch or not", () => {
    expect(visiblePlaneFrames(frames, { ...sketching, showAllPlanes: true, showOverlays: false })).toEqual([]);
    expect(visiblePlaneFrames(frames, { ...sketching, showAllPlanes: true, showSketches: false })).toEqual([]);
  });
});
