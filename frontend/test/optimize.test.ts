import { describe, expect, it } from "vitest";
import {
  advancePlayer,
  deleteOptimizationRequest,
  formatParameterValue,
  frameObjective,
  optimizeRequest,
  parseOptimizeStreamLine,
  playbackFrames,
  setOptimizationValueRequest,
  sparklineCursorPoint,
  splitStreamBuffer,
  sparklineCursorX,
  sparklinePoints,
  startPlayer,
  substituteParameters,
} from "../src/optimize";
import type { OptimizationPayload } from "../src/types";

const optimization: OptimizationPayload = {
  kind: "optimization",
  stableId: null,
  index: 2,
  name: "min-aluminum",
  steps: 30,
  learning_rate: 0.05,
  method: "adam",
  parameters: ["fin_depth", "fin2_tip_l"],
  objective: "aluminum_volume",
  line: 12,
  span: [80, 220],
  editable: true,
};

describe("optimize request builders", () => {
  it("builds the patch body for a numeric argument edit", () => {
    expect(setOptimizationValueRequest(optimization, "steps", 12)).toEqual({
      op: "set_optimization_value",
      optimization: 2,
      argument: "steps",
      value: 12,
    });
    expect(setOptimizationValueRequest(optimization, "learning_rate", 0.1)).toEqual({
      op: "set_optimization_value",
      optimization: 2,
      argument: "learning_rate",
      value: 0.1,
    });
  });

  it("builds the delete body by payload index", () => {
    expect(deleteOptimizationRequest(optimization)).toEqual({
      op: "delete_optimization",
      optimization: 2,
    });
  });

  it("builds the run body, including steps only when overridden", () => {
    expect(optimizeRequest("src", "min-aluminum")).toEqual({
      source: "src",
      name: "min-aluminum",
    });
    expect(optimizeRequest("src", "min-aluminum", 5)).toEqual({
      source: "src",
      name: "min-aluminum",
      steps: 5,
    });
  });
});

describe("substituteParameters", () => {
  const program = [
    "fin_depth = Scalar(1.2, free=True, name='fin_depth')",
    "tip = Vector2(value=[-0.08, 0.85], free=True, name='tip')",
    "anchor = Vector([0.78, 0.0, 0.1], free=True, name='anchor')",
    "scene = extrude(profile, depth=fin_depth)",
  ].join("\n");

  it("rewrites scalar, Vector2, and Vector literals by parameter name", () => {
    const text = substituteParameters(program, {
      fin_depth: 0.75,
      tip: [-0.1, 0.9],
      anchor: [0.5, 0.25, 0.1],
    });
    expect(text).toContain("fin_depth = Scalar(0.75, free=True");
    expect(text).toContain("tip = Vector2(value=[-0.1, 0.9], free=True");
    expect(text).toContain("anchor = Vector([0.5, 0.25, 0.1], free=True");
    // The consuming expression is untouched.
    expect(text).toContain("depth=fin_depth)");
  });

  it("keeps float formatting Python-friendly for integral values", () => {
    expect(formatParameterValue(2)).toBe("2.0");
    expect(formatParameterValue(-0.0000001)).toBe("0.0");
    expect(formatParameterValue([1, -0.5])).toBe("[1.0, -0.5]");
    const text = substituteParameters(program, { fin_depth: 2 });
    expect(text).toContain("fin_depth = Scalar(2.0,");
  });

  it("leaves unknown names alone", () => {
    expect(substituteParameters(program, { missing: 3 })).toBe(program);
  });
});

describe("playbackFrames", () => {
  it("keeps short trajectories whole", () => {
    expect(playbackFrames(0)).toEqual([]);
    expect(playbackFrames(1)).toEqual([0]);
    expect(playbackFrames(5, 16)).toEqual([0, 1, 2, 3, 4]);
  });

  it("thins long trajectories while pinning both ends", () => {
    const frames = playbackFrames(101, 11);
    expect(frames).toHaveLength(11);
    expect(frames[0]).toBe(0);
    expect(frames[frames.length - 1]).toBe(100);
    // Strictly increasing.
    for (let index = 1; index < frames.length; index++) {
      expect(frames[index]).toBeGreaterThan(frames[index - 1]);
    }
  });
});

describe("player state machine", () => {
  it("advances frame by frame and stops at the end", () => {
    let state = startPlayer({ frame: 0, playing: false }, 3);
    expect(state).toEqual({ frame: 0, playing: true });
    state = advancePlayer(state, 3);
    expect(state).toEqual({ frame: 1, playing: true });
    state = advancePlayer(state, 3);
    expect(state).toEqual({ frame: 2, playing: true });
    state = advancePlayer(state, 3);
    expect(state).toEqual({ frame: 2, playing: false });
  });

  it("rewinds when starting from the last frame", () => {
    expect(startPlayer({ frame: 2, playing: false }, 3)).toEqual({
      frame: 0,
      playing: true,
    });
  });

  it("never advances a paused player or an empty trajectory", () => {
    expect(advancePlayer({ frame: 1, playing: false }, 3).frame).toBe(1);
    expect(advancePlayer({ frame: 0, playing: true }, 0).playing).toBe(false);
  });
});

describe("sparkline", () => {
  it("maps a descending history onto the viewbox", () => {
    const points = sparklinePoints([4, 3, 2, 1], 90, 40);
    const pairs = points.split(" ").map((pair) => pair.split(",").map(Number));
    expect(pairs).toHaveLength(4);
    expect(pairs[0][0]).toBe(0);
    expect(pairs[3][0]).toBe(90);
    // First value is the maximum → top of the chart; last → bottom.
    expect(pairs[0][1]).toBeLessThan(pairs[3][1]);
  });

  it("centers a flat history and handles empty input", () => {
    expect(sparklinePoints([], 90, 40)).toBe("");
    const pairs = sparklinePoints([2, 2], 90, 40)
      .split(" ")
      .map((pair) => pair.split(",").map(Number));
    expect(pairs[0][1]).toBeCloseTo(20, 0);
  });

  it("highlights the cursor's sparkline point at the value's height", () => {
    const values = [4, 3, 2, 1];
    const top = sparklineCursorPoint(values, 0, 90, 40)!;
    const bottom = sparklineCursorPoint(values, 3, 90, 40)!;
    expect(top.x).toBe(0);
    expect(bottom.x).toBe(90);
    expect(top.y).toBeLessThan(bottom.y);
    // Out-of-range indices clamp; empty input yields nothing.
    expect(sparklineCursorPoint(values, 99, 90, 40)!.x).toBe(90);
    expect(sparklineCursorPoint([], 0, 90, 40)).toBeNull();
    // A flat run centers vertically.
    expect(sparklineCursorPoint([2, 2], 1, 90, 40)!.y).toBeCloseTo(20, 0);
  });

  it("positions the cursor proportionally", () => {
    expect(sparklineCursorX(0, 11, 100)).toBe(0);
    expect(sparklineCursorX(10, 11, 100)).toBe(100);
    expect(sparklineCursorX(5, 11, 100)).toBe(50);
    expect(sparklineCursorX(0, 1, 100)).toBe(50);
  });

  it("reads a frame's objective defensively", () => {
    const trajectory = [
      { step: 0, objective: 4, parameters: {} },
      { step: 5, objective: 2, parameters: {} },
    ];
    expect(frameObjective(trajectory, 1)).toBe(2);
    expect(frameObjective(trajectory, 7)).toBeNull();
  });
});

describe("optimize NDJSON stream parsing", () => {
  it("splits complete lines from a chunked buffer, keeping the tail", () => {
    const first = splitStreamBuffer('{"event":"progress","step":1}\n{"event":"prog');
    expect(first.lines).toEqual(['{"event":"progress","step":1}']);
    expect(first.rest).toBe('{"event":"prog');
    const second = splitStreamBuffer(first.rest + 'ress","step":2}\n');
    expect(second.lines).toEqual(['{"event":"progress","step":2}']);
    expect(second.rest).toBe("");
  });

  it("drops blank and whitespace-only lines", () => {
    const { lines, rest } = splitStreamBuffer("\n  \n{\"event\":\"done\"}\n");
    expect(lines).toEqual(['{"event":"done"}']);
    expect(rest).toBe("");
  });

  it("parses progress events with optional grad_norm and elapsed", () => {
    const event = parseOptimizeStreamLine(
      '{"event":"progress","step":3,"steps":25,"objective":0.41,"grad_norm":0.02,"elapsed":7.5}',
    );
    expect(event).toEqual({
      kind: "progress",
      step: 3,
      steps: 25,
      objective: 0.41,
      gradNorm: 0.02,
      elapsed: 7.5,
    });
    const sparse = parseOptimizeStreamLine(
      '{"event":"progress","step":1,"steps":10,"objective":1}',
    );
    expect(sparse).toMatchObject({ kind: "progress", gradNorm: null, elapsed: null });
  });

  it("rejects progress lines missing the required numbers", () => {
    expect(
      parseOptimizeStreamLine('{"event":"progress","step":1,"steps":10}'),
    ).toBeNull();
    expect(
      parseOptimizeStreamLine('{"event":"progress","step":"x","steps":10,"objective":1}'),
    ).toBeNull();
  });

  it("unwraps the done event into the classic response shape", () => {
    const event = parseOptimizeStreamLine(
      '{"event":"done","ok":true,"name":"min-aluminum","source":"x = 1","history":[{"step":0,"objective":1,"grad_norm":0.1}]}',
    );
    expect(event?.kind).toBe("done");
    if (event?.kind !== "done") throw new Error("expected done");
    expect(event.response.ok).toBe(true);
    expect(event.response.source).toBe("x = 1");
    expect("event" in event.response).toBe(false);
  });

  it("returns null for untagged lines so a single-JSON body falls back whole", () => {
    // A non-streaming server's response has no "event" discriminator …
    expect(parseOptimizeStreamLine('{"ok":true,"source":"x = 1"}')).toBeNull();
    // … and a pretty-printed body yields unparseable fragments per line.
    expect(parseOptimizeStreamLine("{")).toBeNull();
    expect(parseOptimizeStreamLine('  "ok": true,')).toBeNull();
    expect(parseOptimizeStreamLine("null")).toBeNull();
  });
});
