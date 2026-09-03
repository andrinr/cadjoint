/**
 * What the app starts on, and whether both sides of the picture agree.
 *
 * "Ultra" used to be a word that meant four things in TypeScript and nothing
 * at all in WGSL: the pixel budget, the bounce count, the shadow samples and
 * the sample target crossed the boundary, but the sphere-tracing march was a
 * constant baked into each shader — 96 steps in the preview, 160 in the path
 * tracer — so the preview and the trace it converges to marched differently
 * whatever tier was chosen. The budget is a uniform now, and this holds the
 * ladder and its two shaders to one story.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { DEFAULT_DISPLAY, DEFAULT_QUALITY, QUALITY_PRESETS } from "../src/viewer/display";
import { DEFAULT_RENDER_PRESETS, loadRenderPresetState } from "../src/renderPresets";

const VIEWER = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../cadjoint/viewer",
);

describe("the default quality tier", () => {
  it("is ultra", () => {
    expect(DEFAULT_QUALITY).toBe("ultra");
    expect(QUALITY_PRESETS[DEFAULT_QUALITY].label).toBe("Ultra");
  });

  it("is what every shipped render preset asks for", () => {
    for (const preset of DEFAULT_RENDER_PRESETS) {
      expect(preset.quality, preset.id).toBe(DEFAULT_QUALITY);
    }
  });

  it("is what a missing or damaged persisted setting falls back to", () => {
    const empty = { getItem: () => null, setItem: () => {} };
    for (const preset of loadRenderPresetState(empty).presets) {
      expect(preset.quality, preset.id).toBe(DEFAULT_QUALITY);
    }
    const damaged = { getItem: () => "{\"presets\":[{\"id\":\"xray\"}]}", setItem: () => {} };
    for (const preset of loadRenderPresetState(damaged).presets) {
      expect(preset.quality, preset.id).toBe(DEFAULT_QUALITY);
    }
  });

  it("orders the ladder so every knob that costs something rises together", () => {
    const tiers = [QUALITY_PRESETS.draft, QUALITY_PRESETS.high, QUALITY_PRESETS.ultra];
    for (const key of [
      "pixelBudget",
      "bounces",
      "shadowSamples",
      "samples",
    ] as const) {
      expect(tiers[0][key], key).toBeLessThan(tiers[1][key]);
      expect(tiers[1][key], key).toBeLessThan(tiers[2][key]);
    }
  });

  it("holds the march budget flat, because it is not one of them", () => {
    // A tier should only ladder what a tier actually buys. The step budget is
    // a cap, not a cost: measured on end_cap at 319 k / 900 k / 1600 k
    // pixels, 64 steps against 384 is inside the noise at every one. Laddering
    // it bought no time and cost Draft a broken silhouette, so it is now one
    // number for all three and `DisplaySettings.marchSteps` overrides it.
    const tiers = [QUALITY_PRESETS.draft, QUALITY_PRESETS.high, QUALITY_PRESETS.ultra];
    expect(new Set(tiers.map((tier) => tier.marchSteps)).size).toBe(1);
  });
});

describe("the march budget crosses into WGSL", () => {
  const preview = readFileSync(join(VIEWER, "_webgpu.py"), "utf8");
  const tracer = readFileSync(join(VIEWER, "_pathtracer.py"), "utf8");

  it("is read from the uniform rather than baked into either shader", () => {
    for (const [name, source] of [
      ["preview", preview],
      ["path tracer", tracer],
    ] as const) {
      expect(source, name).toContain("u.path_settings.w");
      expect(source, name).toContain("fn trace_steps()");
      // The old constants are gone from the marching loops.
      expect(source, name).not.toMatch(/for \(var step = 0u; step < 96u;/);
      expect(source, name).not.toMatch(/for \(var i = 0; i < 96;/);
    }
  });

  it("leaves a caller that writes nothing on the old picture", () => {
    // widget.py writes zeros into these six scalars; zero has to be inert.
    expect(preview).toContain("const DEFAULT_TRACE_STEPS: i32 = 96;");
    expect(tracer).toContain("const MAX_TRACE_STEPS: u32 = 160u;");
    const widget = readFileSync(join(VIEWER, "widget.py"), "utf8");
    expect(widget).toContain("size: 112,");
  });
});

describe("the distance-field settings", () => {
  it("start on the solid, centred, and at the real surface", () => {
    expect(DEFAULT_DISPLAY.sdfView).toBe("solid");
    expect(DEFAULT_DISPLAY.sdfFraction).toBe(0.5);
    expect(DEFAULT_DISPLAY.isoOffset).toBe(0);
  });

  it("survives a round trip through the preset store", () => {
    let written = "";
    const storage = {
      getItem: () => written,
      setItem: (_key: string, value: string) => {
        written = value;
      },
    };
    const state = loadRenderPresetState(storage);
    state.presets[0].display.sdfView = "gradient";
    state.presets[0].display.isoOffset = 0.125;
    storage.setItem(
      "",
      JSON.stringify({ activeId: "xray", presets: state.presets }),
    );
    const back = loadRenderPresetState(storage);
    expect(back.presets[0].display.sdfView).toBe("gradient");
    expect(back.presets[0].display.isoOffset).toBe(0.125);
  });
});
