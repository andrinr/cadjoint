/**
 * The three march settings: the step budget, hit refinement, bounds culling.
 *
 * All three are *rendering* choices, not scene edits — they never touch the
 * source and never cause a recompile. Two of them ride in the viewer's own
 * uniform block (the budget in `path_settings.w`, refinement as a bit in
 * `display.z`); the third rides in the scene shader's parameter buffer,
 * because the skip tests it gates are inside the generated module.
 *
 * What is pinned here is the packing and the defaults. A wrong flag bit or a
 * margin written to the wrong slot does not fail loudly — it silently draws
 * a different part, or quietly costs 2.4x the frame.
 */

import { describe, expect, it } from "vitest";
import {
  DEFAULT_DISPLAY,
  DISPLAY,
  MARCH_STEPS_MAX,
  MARCH_STEPS_MIN,
  MARCH_STEPS_TIER,
  QUALITY_PRESETS,
  displayFlags,
  effectiveMarchSteps,
} from "../src/viewer/display";
import {
  CULL_MARGIN_OFF,
  CULL_MARGIN_ON,
  packParameters,
  type ShaderProgramPayload,
} from "../src/viewer/shaderProgram";
import {
  DEFAULT_RENDER_PRESETS,
  RENDER_PRESET_STORAGE_KEY,
  loadRenderPresetState,
} from "../src/renderPresets";

describe("the defaults reproduce the image the viewer has always drawn", () => {
  it("follows the quality tier's step budget", () => {
    expect(DEFAULT_DISPLAY.marchSteps).toBeNull();
    expect(effectiveMarchSteps(DEFAULT_DISPLAY, QUALITY_PRESETS.ultra)).toBe(192);
  });

  it("gives every tier the same budget, because it costs the same on each", () => {
    // Measured on end_cap at three resolutions: 64 against 384 is inside the
    // noise at all of them, so a ladder here bought nothing and cost Draft a
    // broken silhouette. What separates the tiers is the pixel budget.
    for (const tier of Object.values(QUALITY_PRESETS)) {
      expect(tier.marchSteps).toBe(MARCH_STEPS_TIER);
    }
    const budgets = Object.values(QUALITY_PRESETS).map((t) => t.pixelBudget);
    expect(new Set(budgets).size).toBe(budgets.length);
  });

  it("leaves the section off", () => {
    // A capped section changes the geometry that is drawn, so it cannot be
    // the state the viewer opens in.
    expect(DEFAULT_DISPLAY.section).toBe(false);
    expect(displayFlags(DEFAULT_DISPLAY, false) & DISPLAY.section).toBe(0);
  });

  it("leaves refinement off and culling on", () => {
    // Refinement changes pixels, so it cannot default on; culling changes
    // none, so it cannot default off.
    expect(DEFAULT_DISPLAY.refineHit).toBe(false);
    expect(DEFAULT_DISPLAY.cullBounds).toBe(true);
  });

  it("sets no new flag bit", () => {
    expect(displayFlags(DEFAULT_DISPLAY, false) & DISPLAY.refineHit).toBe(0);
  });

  it("ships every preset on the defaults", () => {
    for (const preset of DEFAULT_RENDER_PRESETS) {
      expect(preset.display.marchSteps).toBeNull();
      expect(preset.display.refineHit).toBe(false);
      expect(preset.display.cullBounds).toBe(true);
      expect(preset.display.section).toBe(false);
    }
  });
});

describe("effectiveMarchSteps", () => {
  it("prefers an override to the tier", () => {
    expect(effectiveMarchSteps({ marchSteps: 128 }, QUALITY_PRESETS.ultra)).toBe(128);
  });

  it("clamps an override to the supported range", () => {
    expect(effectiveMarchSteps({ marchSteps: 1 }, QUALITY_PRESETS.ultra)).toBe(
      MARCH_STEPS_MIN,
    );
    expect(effectiveMarchSteps({ marchSteps: 99_999 }, QUALITY_PRESETS.ultra)).toBe(
      MARCH_STEPS_MAX,
    );
  });

  it("rounds a fractional override rather than sending the shader a fraction", () => {
    expect(effectiveMarchSteps({ marchSteps: 100.6 }, QUALITY_PRESETS.ultra)).toBe(101);
  });

  it("falls back to the tier for a value that is not a number", () => {
    // A damaged persisted setting must not blank the viewport.
    expect(
      effectiveMarchSteps({ marchSteps: Number.NaN }, QUALITY_PRESETS.high),
    ).toBe(MARCH_STEPS_TIER);
  });
});

describe("the refinement flag", () => {
  it("is bit 32, matching DISPLAY_REFINE_HIT in _webgpu.py", () => {
    expect(DISPLAY.refineHit).toBe(32);
  });

  it("is set only when asked for, and disturbs no other bit", () => {
    const off = displayFlags(DEFAULT_DISPLAY, false);
    const on = displayFlags({ ...DEFAULT_DISPLAY, refineHit: true }, false);
    expect(on & DISPLAY.refineHit).toBe(DISPLAY.refineHit);
    expect(on & ~DISPLAY.refineHit).toBe(off);
  });
});

describe("the section flag", () => {
  it("is bit 64, matching DISPLAY_SECTION in _webgpu.py", () => {
    expect(DISPLAY.section).toBe(64);
  });

  it("is set only when asked for, and disturbs no other bit", () => {
    const off = displayFlags(DEFAULT_DISPLAY, false);
    const on = displayFlags({ ...DEFAULT_DISPLAY, section: true }, false);
    expect(on & DISPLAY.section).toBe(DISPLAY.section);
    expect(on & ~DISPLAY.section).toBe(off);
  });

  it("composes with a field view rather than excluding it", () => {
    // Both use the one plane. The section cuts the solid; the card draws the
    // field on the same plane and covers the cut. Nothing forbids the pair.
    const both = displayFlags(
      { ...DEFAULT_DISPLAY, section: true, sdfView: "slice" },
      false,
    );
    expect(both & DISPLAY.section).toBe(DISPLAY.section);
  });
});

describe("the cull margin in the parameter buffer", () => {
  const program: ShaderProgramPayload = {
    group: 3,
    binding: 0,
    buffer_bytes: 48,
    nan_offset: 16,
    cull_margin_offset: 32,
    parameters: [
      { name: "radius", offset: 0, components: 1, value: [0.6], free: true },
    ],
  };

  it("writes the on-margin by default", () => {
    // The buffer is float32, so the margin arrives rounded; what matters is
    // that it is the on-value and not the off-value.
    expect(packParameters(program)[8]).toBeCloseTo(CULL_MARGIN_ON, 10);
  });

  it("writes infinity to switch culling off", () => {
    const packed = packParameters(program, undefined, CULL_MARGIN_OFF);
    expect(packed[8]).toBe(Number.POSITIVE_INFINITY);
    // The parameters and the NaN slot are untouched by the toggle.
    expect(packed[0]).toBeCloseTo(0.6, 6);
    expect(Number.isNaN(packed[4])).toBe(true);
  });

  it("leaves a program without the slot alone", () => {
    // Absent and explicitly null both mean "no margin slot": a literal-form
    // shader, or one built before the toggle existed. Writing on that basis
    // would land on a real parameter.
    const { cull_margin_offset: _absent, ...older } = program;
    expect(packParameters(older, undefined, CULL_MARGIN_OFF)[0]).toBeCloseTo(0.6, 6);
    const nulled = { ...program, cull_margin_offset: null };
    expect(packParameters(nulled, undefined, CULL_MARGIN_OFF)[0]).toBeCloseTo(0.6, 6);
  });

  it("is off only at infinity — a large finite margin would still cull", () => {
    // The guarantee is that *no* box test can pass. Only infinity gives it
    // for a scene of unknown extent.
    expect(CULL_MARGIN_OFF).toBe(Number.POSITIVE_INFINITY);
    expect(CULL_MARGIN_ON).toBeGreaterThan(0);
  });
});

describe("persisted presets", () => {
  const storage = (value: string | null) => {
    let held = value;
    return {
      getItem: () => held,
      setItem: (_key: string, next: string) => {
        held = next;
      },
    };
  };

  it("discards an entry from before the march settings existed", () => {
    // v3 presets have no marchSteps/refineHit/cullBounds, so they validate as
    // incomplete and the shipped preset is used instead — a half-applied
    // bundle would leave the viewer marching on a setting nobody chose.
    const stale = JSON.stringify({
      activeId: "studio",
      presets: [{ id: "xray", pathTracing: false, quality: "ultra", display: {} }],
    });
    const state = loadRenderPresetState(storage(stale));
    expect(state.presets[0].display.marchSteps).toBeNull();
    expect(state.presets[0].display.cullBounds).toBe(true);
    expect(state.presets[0].display.section).toBe(false);
  });

  it("round-trips a customised budget", () => {
    const store = storage(null);
    const saved = JSON.stringify({
      activeId: "xray",
      presets: DEFAULT_RENDER_PRESETS.map((preset) => ({
        id: preset.id,
        pathTracing: preset.pathTracing,
        quality: preset.quality,
        display: { ...preset.display, marchSteps: 256, refineHit: true },
      })),
    });
    store.setItem(RENDER_PRESET_STORAGE_KEY, saved);
    const state = loadRenderPresetState(store);
    expect(state.presets[0].display.marchSteps).toBe(256);
    expect(state.presets[0].display.refineHit).toBe(true);
  });
});
