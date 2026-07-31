import { describe, expect, it } from "vitest";
import {
  DEFAULT_RENDER_PRESETS,
  loadRenderPresetState,
  persistRenderPresetState,
  renderPresetMatches,
} from "../src/renderPresets";

function memoryStorage(initial: string | null = null) {
  let value = initial;
  return {
    getItem: () => value,
    setItem: (_key: string, next: string) => {
      value = next;
    },
    value: () => value,
  };
}

describe("render presets", () => {
  it("ships three distinct presets with X-Ray as the default", () => {
    const state = loadRenderPresetState(undefined);

    expect(state.activeId).toBe("xray");
    expect(state.presets.map((preset) => preset.id)).toEqual([
      "xray",
      "studio",
      "wire",
    ]);
    expect(state.presets[0].display.projection).toBe("orthographic");
    expect(state.presets[0].display.xray).toBe(1);
    expect(state.presets[0].pathTracing).toBe(false);
    expect(state.presets[1].pathTracing).toBe(true);
  });

  it("round-trips edited settings through storage", () => {
    const storage = memoryStorage();
    const presets = DEFAULT_RENDER_PRESETS.map((preset) => ({
      ...preset,
      display: { ...preset.display },
    }));
    presets[1].display.shadows = "off";
    presets[1].quality = "draft";
    presets[1].pathTracing = false;

    persistRenderPresetState(
      { presets, activeId: "studio" },
      storage,
    );
    const loaded = loadRenderPresetState(storage);

    expect(storage.value()).toContain("studio");
    expect(loaded.activeId).toBe("studio");
    expect(loaded.presets[1].display.shadows).toBe("off");
    expect(loaded.presets[1].quality).toBe("draft");
    expect(loaded.presets[1].pathTracing).toBe(false);
  });

  it("detects unsaved changes", () => {
    const preset = DEFAULT_RENDER_PRESETS[0];

    expect(
      renderPresetMatches(
        preset,
        preset.display,
        preset.quality,
        preset.pathTracing,
      ),
    ).toBe(true);
    expect(
      renderPresetMatches(
        preset,
        { ...preset.display, xray: 0 },
        preset.quality,
        preset.pathTracing,
      ),
    ).toBe(false);
    expect(
      renderPresetMatches(
        preset,
        preset.display,
        preset.quality,
        !preset.pathTracing,
      ),
    ).toBe(false);
  });
});
