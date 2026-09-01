/**
 * Everything the top bar's render controls read and write.
 *
 * Display settings, quality, path tracing and the preset library are one
 * concern: the presets are just named bundles of the other three, and every
 * change has to reach the renderer and force a redraw. Keeping them together
 * means the "apply to the renderer, then invalidate" step is written once per
 * setting instead of once per call site, and the browser-local persistence
 * (`renderPresets.ts`) has a single owner.
 *
 * The camera view presets live here too — they are the other half of "what
 * the viewport looks like", and they publish the camera angles the ViewCube
 * reads back.
 */

import { createSignal, type Accessor } from "solid-js";
import {
  DEFAULT_RENDER_PRESETS,
  loadRenderPresetState,
  persistRenderPresetState,
  type RenderPreset,
  type RenderPresetId,
} from "../renderPresets";
import { setCameraAngles } from "../state";
import {
  QUALITY_PRESETS,
  type DisplaySettings,
  type Renderer,
} from "../viewer/renderer";

export interface RenderState {
  display: Accessor<DisplaySettings>;
  presets: Accessor<RenderPreset[]>;
  selectedPreset: Accessor<RenderPresetId>;
  pathTracing: Accessor<boolean>;
  quality: Accessor<string>;
  viewPreset: Accessor<string>;
  /** Merge a display patch, push it to the renderer, redraw. */
  applyDisplay: (patch: Partial<DisplaySettings>) => void;
  applyQuality: (key: string) => void;
  applyPathTracing: (enabled: boolean) => void;
  activateRenderPreset: (id: RenderPresetId) => void;
  saveRenderPreset: (id: RenderPresetId) => void;
  resetRenderPreset: (id: RenderPresetId) => void;
  /** Move the camera to a named view ("iso", "front", …). */
  applyPreset: (key: string) => void;
}

/**
 * Seed the signals from browser-local storage and hand the renderer its
 * starting settings, so the first frame already matches the active preset.
 */
export function createRenderState(renderer: Renderer): RenderState {
  const initialPresetState = loadRenderPresetState();
  const initialPreset =
    initialPresetState.presets.find(
      (preset) => preset.id === initialPresetState.activeId,
    ) ?? initialPresetState.presets[0];

  const [pathTracing, setPathTracing] = createSignal(initialPreset.pathTracing);
  const [quality, setQuality] = createSignal(initialPreset.quality);
  const [display, setDisplay] = createSignal<DisplaySettings>({
    ...initialPreset.display,
  });
  const [presets, setPresets] = createSignal(initialPresetState.presets);
  const [selectedPreset, setSelectedPreset] = createSignal<RenderPresetId>(
    initialPresetState.activeId,
  );
  const [viewPreset, setViewPreset] = createSignal("iso");

  renderer.display = { ...initialPreset.display };
  renderer.quality = QUALITY_PRESETS[initialPreset.quality];
  renderer.pathTracing = initialPreset.pathTracing;

  /** Push display settings to the renderer and redraw. */
  const applyDisplay = (patch: Partial<DisplaySettings>) => {
    const next = { ...display(), ...patch };
    setDisplay(next);
    renderer.display = next;
    renderer.invalidate();
  };

  const applyQuality = (key: string) => {
    const preset = QUALITY_PRESETS[key];
    if (!preset) return;
    setQuality(key);
    renderer.quality = preset;
    renderer.invalidate();
  };

  const applyPathTracing = (enabled: boolean) => {
    setPathTracing(enabled);
    renderer.pathTracing = enabled;
    renderer.invalidate();
  };

  const activateRenderPreset = (id: RenderPresetId) => {
    const preset = presets().find((item) => item.id === id);
    if (!preset) return;
    setSelectedPreset(id);
    applyDisplay(preset.display);
    applyQuality(preset.quality);
    applyPathTracing(preset.pathTracing);
    persistRenderPresetState({ presets: presets(), activeId: id });
  };

  const saveRenderPreset = (id: RenderPresetId) => {
    const next = presets().map((preset) =>
      preset.id === id
        ? {
            ...preset,
            pathTracing: pathTracing(),
            quality: quality(),
            display: { ...display() },
          }
        : preset,
    );
    setPresets(next);
    setSelectedPreset(id);
    persistRenderPresetState({ presets: next, activeId: id });
  };

  const resetRenderPreset = (id: RenderPresetId) => {
    const original = DEFAULT_RENDER_PRESETS.find((preset) => preset.id === id);
    if (!original) return;
    const next = presets().map((preset) =>
      preset.id === id
        ? { ...original, display: { ...original.display } }
        : preset,
    );
    setPresets(next);
    setSelectedPreset(id);
    applyDisplay(original.display);
    applyQuality(original.quality);
    applyPathTracing(original.pathTracing);
    persistRenderPresetState({ presets: next, activeId: id });
  };

  const applyPreset = (key: string) => {
    setViewPreset(key);
    renderer.applyViewPreset(key);
    setDisplay({ ...renderer.display });
    setCameraAngles({ yaw: renderer.camera.yaw, pitch: renderer.camera.pitch });
  };

  return {
    display,
    presets,
    selectedPreset,
    pathTracing,
    quality,
    viewPreset,
    applyDisplay,
    applyQuality,
    applyPathTracing,
    activateRenderPreset,
    saveRenderPreset,
    resetRenderPreset,
    applyPreset,
  };
}
