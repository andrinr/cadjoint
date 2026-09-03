/** Editable, browser-local render presets for the viewer settings panel. */

import {
  DEFAULT_DISPLAY,
  DEFAULT_QUALITY,
  QUALITY_PRESETS,
  type DisplaySettings,
} from "./viewer/renderer";

export type RenderPresetId = "xray" | "studio" | "wire";

export interface RenderPreset {
  id: RenderPresetId;
  name: string;
  hint: string;
  pathTracing: boolean;
  quality: string;
  display: DisplaySettings;
}

export interface RenderPresetState {
  presets: RenderPreset[];
  activeId: RenderPresetId;
}

interface PresetStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/**
 * Bumped with every change to the shape of a stored preset.
 *
 * v3 adds the SDF-view settings and moves every preset to the Ultra quality
 * tier. v4 adds the three march settings — the step budget, hit refinement
 * and bounds culling; v5 adds the capped section.
 * An older entry validates as incomplete and is discarded rather than
 * migrated: these are three named bundles of defaults, not user documents, and
 * a reader that silently half-applies an old bundle is worse than one that
 * starts from the shipped preset.
 */
export const RENDER_PRESET_STORAGE_KEY = "cadjoint.render-presets.v5";

export const DEFAULT_RENDER_PRESETS: RenderPreset[] = [
  {
    id: "xray",
    name: "X-Ray",
    hint: "Model & inspect",
    pathTracing: false,
    quality: DEFAULT_QUALITY,
    display: { ...DEFAULT_DISPLAY },
  },
  {
    id: "studio",
    name: "Studio",
    hint: "Clean materials",
    pathTracing: true,
    quality: DEFAULT_QUALITY,
    display: {
      ...DEFAULT_DISPLAY,
      projection: "perspective",
      shadows: "soft",
      reflections: true,
      flatShading: false,
      xray: 0,
      showGraticule: false,
      showOverlays: false,
      showSketches: false,
      showConstraints: false,
    },
  },
  {
    id: "wire",
    name: "Wire",
    hint: "Construction only",
    pathTracing: false,
    quality: DEFAULT_QUALITY,
    display: {
      ...DEFAULT_DISPLAY,
      projection: "orthographic",
      shadows: "off",
      reflections: false,
      flatShading: true,
      hideSolid: true,
      xray: 0,
      showSketches: true,
      showConstraints: true,
    },
  },
];

const clonePreset = (preset: RenderPreset): RenderPreset => ({
  ...preset,
  display: { ...preset.display },
});

export function defaultRenderPresets(): RenderPreset[] {
  return DEFAULT_RENDER_PRESETS.map(clonePreset);
}

function isDisplaySettings(value: unknown): value is DisplaySettings {
  if (!value || typeof value !== "object") return false;
  const display = value as Record<string, unknown>;
  return (
    (display.projection === "perspective" ||
      display.projection === "orthographic") &&
    (display.shadows === "off" ||
      display.shadows === "hard" ||
      display.shadows === "soft") &&
    typeof display.reflections === "boolean" &&
    typeof display.flatShading === "boolean" &&
    typeof display.hideSolid === "boolean" &&
    typeof display.xray === "number" &&
    Number.isFinite(display.xray) &&
    typeof display.showGraticule === "boolean" &&
    typeof display.showOverlays === "boolean" &&
    typeof display.showSketches === "boolean" &&
    typeof display.showMeshEdges === "boolean" &&
    typeof display.showMeshWireframe === "boolean" &&
    typeof display.showConstraints === "boolean" &&
    typeof display.showFixedConstraints === "boolean" &&
    typeof display.showDistanceConstraints === "boolean" &&
    typeof display.showConstraintValues === "boolean" &&
    (display.sdfView === "solid" ||
      display.sdfView === "slice" ||
      display.sdfView === "gradient") &&
    (display.sdfAxis === 0 || display.sdfAxis === 1 || display.sdfAxis === 2) &&
    typeof display.sdfFraction === "number" &&
    Number.isFinite(display.sdfFraction) &&
    typeof display.isoOffset === "number" &&
    Number.isFinite(display.isoOffset) &&
    // `null` is the legitimate "follow the quality tier" value, so this is
    // one of the few fields where null has to pass.
    (display.marchSteps === null ||
      (typeof display.marchSteps === "number" &&
        Number.isFinite(display.marchSteps))) &&
    typeof display.refineHit === "boolean" &&
    typeof display.cullBounds === "boolean" &&
    typeof display.section === "boolean"
  );
}

function browserStorage(): PresetStorage | undefined {
  return typeof window === "undefined" ? undefined : window.localStorage;
}

/** Load only complete, valid settings; stale or damaged entries fall back safely. */
export function loadRenderPresetState(
  storage: PresetStorage | undefined = browserStorage(),
): RenderPresetState {
  const fallback = { presets: defaultRenderPresets(), activeId: "xray" as const };
  if (!storage) return fallback;
  try {
    const stored = JSON.parse(
      storage.getItem(RENDER_PRESET_STORAGE_KEY) ?? "null",
    ) as {
      activeId?: unknown;
      presets?: unknown;
    } | null;
    if (!stored || !Array.isArray(stored.presets)) return fallback;
    const storedPresets = stored.presets;

    const presets = DEFAULT_RENDER_PRESETS.map((base) => {
      const candidate = storedPresets.find(
        (item) =>
          item &&
          typeof item === "object" &&
          (item as { id?: unknown }).id === base.id,
      ) as
        | {
            display?: unknown;
            pathTracing?: unknown;
            quality?: unknown;
          }
        | undefined;
      if (
        !candidate ||
        !isDisplaySettings(candidate.display) ||
        typeof candidate.pathTracing !== "boolean" ||
        typeof candidate.quality !== "string" ||
        !QUALITY_PRESETS[candidate.quality]
      ) {
        return clonePreset(base);
      }
      return {
        ...base,
        pathTracing: candidate.pathTracing,
        quality: candidate.quality,
        display: { ...candidate.display },
      };
    });
    const activeId = DEFAULT_RENDER_PRESETS.some(
      (preset) => preset.id === stored.activeId,
    )
      ? (stored.activeId as RenderPresetId)
      : "xray";
    return { presets, activeId };
  } catch {
    return fallback;
  }
}

export function persistRenderPresetState(
  state: RenderPresetState,
  storage: PresetStorage | undefined = browserStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(
      RENDER_PRESET_STORAGE_KEY,
      JSON.stringify({
        activeId: state.activeId,
        presets: state.presets.map(({ id, pathTracing, quality, display }) => ({
          id,
          pathTracing,
          quality,
          display,
        })),
      }),
    );
  } catch {
    // Private browsing or a full storage quota must not break the viewer.
  }
}

export function renderPresetMatches(
  preset: RenderPreset,
  display: DisplaySettings,
  quality: string,
  pathTracing: boolean,
): boolean {
  return (
    preset.pathTracing === pathTracing &&
    preset.quality === quality &&
    Object.keys(DEFAULT_DISPLAY).every(
      (key) =>
        preset.display[key as keyof DisplaySettings] ===
        display[key as keyof DisplaySettings],
    )
  );
}
