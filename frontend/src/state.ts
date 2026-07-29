/**
 * Application state.
 *
 * The Python source is the single source of truth: viewer edits go through
 * `/patch`, the patched text replaces the editor contents, and a recompile
 * regenerates the construction tree. Nothing about a sketch is stored outside
 * the program text except the transient drag override below.
 */

import { createSignal } from "solid-js";
import type { ConstructionProfile, Selection, ToolMode } from "./types";

export const [source, setSource] = createSignal("");
export const [profiles, setProfiles] = createSignal<ConstructionProfile[]>([]);
export const [selection, setSelection] = createSignal<Selection | null>(null);
export const [hover, setHover] = createSignal<Selection | null>(null);
export const [tool, setTool] = createSignal<ToolMode>("select");
export const [status, setStatus] = createSignal({ kind: "", text: "Starting…" });
export const [viewerError, setViewerError] = createSignal("");
export const [consoleText, setConsoleText] = createSignal("");
export const [busy, setBusy] = createSignal(false);
export const [dirty, setDirty] = createSignal(false);

/** Vertex being dragged, with its live sketch-plane position. */
export interface DragState {
  profileId: string;
  vertexIndex: number;
  xy: [number, number];
}

export const [drag, setDrag] = createSignal<DragState | null>(null);

/** Find a profile by id. */
export function profileById(id: string): ConstructionProfile | undefined {
  return profiles().find((profile) => profile.id === id);
}

/**
 * Profiles as they should be drawn right now.
 *
 * While a drag is in flight the moved vertex is patched in locally so the
 * sketch follows the pointer; the solid only rebuilds once the drag ends and
 * the source is recompiled.
 */
export function displayProfiles(): ConstructionProfile[] {
  const active = drag();
  const all = profiles();
  if (!active) return all;
  return all.map((profile) => {
    if (profile.id !== active.profileId) return profile;
    const { origin, u, v } = profile.plane;
    const [x, y] = active.xy;
    const world: [number, number, number] = [
      origin[0] + u[0] * x + v[0] * y,
      origin[1] + u[1] * x + v[1] * y,
      origin[2] + u[2] * x + v[2] * y,
    ];
    const vertices = profile.vertices.map((vertex, index) =>
      index === active.vertexIndex ? { ...vertex, uv: active.xy, world } : vertex,
    );
    return { ...profile, vertices };
  });
}
