/**
 * The scene browser: what is saved beside this document, and what is in it.
 *
 * `File → Open…` is a list of file names, which is the right control when you
 * already know which file you want and the wrong one every other time. This
 * window answers the other question — *which* of these is the bracket with
 * the thermal study, which one has free parameters an optimization can move,
 * which one did I touch last week — and it answers it without running a line
 * of any of them. The counts, the summary and the material names all come
 * from the server's `ast` pass over the file (`_scenes.py`); the only thing
 * that costs a compile is the picture, and that is cached by source hash and
 * drawn one at a time (`sceneThumbnailer.ts`).
 *
 * Opening a scene goes through exactly the path the menu uses, guard
 * included: an unsaved buffer asks before it is replaced.
 */

import { For, Show, createSignal, onCleanup, onMount } from "solid-js";
import * as api from "../api";
import { createThumbnailer, type ThumbnailOutcome } from "../sceneThumbnailer";
import { createQueue, firstLine, formatFileSize, formatModified } from "../thumbnails";
import { confirmDiscardChanges } from "../scenes";
import { sceneName, setStatus } from "../state";
import { Section, Stat, StatRow } from "./ui";
import type { SceneEntry } from "../types";
import "./scenes.css";

export interface ScenesPanelProps {
  /** Adopt a loaded scene as the current program — the menu's own path. */
  onOpen: (name: string, source: string) => void;
}

/** What the browser knows about one card's picture right now. */
type Picture =
  | { state: "queued" }
  | { state: "drawing" }
  | ThumbnailOutcome;

export function ScenesPanel(props: ScenesPanelProps) {
  const [entries, setEntries] = createSignal<SceneEntry[] | null>(null);
  const [listError, setListError] = createSignal("");
  const [pictures, setPictures] = createSignal<Record<string, Picture>>({});

  const thumbnailer = createThumbnailer();
  const queue = createQueue();
  /** Set on teardown so a queued thumbnail stops rather than drawing on. */
  let gone = false;
  onCleanup(() => {
    gone = true;
    thumbnailer.dispose();
  });

  const setPicture = (name: string, picture: Picture) =>
    setPictures((current) => ({ ...current, [name]: picture }));

  /**
   * Draw the pictures for a listing, one at a time.
   *
   * Strictly serial, and deliberately so: each thumbnail holds a compile
   * worker on the server and a GPU frame in the browser, and the gearbox
   * end-cap alone is several seconds of that. In a queue the first card fills
   * in while the panel is still being read; in parallel they would all arrive
   * late and the panel would be unusable until they did.
   */
  const drawAll = (listing: readonly SceneEntry[]) => {
    for (const entry of listing) {
      if (pictures()[entry.name]?.state === "ok") continue;
      setPicture(entry.name, { state: "queued" });
      void queue.push(async () => {
        if (gone) return;
        setPicture(entry.name, { state: "drawing" });
        const outcome = await thumbnailer.render(entry.name, entry.source_hash);
        if (!gone) setPicture(entry.name, outcome);
      });
    }
  };

  const refresh = async () => {
    setListError("");
    try {
      const listing = await api.listScenes();
      if (gone) return;
      const described = listing.scenes ?? [];
      setEntries(described);
      drawAll(described);
    } catch (error) {
      setEntries([]);
      setListError(error instanceof Error ? error.message : String(error));
    }
  };

  onMount(() => void refresh());

  const open = async (entry: SceneEntry) => {
    if (!confirmDiscardChanges()) return;
    try {
      const loaded = await api.loadScene(entry.name);
      if (!loaded.ok || typeof loaded.source !== "string") {
        setStatus({ kind: "error", text: loaded.error ?? `Could not open ${entry.name}.` });
        return;
      }
      props.onOpen(entry.name, loaded.source);
    } catch (error) {
      setStatus({
        kind: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    }
  };

  /**
   * The picture, or the honest placeholder for why there is none.
   *
   * The placeholder is hatched rather than blank: an empty rectangle reads as
   * "still loading" forever, and a scene that will never draw should not look
   * like one that is about to.
   */
  const frame = (entry: SceneEntry) => {
    const picture = (): Picture => pictures()[entry.name] ?? { state: "queued" };
    return (
      <div
        class="scene-frame"
        data-testid={`scene-thumb-${entry.name}`}
        data-state={picture().state}
      >
        <Show
          when={picture().state === "ok" ? (picture() as { dataUrl: string }) : null}
          fallback={
            <div class="scene-frame-empty">
              <span>
                {picture().state === "drawing"
                  ? "drawing…"
                  : picture().state === "queued"
                    ? "queued"
                    : firstLine(
                        (picture() as { message?: string }).message ?? entry.error,
                      )}
              </span>
            </div>
          }
        >
          {(drawn) => <img src={drawn().dataUrl} alt={`${entry.name} rendered`} />}
        </Show>
      </div>
    );
  };

  return (
    <aside class="sim-panel scenes-panel" data-testid="scenes-panel">
      <header>
        <span>
          <small>FILES</small>
          Scenes
        </span>
      </header>

      <Section
        title="Saved"
        count={entries()?.length ?? ""}
        testId="scenes-list"
        actions={
          <button
            type="button"
            class="sim-add-inline"
            onClick={() => void refresh()}
            title="Re-read the scenes directory"
            data-testid="scenes-refresh"
          >
            Refresh
          </button>
        }
      >
        <Show when={listError()}>
          <p class="sim-error" data-testid="scenes-error">
            {listError()}
          </p>
        </Show>

        <Show
          when={(entries()?.length ?? 0) > 0}
          fallback={
            <Show when={entries() !== null}>
              <p class="sim-help" data-testid="scenes-empty">
                No saved scenes yet. File → Save As writes one into the
                server's <code>scenes</code> directory.
              </p>
            </Show>
          }
        >
          <ul class="scene-cards">
            <For each={entries()!}>
              {(entry) => (
                <li
                  class="scene-card"
                  classList={{ current: sceneName() === entry.name }}
                  data-testid={`scene-${entry.name}`}
                >
                  <button
                    type="button"
                    class="scene-open"
                    onClick={() => void open(entry)}
                    title={`Open ${entry.name}`}
                    data-testid={`scene-open-${entry.name}`}
                  >
                    {frame(entry)}
                    <strong>{entry.name}</strong>
                  </button>
                  <p class="scene-summary" data-testid={`scene-summary-${entry.name}`}>
                    {entry.summary || "No module docstring."}
                  </p>
                  <StatRow testId={`scene-counts-${entry.name}`}>
                    <Stat label="params" value={entry.counts.parameters} />
                    <Stat label="free" value={entry.counts.free} />
                    <Stat label="studies" value={entry.counts.studies} />
                    <Stat label="meshes" value={entry.counts.meshes} />
                    <Stat label="optims" value={entry.counts.optimizations} />
                  </StatRow>
                  <Show when={entry.materials.length > 0}>
                    <p class="scene-materials" data-testid={`scene-materials-${entry.name}`}>
                      <For each={entry.materials}>
                        {(material) => <span class="sim-kind">{material}</span>}
                      </For>
                    </p>
                  </Show>
                  <p class="scene-meta" data-testid={`scene-meta-${entry.name}`}>
                    {formatModified(entry.modified)} · {formatFileSize(entry.bytes)}
                  </p>
                  <Show when={entry.error}>
                    <p class="sim-error">{firstLine(entry.error)}</p>
                  </Show>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </Section>
    </aside>
  );
}
