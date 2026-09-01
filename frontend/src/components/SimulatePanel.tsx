/**
 * FEM simulation panel — a patch layer over meshes and studies in the code.
 *
 * The compile payload lists every SimMesh and ThermalStudy/ElasticStudy in
 * the scene program; this panel renders them and edits them exclusively
 * through /patch source operations (add_mesh, set_mesh_value, add_study_bc,
 * …), the same round-trip the sketch tools use. Solving posts the study's
 * *name*; the server re-derives everything from the declaration and returns
 * the nodal fields, the result summary, and the mesh inspection report.
 *
 * This file is only the frame: the tab strip, the always-visible view
 * controls, and the two failure notices. Each tab is its own module under
 * `simulate/`, and every piece of state and every request they share lives in
 * `simulate/controller.ts` — including the effects that drive the renderer's
 * simulation view and the shared simView/bcProposal signals in state.ts.
 */

import { Show } from "solid-js";
import { OptimizeCards } from "./OptimizeCards";
import { MeshesTab } from "./simulate/MeshesTab";
import { ResultsTab } from "./simulate/ResultsTab";
import { StudiesTab } from "./simulate/StudiesTab";
import { ViewControls } from "./simulate/ViewControls";
import { createSimulateController, type PanelTab } from "./simulate/controller";
import type { Renderer } from "../viewer/renderer";

export interface SimulatePanelProps {
  renderer: Renderer;
  /** Serialized /patch queue owned by the app shell. */
  onPatch: (body: Record<string, unknown>) => Promise<void>;
  /** Adopt server-produced source like a patch response (optimize runs). */
  onAdoptSource: (source: string) => Promise<void>;
  /** Compile-and-render a transient program without committing it. */
  onGhostCompile: (source: string) => Promise<boolean>;
}

const TABS: { value: PanelTab; label: string }[] = [
  { value: "meshes", label: "Meshes" },
  { value: "studies", label: "Studies" },
  { value: "optimize", label: "Optimize" },
  { value: "results", label: "Results" },
];

export function SimulatePanel(props: SimulatePanelProps) {
  const sim = createSimulateController(props);

  return (
    <aside class="sim-panel" data-testid="simulate-panel">
      <header>
        <span>
          <small>FEM</small>
          Simulate
        </span>
      </header>

      {/* The tab strip is hand-written rather than a shared Segmented: the
          Results tab carries a badge when a solve is waiting there. */}
      <div class="sim-tabs" role="tablist" data-testid="sim-tabs">
        {TABS.map((entry) => (
          <button
            type="button"
            role="tab"
            aria-selected={sim.tab() === entry.value}
            classList={{ active: sim.tab() === entry.value }}
            onClick={() => sim.setTab(entry.value)}
            data-testid={`sim-tab-${entry.value}`}
          >
            {entry.label}
            <Show when={entry.value === "results" && sim.result() !== null}>
              <i class="sim-tab-badge" />
            </Show>
          </button>
        ))}
      </div>

      <Show when={sim.tab() === "meshes"}>
        <MeshesTab sim={sim} />
      </Show>

      <Show when={sim.tab() === "studies"}>
        <StudiesTab sim={sim} />
      </Show>

      {/* Optimize tab: the shared optimization cards, next to the simulation
          they drive — a study-backed run lands in Results. */}
      <Show when={sim.tab() === "optimize"}>
        <OptimizeCards
          onPatch={props.onPatch}
          onAdoptSource={props.onAdoptSource}
          onGhostCompile={props.onGhostCompile}
        />
      </Show>

      <Show when={sim.tab() === "results" && !sim.result()}>
        <p class="sim-help" data-testid="simulate-results-empty">
          No results yet — solve a study or run a study-backed optimization.
        </p>
      </Show>
      <Show when={sim.tab() === "results"}>
        <ResultsTab sim={sim} />
      </Show>

      <Show when={sim.result() || sim.inspected()}>
        <ViewControls sim={sim} />
      </Show>

      <Show when={sim.unavailable()}>
        <p class="sim-note" data-testid="simulate-unavailable">
          FEM solves need the optional jax-fem extra on the server. Install it
          with <code> pip install cadjoint[fem]</code> and restart the playground.
          Study declarations still compile and stay editable without it.
        </p>
      </Show>
      <Show when={sim.error() && !sim.unavailable()}>
        <p class="sim-error" data-testid="simulate-error">
          {sim.error()}
        </p>
      </Show>
    </aside>
  );
}
