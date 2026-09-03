/**
 * The four simulation windows, and what they each put in the dock.
 *
 * Simulation used to be one panel with a tab strip — Meshes, Studies,
 * Optimize, Results — inside a mode that had its own second-level navigation.
 * That was one arrangement too many: the dock already tabs windows, and a
 * strip inside a tab is a strip the user cannot move, split, float or park.
 * So the tabs became windows. "Simulate" is now purely a *desk*: the mode's
 * default layout puts Studies over Results beside the viewport, with Meshes
 * tabbed behind the first and Optimize behind the second — setup above,
 * outcomes below — and every one of them can be dragged anywhere from there.
 *
 * The views themselves are unchanged: each window hosts the same
 * `MeshesTab` / `StudiesTab` / `ResultsTab` module it hosted as a tab, over
 * the same shared controller. What each window adds is only its frame.
 *
 * Two placements are worth stating, because they are what keeps a shared
 * controller from showing the same control twice:
 *
 *   - **The view controls** (viewport switch, element edges, slice) belong to
 *     whatever is displayed, and exactly one thing ever is: a solve clears
 *     the inspection and an inspection clears the solve. So they ride with
 *     the inspected mesh in Meshes and with the solved field in Results,
 *     which is also where the user just was when either arrived.
 *   - **The notices** (a failed request, a missing FEM extra, an evicted
 *     result) are shared state, so each window shows them under its own id
 *     rather than three windows claiming the same one.
 *
 * The Optimize window is `components/OptimizePanel.tsx`: it was already a
 * window in the Model desk, and folding the Simulate tab into it is what
 * makes the two homes literally the same window rather than two copies of
 * one card list.
 */

import { Show, onCleanup } from "solid-js";
import { setFocusedJob } from "../../jobs";
import { windowManager } from "../../windows/manager";
import { MeshesTab } from "./MeshesTab";
import { ResultsTab } from "./ResultsTab";
import { StudiesTab } from "./StudiesTab";
import { ViewControls } from "./ViewControls";
import type { SimulateController } from "./controller";
import "./simWindows.css";

export interface SimWindowProps {
  sim: SimulateController;
}

interface NoticesProps extends SimWindowProps {
  /** Which window is speaking, so three copies do not share one test id. */
  prefix: string;
}

/**
 * The three things that can go wrong, in the window they went wrong in.
 *
 * The registry keeps the last 50 jobs and 64 MB of payloads; past that a
 * stored result is gone, and the honest answer is to say so and let the user
 * run it again rather than to show an empty window.
 */
function SimNotices(props: NoticesProps) {
  return (
    <>
      <Show when={props.sim.expired()}>
        <p class="sim-note" data-testid={`${props.prefix}-expired`}>
          {props.sim.expired()}
        </p>
      </Show>
      <Show when={props.sim.unavailable()}>
        <p class="sim-note" data-testid={`${props.prefix}-unavailable`}>
          FEM solves need the optional jax-fem extra on the server. Install it
          with <code> pip install cadjoint[fem]</code> and restart the playground.
          Study declarations still compile and stay editable without it.
        </p>
      </Show>
      {/* A failed solve is not a status line.
          TetGen refusing a self-intersecting surface is three sentences the
          user needs in order to know what to change, and before this they
          went to the server log while the panel showed the previous result
          with a "stale" chip — which reads as "still thinking", not as "that
          run failed". So the whole message is shown, in the window that
          asked for the work, in front of whatever is still on screen; the
          old result stays visible *behind* it rather than being cleared,
          because it is the thing being compared against. */}
      <Show when={props.sim.error() && !props.sim.unavailable()}>
        <div class="sim-failure" data-testid={`${props.prefix}-failure`}>
          <p class="sim-error" data-testid={`${props.prefix}-error`}>
            {props.sim.error()}
          </p>
          <Show when={props.sim.errorJob()}>
            {(job) => (
              <p class="sim-failure-job">
                <span>job {job()}</span>
                <button
                  type="button"
                  class="sim-add-inline"
                  onClick={() => {
                    setFocusedJob(job());
                    windowManager()?.open("processes");
                  }}
                  title="Open this job's row in the process monitor"
                  data-testid={`${props.prefix}-error-job`}
                >
                  Show in Processes
                </button>
              </p>
            )}
          </Show>
        </div>
      </Show>
    </>
  );
}

/** Declare, generate and judge discretizations. */
export function MeshesWindow(props: SimWindowProps) {
  onCleanup(props.sim.attach("meshes"));
  return (
    <aside class="sim-panel" data-testid="meshes-panel">
      <header>
        <span>
          <small>FEM</small>
          Meshes
        </span>
      </header>
      <MeshesTab sim={props.sim} />
      <Show when={props.sim.inspected()}>
        <ViewControls sim={props.sim} />
      </Show>
      <SimNotices sim={props.sim} prefix="meshes" />
    </aside>
  );
}

/**
 * The study declarations, their boundary conditions, and Solve.
 *
 * This is the window the Simulate desk opens on, so it keeps the two ids the
 * suite uses to ask "am I in a simulation desk at all": `mode-simulate` on
 * the slot and `simulate-panel` on the sheet.
 */
export function StudiesWindow(props: SimWindowProps) {
  onCleanup(props.sim.attach("studies"));
  return (
    <div class="mode-simulate-slot" data-testid="mode-simulate">
      <aside class="sim-panel" data-testid="simulate-panel">
        <header>
          <span>
            <small>FEM</small>
            Studies
          </span>
        </header>
        <StudiesTab sim={props.sim} />
        <SimNotices sim={props.sim} prefix="simulate" />
      </aside>
    </div>
  );
}

/** What a solve left behind, and the lenses over it. */
export function ResultsWindow(props: SimWindowProps) {
  onCleanup(props.sim.attach("results"));
  return (
    <aside class="sim-panel" data-testid="results-panel">
      <header>
        <span>
          <small>FEM</small>
          Results
        </span>
      </header>
      {/* In front of the result, not after it: a solve that failed while an
          older one is on screen has to say so above the field it is not. */}
      <SimNotices sim={props.sim} prefix="results" />
      <Show when={!props.sim.result()}>
        <p class="sim-help" data-testid="simulate-results-empty">
          No results yet — solve a study or run a study-backed optimization.
        </p>
      </Show>
      <ResultsTab sim={props.sim} />
      <Show when={props.sim.result()}>
        <ViewControls sim={props.sim} />
      </Show>
    </aside>
  );
}
