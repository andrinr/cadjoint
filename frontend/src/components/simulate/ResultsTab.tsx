/**
 * Results tab — browsing what a solve left behind.
 *
 * A solve returns more than one number per node, so this tab is a set of
 * lenses over the same payload: which field the surface is coloured by, the
 * mesh-quality heatmap in place of a field, and the warped shape for elastic
 * results. Switching between them never re-solves — the fields all ride on
 * the response — which is what makes the picker worth having.
 *
 * A study-backed optimization lands here too, with its trajectory player, so
 * an optimization run ends showing the optimized design's solved field.
 */

import { For, Show } from "solid-js";
import { formatScalar, rampCss } from "../../simulation";
import { optimizeRun, simProbe } from "../../state";
import { Segmented, Stat, StatRow, ToggleSwitch } from "../ui";
import { TrajectoryPlayer } from "../TrajectoryPlayer";
import type { SimulateController, ViewportMode } from "./controller";

export interface ResultsTabProps {
  sim: SimulateController;
}

export function ResultsTab(props: ResultsTabProps) {
  const sim = () => props.sim;

  return (
    <Show when={sim().result()}>
      {(current) => (
        <>
          <Show when={Object.keys(current().fields).length > 1}>
            <Segmented
              class="sim-field-picker"
              testId="simulate-fields"
              options={Object.keys(current().fields).map((field) => ({
                value: field,
                label: field.replaceAll("_", " "),
                testId: `simulate-field-${field}`,
              }))}
              value={
                sim().qualityView()
                  ? null
                  : (sim().activeField() ?? current().defaultField)
              }
              onSelect={(field) => {
                sim().setQualityView(false);
                sim().setActiveField(field);
                sim().applyView();
              }}
            />
          </Show>

          <div class="sim-legend" data-testid="simulate-legend">
            <small>
              {current().name} · {sim().activeScalars()?.label ?? ""}
              {/* A result that no longer describes the program says so and
                  stays: throwing it away would be worse than showing it
                  with a date on it. Quiet, and not a colour — a warning
                  hue here would fight the field ramp underneath it. */}
              <Show when={sim().stale()}>
                <span class="sim-kind" data-testid="simulate-stale">
                  stale · source changed
                </span>
              </Show>
            </small>
            <div class="sim-ramp" style={{ background: rampCss() }} />
            <div class="sim-legend-values">
              <span>{formatScalar(sim().activeScalars()?.range[0] ?? NaN)}</span>
              <span>{formatScalar(sim().activeScalars()?.range[1] ?? NaN)}</span>
            </div>
          </div>

          <Show when={current().result}>
            {(summary) => (
              <div class="sim-result-summary" data-testid="simulate-result-summary">
                <StatRow>
                  <Stat label="nodes" value={summary().nodes} />
                  <Stat label="elements" value={summary().elements} />
                  <Show when={summary().mesh}>
                    <Stat label="mesh" value={summary().mesh} />
                  </Show>
                </StatRow>
                <table class="sim-field-table">
                  <thead>
                    <tr>
                      <th>field</th>
                      <th>min</th>
                      <th>mean</th>
                      <th>max</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={Object.entries(summary().fields)}>
                      {([field, values]) => (
                        <tr>
                          <td>{field.replaceAll("_", " ")}</td>
                          <td>{formatScalar(values.min)}</td>
                          <td>{formatScalar(values.mean)}</td>
                          <td>{formatScalar(values.max)}</td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            )}
          </Show>

          {/* The step-through control lands with the user: when this result
              came from an optimization run, its trajectory player mounts
              here too (same shared state as the Optimize card). Replay hands
              the viewport to the raymarched scene so the geometry visibly
              morphs, then puts back whichever view the user was in when it
              started. */}
          <Show when={optimizeRun()?.name === current().name ? optimizeRun() : null}>
            {(run) => {
              let viewBeforeReplay: ViewportMode = sim().viewportMode();
              return (
                <TrajectoryPlayer
                  onGhostCompile={sim().props.onGhostCompile}
                  sparkTestId="results-optimize-history"
                  fieldNote={run().study !== null}
                  onReplayStart={() => {
                    viewBeforeReplay = sim().viewportMode();
                    sim().setViewport("scene");
                  }}
                  onReplayEnd={() => sim().setViewport(viewBeforeReplay)}
                />
              );
            }}
          </Show>

          <ToggleSwitch
            compact
            checked={sim().qualityView()}
            disabled={sim().inspecting() !== null}
            onChange={() => void sim().toggleQualityView()}
            testId="simulate-quality-toggle"
          >
            View mesh quality
          </ToggleSwitch>

          <Show when={current().displacements}>
            <div class="sim-row sim-deform">
              <ToggleSwitch
                compact
                checked={sim().deformed()}
                onChange={(checked) => {
                  sim().setDeformed(checked);
                  sim().applyView();
                }}
                testId="simulate-deformed"
              >
                Deformed
              </ToggleSwitch>
              <input
                type="range"
                min="0"
                max="3"
                step="0.05"
                value={sim().deformFactor()}
                disabled={!sim().deformed()}
                onInput={(event) => {
                  sim().setDeformFactor(Number(event.currentTarget.value));
                  sim().applyView();
                }}
                title="Warp scale, relative to the automatic 10%-of-diagonal"
                data-testid="simulate-deformed-scale"
              />
            </div>
          </Show>

          {/* The last probed point, mirrored from the viewport chip. */}
          <Show when={simProbe()}>
            {(probe) => (
              <StatRow class="sim-probe-row" testId="simulate-probe-row">
                <Stat label="probe" value={formatScalar(probe().value)} />
                <Stat
                  label="at"
                  value={`[${probe()
                    .world.map((component) => component.toFixed(3))
                    .join(", ")}]`}
                />
              </StatRow>
            )}
          </Show>
        </>
      )}
    </Show>
  );
}
