/**
 * Contextual sketch history and constraint controls.
 *
 * The source remains authoritative: every button inserts a readable Python
 * operation, then the normal compile cycle rebuilds this panel from metadata.
 */

import {
  For,
  Show,
  createEffect,
  createMemo,
  createSignal,
} from "solid-js";
import {
  busy,
  nodeById,
  selection,
  setSelectionMode,
  setTool,
  solverRuns,
  tool,
} from "../state";
import type { ConstraintSolverMethod } from "../types";

export interface SketchPanelProps {
  onFix: () => void;
  onSolve: (method: ConstraintSolverMethod, iterations: number) => void;
  onExtrude: () => void;
}

export function SketchPanel(props: SketchPanelProps) {
  const [solverOpen, setSolverOpen] = createSignal(false);
  const [solverMethod, setSolverMethod] =
    createSignal<ConstraintSolverMethod>("newton");
  const [solverIterations, setSolverIterations] = createSignal(8);
  const profile = createMemo(() => {
    const active = selection();
    if (!active) return null;
    const node = nodeById(active.nodeId);
    return node?.kind === "profile" ? node : null;
  });
  const selectedVertex = () => selection()?.vertexIndex;
  const hasExtrusion = () => profile()?.operators.some((item) => item.kind === "extrude");
  const lastRun = createMemo(() => {
    const node = profile();
    if (!node) return null;
    return (
      solverRuns()
        .filter((run) => run.node === node.id)
        .at(-1) ?? null
    );
  });
  const lossCurve = createMemo(() => {
    const losses = lastRun()?.losses.filter(Number.isFinite) ?? [];
    if (losses.length === 0) return "";
    const logarithms = losses.map((loss) =>
      Math.log10(Math.max(Math.abs(loss), 1e-12)),
    );
    const low = Math.min(...logarithms);
    const high = Math.max(...logarithms);
    const range = Math.max(high - low, 1e-6);
    return logarithms
      .map((loss, index) => {
        const x = 8 + (index / Math.max(logarithms.length - 1, 1)) * 284;
        const y = 8 + ((high - loss) / range) * 50;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  });
  const formatLoss = (loss: number | undefined) =>
    loss === undefined || !Number.isFinite(loss)
      ? "—"
      : loss.toExponential(2);

  createEffect(() => {
    const run = lastRun();
    if (!run) return;
    setSolverMethod(run.method);
    setSolverIterations(run.iterations);
  });

  return (
    <Show when={profile()}>
      {(node) => (
        <aside class="sketch-panel" data-testid="sketch-panel">
          <header>
            <span>
              <small>Sketch</small>
              {node().name ?? "profile"}
            </span>
            <b>{node().vertices.length} pts</b>
          </header>

          <Show
            when={node().constraints.length > 0 || node().operators.length > 0}
            fallback={<p class="empty-history">No constraints or operators yet</p>}
          >
            <div class="history-chips">
              <For each={node().constraints}>
                {(constraint) => (
                  <span>
                    {constraint.kind === "fixed"
                      ? `fix · P${constraint.vertices[0] + 1}`
                      : `distance · P${constraint.vertices[0] + 1}–P${constraint.vertices[1] + 1}`}
                  </span>
                )}
              </For>
              <For each={node().operators}>
                {(operator) => <span>{operator.kind}</span>}
              </For>
            </div>
          </Show>

          <div class="sketch-actions">
            <button
              type="button"
              disabled={busy() || selectedVertex() == null}
              onClick={props.onFix}
              title="Anchor the selected point"
              data-testid="constraint-fix"
            >
              Fix point
            </button>
            <button
              type="button"
              class={tool() === "distance" ? "active" : ""}
              disabled={busy()}
              onClick={() => {
                setSelectionMode("vertex");
                setTool(tool() === "distance" ? "select" : "distance");
              }}
              title="Choose two points to preserve their current distance"
              data-testid="constraint-distance"
            >
              Distance
            </button>
            <button
              type="button"
              class={solverOpen() ? "active" : ""}
              onClick={() => setSolverOpen(!solverOpen())}
              title="Configure the constraint solver"
              aria-expanded={solverOpen()}
              data-testid="solver-toggle"
            >
              Solver
            </button>
            <button
              type="button"
              disabled={busy() || hasExtrusion()}
              onClick={props.onExtrude}
              title={hasExtrusion() ? "This sketch is already extruded" : "Extrude into the scene"}
              data-testid="sketch-extrude"
            >
              Extrude
            </button>
          </div>

          <Show when={solverOpen()}>
            <section class="solver-panel" data-testid="solver-panel">
              <div class="solver-controls">
                <label>
                  <span>Optimizer</span>
                  <select
                    value={solverMethod()}
                    onChange={(event) =>
                      setSolverMethod(
                        event.currentTarget.value as ConstraintSolverMethod,
                      )
                    }
                    data-testid="solver-method"
                  >
                    <option value="newton">Newton projection</option>
                    <option value="adam">Adam</option>
                    <option value="sgd">SGD</option>
                  </select>
                </label>
                <label>
                  <span>Iterations</span>
                  <input
                    type="number"
                    min="1"
                    max="512"
                    step="1"
                    value={solverIterations()}
                    onInput={(event) => {
                      const value = Number(event.currentTarget.value);
                      if (Number.isInteger(value)) {
                        setSolverIterations(Math.max(1, Math.min(512, value)));
                      }
                    }}
                    data-testid="solver-iterations"
                  />
                </label>
                <button
                  type="button"
                  class="primary"
                  disabled={busy() || node().constraints.length === 0}
                  onClick={() =>
                    props.onSolve(solverMethod(), solverIterations())
                  }
                  data-testid="constraint-solve"
                >
                  Run
                </button>
              </div>

              <Show
                when={lastRun()}
                fallback={
                  <p class="solver-empty">Run the solver to record its residual loss.</p>
                }
              >
                {(run) => (
                  <div class="solver-chart">
                    <div>
                      <span>Last solve</span>
                      <b>
                        {run().method} · {run().iterations} iterations
                      </b>
                    </div>
                    <svg
                      viewBox="0 0 300 66"
                      role="img"
                      aria-label="Constraint residual loss curve"
                      data-testid="solver-loss-chart"
                    >
                      <line x1="8" y1="58" x2="292" y2="58" />
                      <line x1="8" y1="8" x2="8" y2="58" />
                      <polyline points={lossCurve()} />
                    </svg>
                    <footer>
                      <span>{formatLoss(run().losses[0])}</span>
                      <span>
                        {formatLoss(run().losses[run().losses.length - 1])}
                      </span>
                    </footer>
                  </div>
                )}
              </Show>
            </section>
          </Show>
        </aside>
      )}
    </Show>
  );
}
