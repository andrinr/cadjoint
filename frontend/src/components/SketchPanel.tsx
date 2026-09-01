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
  pendingLoft,
  selection,
  setPendingLoft,
  setSelectionMode,
  setTool,
  solverRuns,
  tool,
} from "../state";
import type { ConstraintSolverMethod, ToolMode } from "../types";
import { constraintLabel, isEdgeConstraintTool } from "../constraints";

export interface SketchPanelProps {
  onFix: () => void;
  onSolve: (method: ConstraintSolverMethod, iterations: number) => void;
  onExtrude: () => void;
  onRevolve: () => void;
  /** Remove the profile's index-th serialized constraint. */
  onDeleteConstraint: (line: number, index: number) => void;
  /** Rewrite a distance constraint's numeric target. */
  onSetConstraintValue: (line: number, index: number, value: number) => void;
}

/** Two-click constraint tools offered in the panel's constraint row. */
const CONSTRAINT_TOOLS: { key: ToolMode; label: string; hint: string }[] = [
  {
    key: "distance",
    label: "Distance",
    hint: "Choose two points to preserve their current distance",
  },
  { key: "horizontal", label: "Horiz", hint: "Choose two points to keep level" },
  { key: "vertical", label: "Vert", hint: "Choose two points to keep stacked" },
  { key: "coincident", label: "Coinc", hint: "Choose two points to join" },
  { key: "parallel", label: "∥ Edges", hint: "Choose two edges to keep parallel" },
  {
    key: "perpendicular",
    label: "⊥ Edges",
    hint: "Choose two edges to keep perpendicular",
  },
];

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
  const hasOperator = () => (profile()?.operators.length ?? 0) > 0;
  /** Distance chip currently open for numeric editing, by constraint identity. */
  const [editingConstraint, setEditingConstraint] = createSignal<number | null>(null);
  const commitConstraintValue = (identity: number, raw: string) => {
    const line = profile()?.line;
    const value = Number(raw);
    setEditingConstraint(null);
    if (line == null || raw.trim() === "" || !Number.isFinite(value)) return;
    props.onSetConstraintValue(line, identity, value);
  };
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

  // Close an open value editor when the selection moves elsewhere.
  createEffect(() => {
    selection();
    setEditingConstraint(null);
  });

  return (
    <Show when={profile()}>
      {(node) => (
        <aside class="sketch-panel" data-testid="sketch-panel">
          {/* The point count belongs on the kicker line, not beside the
              two-line title stack: as a sibling it had no shared baseline or
              centre with the stack it sat next to. */}
          <header>
            <span>
              <small>
                Sketch
                <b>{node().vertices.length} pts</b>
              </small>
              {node().name ?? "profile"}
            </span>
          </header>

          <Show
            when={node().constraints.length > 0 || node().operators.length > 0}
            fallback={<p class="empty-history">No constraints or operators yet</p>}
          >
            <div class="history-chips">
              <For each={node().constraints}>
                {(constraint, position) => {
                  const identity = () => constraint.index ?? position();
                  const editing = () =>
                    constraint.kind === "distance" &&
                    editingConstraint() === identity();
                  return (
                    <span
                      class="constraint-chip"
                      data-testid={`constraint-chip-${identity()}`}
                    >
                      <Show
                        when={editing()}
                        fallback={
                          <span
                            class="chip-label"
                            classList={{ editable: constraint.kind === "distance" }}
                            title={
                              constraint.kind === "distance"
                                ? "Click to edit the target distance"
                                : undefined
                            }
                            onClick={() => {
                              if (constraint.kind === "distance") {
                                setEditingConstraint(identity());
                              }
                            }}
                            data-testid={`constraint-label-${identity()}`}
                          >
                            {constraintLabel(constraint)}
                          </span>
                        }
                      >
                        <input
                          class="chip-value"
                          type="number"
                          step="0.1"
                          value={
                            typeof constraint.value === "number"
                              ? constraint.value
                              : ""
                          }
                          onChange={(event) =>
                            commitConstraintValue(
                              identity(),
                              event.currentTarget.value,
                            )
                          }
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              commitConstraintValue(
                                identity(),
                                event.currentTarget.value,
                              );
                            }
                            if (event.key === "Escape") {
                              event.stopPropagation();
                              setEditingConstraint(null);
                            }
                          }}
                          data-testid={`constraint-value-${identity()}`}
                        />
                      </Show>
                      <button
                        type="button"
                        class="chip-delete"
                        disabled={busy() || node().line === null}
                        onClick={() => {
                          const line = node().line;
                          if (line !== null) {
                            props.onDeleteConstraint(line, identity());
                          }
                        }}
                        title="Delete this constraint"
                        aria-label="Delete constraint"
                        data-testid={`constraint-delete-${identity()}`}
                      >
                        ×
                      </button>
                    </span>
                  );
                }}
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
            <For each={CONSTRAINT_TOOLS}>
              {(entry) => (
                <button
                  type="button"
                  class={tool() === entry.key ? "active" : ""}
                  disabled={busy()}
                  onClick={() => {
                    if (!isEdgeConstraintTool(entry.key)) {
                      setSelectionMode("vertex");
                    }
                    setTool(tool() === entry.key ? "select" : entry.key);
                  }}
                  title={entry.hint}
                  data-testid={`constraint-${entry.key}`}
                >
                  {entry.label}
                </button>
              )}
            </For>
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
            <button
              type="button"
              disabled={busy() || hasOperator()}
              onClick={props.onRevolve}
              title={
                hasOperator()
                  ? "This sketch already drives an operation"
                  : "Revolve around the sketch's vertical axis"
              }
              data-testid="sketch-revolve"
            >
              Revolve
            </button>
            <button
              type="button"
              class={pendingLoft()?.nodeId === node().id ? "active" : ""}
              disabled={busy() || hasOperator() || node().line === null}
              onClick={() => {
                if (pendingLoft()?.nodeId === node().id) {
                  setPendingLoft(null);
                  return;
                }
                const line = node().line;
                if (line === null) return;
                setSelectionMode("object");
                setPendingLoft({ nodeId: node().id, line });
              }}
              title={
                hasOperator()
                  ? "This sketch already drives an operation"
                  : "Loft to a second sketch with the same vertex count"
              }
              data-testid="sketch-loft"
            >
              Loft
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
