/**
 * FEM simulation panel.
 *
 * Drives the /api/simulate endpoint: a probe meshes the current scene into
 * hexahedra and returns its boundary face groups; the user assigns boundary
 * conditions per group (hover tints the group's faces in the viewport),
 * sets material parameters, and runs a thermal or elastic solve. Results
 * render through the renderer's triangle-mesh pipeline with a viridis ramp
 * legend and a ParaView-style slicing slider.
 */

import { For, Show, createSignal, onCleanup, onMount } from "solid-js";
import * as api from "../api";
import {
  DEFAULT_SLICE,
  canSolve,
  formatScalar,
  rampCss,
  requestBcs,
  setAssignment,
  type BcAssignment,
  type BcMap,
  type SliceState,
} from "../simulation";
import { source } from "../state";
import type { SimulationFaceGroup } from "../types";
import type { Renderer } from "../viewer/renderer";

export interface SimulatePanelProps {
  renderer: Renderer;
}

const AXIS_LABELS = ["X", "Y", "Z"] as const;

type AssignmentChoice = "none" | "dirichlet" | "traction";

export function SimulatePanel(props: SimulatePanelProps) {
  const [expanded, setExpanded] = createSignal(false);
  const [kind, setKind] = createSignal<"thermal" | "elastic">("thermal");
  const [resolution, setResolution] = createSignal(20);
  const [groups, setGroups] = createSignal<SimulationFaceGroup[]>([]);
  const [bcs, setBcs] = createSignal<BcMap>({});
  const [material, setMaterial] = createSignal<Record<string, number>>({
    conductivity: 1,
    source: 0,
    youngs: 200,
    poisson: 0.3,
  });
  const [running, setRunning] = createSignal(false);
  const [error, setError] = createSignal("");
  const [unavailable, setUnavailable] = createSignal(false);
  const [result, setResult] = createSignal<{ field: string; range: [number, number] } | null>(
    null,
  );
  const [slice, setSlice] = createSignal<SliceState>({ ...DEFAULT_SLICE });

  const applySlice = (patch: Partial<SliceState>) => {
    const next = { ...slice(), ...patch };
    setSlice(next);
    props.renderer.setSimulationClip(next);
  };

  /** Mesh the scene and load its face-group catalog (no solve). */
  const probe = async () => {
    setRunning(true);
    setError("");
    setResult(null);
    try {
      const response = await api.simulate({
        source: source(),
        kind: "probe",
        resolution: resolution(),
        bcs: [],
        material: {},
      });
      if (!response.ok || !response.mesh) {
        setError(response.error ?? "Meshing failed.");
        return;
      }
      setGroups(response.mesh.groups);
      // Drop assignments whose group no longer exists at this resolution.
      const known = new Set(response.mesh.groups.map((group) => group.id));
      setBcs(
        Object.fromEntries(Object.entries(bcs()).filter(([id]) => known.has(id))),
      );
      props.renderer.setSimulationMesh(response.mesh);
      props.renderer.simulationActive = true;
      props.renderer.setSimulationClip(slice());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  const run = async () => {
    setRunning(true);
    setError("");
    try {
      const response = await api.simulate({
        source: source(),
        kind: kind(),
        resolution: resolution(),
        bcs: requestBcs(bcs(), kind()),
        material: material(),
      });
      if (response.error_kind === "fem_unavailable") {
        setUnavailable(true);
        setError(response.error ?? "The jax-fem extra is not installed.");
        return;
      }
      if (!response.ok || !response.mesh) {
        setError(response.error ?? "The solve failed.");
        return;
      }
      setGroups(response.mesh.groups);
      setResult({ field: response.field ?? "", range: response.mesh.range });
      props.renderer.setSimulationMesh(response.mesh);
      props.renderer.simulationActive = true;
      props.renderer.setSimulationClip(slice());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  const open = () => {
    setExpanded(true);
    if (groups().length === 0) void probe();
    else props.renderer.simulationActive = true;
  };

  const close = () => {
    props.renderer.setSimulationHighlight(null);
    props.renderer.simulationActive = false;
    setExpanded(false);
  };

  // The shell mounts this panel when the user enters simulate mode, so open
  // right away; leaving the mode unmounts it, which must hand the viewport
  // back to the raymarched scene.
  onMount(open);
  onCleanup(() => {
    props.renderer.setSimulationHighlight(null);
    props.renderer.simulationActive = false;
  });

  const choiceFor = (id: string): AssignmentChoice => bcs()[id]?.type ?? "none";

  const assign = (id: string, choice: AssignmentChoice) => {
    const assignment: BcAssignment | null =
      choice === "none"
        ? null
        : choice === "traction"
          ? { type: "traction", value: [0, 0, -1] }
          : { type: "dirichlet", value: kind() === "thermal" ? 100 : 0 };
    setBcs(setAssignment(bcs(), id, assignment));
  };

  const setBcValue = (id: string, value: number, component?: number) => {
    const current = bcs()[id];
    if (!current) return;
    if (current.type === "traction" && component !== undefined) {
      const vector = [...current.value] as [number, number, number];
      vector[component] = value;
      setBcs(setAssignment(bcs(), id, { type: "traction", value: vector }));
    } else if (current.type === "dirichlet") {
      setBcs(setAssignment(bcs(), id, { type: "dirichlet", value }));
    }
  };

  const setMaterialValue = (key: string, raw: string) => {
    const value = Number(raw);
    if (Number.isFinite(value)) setMaterial({ ...material(), [key]: value });
  };

  const hoverGroup = (group: SimulationFaceGroup | null) => {
    props.renderer.setSimulationHighlight(
      group ? { start: group.start, count: group.count } : null,
    );
  };

  const dirichletLabel = () => (kind() === "thermal" ? "Fixed temperature" : "Fixed support");

  return (
    <Show
      when={expanded()}
      fallback={
        <button
          type="button"
          class="sim-launch"
          onClick={open}
          title="Open the FEM simulation panel"
          data-testid="simulate-open"
        >
          Simulate
        </button>
      }
    >
      <aside class="sim-panel" data-testid="simulate-panel">
        <header>
          <span>
            <small>FEM</small>
            Simulate
          </span>
          <button
            type="button"
            onClick={close}
            title="Close simulation"
            aria-label="Close simulation"
            data-testid="simulate-close"
          >
            ×
          </button>
        </header>

        <div class="sim-row">
          <label>
            <span>Study</span>
            <select
              value={kind()}
              disabled={running()}
              onChange={(event) =>
                setKind(event.currentTarget.value as "thermal" | "elastic")
              }
              data-testid="simulate-kind"
            >
              <option value="thermal">Thermal</option>
              <option value="elastic">Elastic</option>
            </select>
          </label>
          <label>
            <span>Resolution</span>
            <input
              type="number"
              min="4"
              max="64"
              value={resolution()}
              disabled={running()}
              onChange={(event) => {
                const value = Math.round(Number(event.currentTarget.value));
                if (Number.isFinite(value)) {
                  setResolution(Math.min(64, Math.max(4, value)));
                }
              }}
              data-testid="simulate-resolution"
            />
          </label>
          <button
            type="button"
            disabled={running()}
            onClick={() => void probe()}
            title="Re-mesh the scene and refresh the face groups"
            data-testid="simulate-remesh"
          >
            Remesh
          </button>
        </div>

        <Show when={groups().length > 0}>
          <p class="sim-help">Face groups — hover to highlight, assign conditions:</p>
          <ul class="sim-groups" data-testid="simulate-groups">
            <For each={groups()}>
              {(group) => (
                <li
                  onMouseEnter={() => hoverGroup(group)}
                  onMouseLeave={() => hoverGroup(null)}
                  data-testid={`simulate-group-${group.id}`}
                >
                  <code>{group.id}</code>
                  <small>
                    {group.faces} faces · {formatScalar(group.area)} area
                  </small>
                  <select
                    value={choiceFor(group.id)}
                    disabled={running()}
                    onChange={(event) =>
                      assign(group.id, event.currentTarget.value as AssignmentChoice)
                    }
                    data-testid={`simulate-bc-${group.id}`}
                  >
                    <option value="none">Free</option>
                    <option value="dirichlet">{dirichletLabel()}</option>
                    <Show when={kind() === "elastic"}>
                      <option value="traction">Traction</option>
                    </Show>
                  </select>
                  <Show when={bcs()[group.id]?.type === "dirichlet" && kind() === "thermal"}>
                    <input
                      type="number"
                      value={bcs()[group.id]!.value as number}
                      disabled={running()}
                      onChange={(event) =>
                        setBcValue(group.id, Number(event.currentTarget.value))
                      }
                      title="Prescribed temperature"
                    />
                  </Show>
                  <Show when={bcs()[group.id]?.type === "traction"}>
                    <span class="sim-vector">
                      <For each={[0, 1, 2]}>
                        {(component) => (
                          <input
                            type="number"
                            value={(bcs()[group.id]!.value as number[])[component]}
                            disabled={running()}
                            onChange={(event) =>
                              setBcValue(
                                group.id,
                                Number(event.currentTarget.value),
                                component,
                              )
                            }
                            title={`Traction ${AXIS_LABELS[component]}`}
                          />
                        )}
                      </For>
                    </span>
                  </Show>
                </li>
              )}
            </For>
          </ul>
        </Show>

        <div class="sim-row sim-material">
          <Show
            when={kind() === "thermal"}
            fallback={
              <>
                <label>
                  <span>Young's</span>
                  <input
                    type="number"
                    value={material().youngs}
                    disabled={running()}
                    onChange={(event) => setMaterialValue("youngs", event.currentTarget.value)}
                    data-testid="simulate-youngs"
                  />
                </label>
                <label>
                  <span>Poisson</span>
                  <input
                    type="number"
                    step="0.05"
                    value={material().poisson}
                    disabled={running()}
                    onChange={(event) => setMaterialValue("poisson", event.currentTarget.value)}
                    data-testid="simulate-poisson"
                  />
                </label>
              </>
            }
          >
            <label>
              <span>Conductivity</span>
              <input
                type="number"
                value={material().conductivity}
                disabled={running()}
                onChange={(event) =>
                  setMaterialValue("conductivity", event.currentTarget.value)
                }
                data-testid="simulate-conductivity"
              />
            </label>
            <label>
              <span>Heat source</span>
              <input
                type="number"
                value={material().source}
                disabled={running()}
                onChange={(event) => setMaterialValue("source", event.currentTarget.value)}
                data-testid="simulate-source"
              />
            </label>
          </Show>
        </div>

        <button
          type="button"
          class="sim-run"
          disabled={running() || unavailable() || !canSolve(bcs(), kind())}
          onClick={() => void run()}
          title={
            canSolve(bcs(), kind())
              ? "Run the finite-element solve"
              : `Assign at least one ${dirichletLabel().toLowerCase()} first`
          }
          data-testid="simulate-run"
        >
          {running() ? "Solving…" : "Run"}
        </button>

        <Show when={result()}>
          {(current) => (
            <div class="sim-legend" data-testid="simulate-legend">
              <small>{current().field.replaceAll("_", " ")}</small>
              <div class="sim-ramp" style={{ background: rampCss() }} />
              <div class="sim-legend-values">
                <span>{formatScalar(current().range[0])}</span>
                <span>{formatScalar(current().range[1])}</span>
              </div>
            </div>
          )}
        </Show>

        <div class="sim-row sim-slice">
          <label class="sim-slice-toggle">
            <input
              type="checkbox"
              checked={slice().enabled}
              onChange={(event) => applySlice({ enabled: event.currentTarget.checked })}
              data-testid="simulate-slice-enabled"
            />
            <span>Slice</span>
          </label>
          <For each={[0, 1, 2] as const}>
            {(axis) => (
              <button
                type="button"
                class={slice().axis === axis ? "active" : ""}
                onClick={() => applySlice({ axis })}
                title={`Slice along ${AXIS_LABELS[axis]}`}
                data-testid={`simulate-slice-${AXIS_LABELS[axis].toLowerCase()}`}
              >
                {AXIS_LABELS[axis]}
              </button>
            )}
          </For>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={slice().fraction}
            disabled={!slice().enabled}
            onInput={(event) =>
              applySlice({ fraction: Number(event.currentTarget.value) })
            }
            data-testid="simulate-slice-fraction"
          />
        </div>

        <Show when={unavailable()}>
          <p class="sim-note" data-testid="simulate-unavailable">
            FEM solves need the optional jax-fem extra on the server. Install it with
            <code> pip install jaxcad[fem]</code> and restart the playground. Meshing and
            face groups still work without it.
          </p>
        </Show>
        <Show when={error() && !unavailable()}>
          <p class="sim-error" data-testid="simulate-error">
            {error()}
          </p>
        </Show>
      </aside>
    </Show>
  );
}
