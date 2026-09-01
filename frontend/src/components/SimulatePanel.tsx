/**
 * FEM simulation panel — a patch layer over studies declared in the code.
 *
 * The compile payload lists every ThermalStudy/ElasticStudy in the scene
 * program; this panel renders them and edits them exclusively through /patch
 * source operations (add_study, add_study_bc, set_study_value, …), the same
 * round-trip the sketch tools use. Boundary conditions target programmatic
 * vertex selections (side/box/sphere/halfspace) rather than face groups.
 * Solving posts the study's *name*; the server re-derives everything from the
 * declaration and returns a nodal field rendered through the renderer's
 * triangle-mesh pipeline with a viridis ramp and a ParaView-style slice.
 */

import { For, Index, Show, createSignal, onCleanup, onMount } from "solid-js";
import * as api from "../api";
import { DEFAULT_SLICE, formatScalar, rampCss, type SliceState } from "../simulation";
import { source, studies } from "../state";
import {
  BC_LABELS,
  addBcRequest,
  addStudyRequest,
  bcTypesFor,
  bcValue,
  defaultDraft,
  deleteBcRequest,
  deleteStudyRequest,
  describeSelection,
  setArgumentRequest,
  setBcValueRequest,
  studyArguments,
  type BcDraft,
  type BuilderSelectionKind,
} from "../studies";
import type { StudyPayload } from "../types";
import type { Renderer } from "../viewer/renderer";

export interface SimulatePanelProps {
  renderer: Renderer;
  /** Serialized /patch queue owned by the app shell. */
  onPatch: (body: Record<string, unknown>) => Promise<void>;
}

const AXIS_LABELS = ["X", "Y", "Z"] as const;
const SIDES = ["+x", "-x", "+y", "-y", "+z", "-z"] as const;
const SELECTION_KINDS: { value: BuilderSelectionKind; label: string }[] = [
  { value: "side", label: "Side" },
  { value: "box", label: "Box" },
  { value: "sphere", label: "Sphere" },
  { value: "halfspace", label: "Half-space" },
];

/** Numeric input helper: commit only finite values. */
const parse = (raw: string): number | null => {
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
};

export function SimulatePanel(props: SimulatePanelProps) {
  /** Which study's add-BC builder is open, by study index. */
  const [building, setBuilding] = createSignal<number | null>(null);
  const [draft, setDraft] = createSignal<BcDraft>(defaultDraft("thermal"));
  const [solving, setSolving] = createSignal<string | null>(null);
  const [error, setError] = createSignal("");
  const [unavailable, setUnavailable] = createSignal(false);
  const [result, setResult] = createSignal<{
    name: string;
    field: string;
    range: [number, number];
  } | null>(null);
  const [slice, setSlice] = createSignal<SliceState>({ ...DEFAULT_SLICE });

  const applySlice = (patch: Partial<SliceState>) => {
    const next = { ...slice(), ...patch };
    setSlice(next);
    props.renderer.setSimulationClip(next);
  };

  // Entering simulate mode mounts the panel; hand the viewport to the mesh
  // view, and give it back to the raymarched scene when the mode is left.
  onMount(() => {
    if (result()) props.renderer.simulationActive = true;
  });
  onCleanup(() => {
    props.renderer.simulationActive = false;
  });

  const solve = async (study: StudyPayload) => {
    setSolving(study.name);
    setError("");
    try {
      const response = await api.simulateStudy({
        source: source(),
        kind: "study",
        name: study.name,
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
      setResult({
        name: study.name,
        field: response.field ?? "",
        range: response.mesh.range,
      });
      props.renderer.setSimulationMesh(response.mesh);
      props.renderer.simulationActive = true;
      props.renderer.setSimulationClip(slice());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSolving(null);
    }
  };

  /** Route one patch body through the app queue, surfacing failures here. */
  const patch = async (body: Record<string, unknown>) => {
    setError("");
    await props.onPatch(body);
  };

  const openBuilder = (study: StudyPayload) => {
    setDraft(defaultDraft(study.kind));
    setBuilding(study.index);
  };

  const submitBc = async (study: StudyPayload) => {
    await patch(addBcRequest(study, draft()));
    setBuilding(null);
  };

  const setVector = (
    key: "minCorner" | "maxCorner" | "center" | "point" | "normal" | "vector",
    component: number,
    raw: string,
  ) => {
    const value = parse(raw);
    if (value === null) return;
    const next = { ...draft() };
    const vector = [...next[key]] as [number, number, number];
    vector[component] = value;
    next[key] = vector;
    setDraft(next);
  };

  const vectorRow = (
    key: "minCorner" | "maxCorner" | "center" | "point" | "normal",
    label: string,
  ) => (
    <label class="sim-builder-vector">
      <span>{label}</span>
      <Index each={[0, 1, 2]}>
        {(component) => (
          <input
            type="number"
            step="0.1"
            value={draft()[key][component()]}
            onChange={(event) => setVector(key, component(), event.currentTarget.value)}
            title={`${label} ${AXIS_LABELS[component()]}`}
          />
        )}
      </Index>
    </label>
  );

  return (
    <aside class="sim-panel" data-testid="simulate-panel">
      <header>
        <span>
          <small>FEM</small>
          Studies
        </span>
      </header>

      <Show
        when={studies().length > 0}
        fallback={
          <p class="sim-help" data-testid="simulate-empty">
            No studies declared. Add one — it becomes a ThermalStudy or
            ElasticStudy call in the code, and stays editable from either side.
          </p>
        }
      >
        <ul class="sim-studies" data-testid="simulate-studies">
          <For each={studies()}>
            {(study) => (
              <li class="sim-study" data-testid={`simulate-study-${study.name}`}>
                <div class="sim-study-head">
                  <span class={`sim-kind sim-kind-${study.kind}`}>{study.kind}</span>
                  <strong>{study.name}</strong>
                  <button
                    type="button"
                    class="sim-delete"
                    onClick={() => void patch(deleteStudyRequest(study))}
                    title="Delete this study from the code"
                    aria-label={`Delete study ${study.name}`}
                    data-testid={`simulate-delete-${study.name}`}
                  >
                    ×
                  </button>
                </div>

                <Show
                  when={study.editable}
                  fallback={
                    <p class="sim-note">
                      Defined dynamically in code — edit it there.
                    </p>
                  }
                >
                  <div class="sim-args">
                    <For each={studyArguments(study)}>
                      {(argument) => (
                        <label>
                          <span>{argument.key}</span>
                          <input
                            type="number"
                            value={argument.value}
                            disabled={solving() !== null}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) {
                                void patch(setArgumentRequest(study, argument.key, value));
                              }
                            }}
                            data-testid={`simulate-arg-${study.name}-${argument.key}`}
                          />
                        </label>
                      )}
                    </For>
                  </div>

                  <ul class="sim-bcs" data-testid={`simulate-bcs-${study.name}`}>
                    <For each={study.bcs}>
                      {(bc, bcIndex) => (
                        <li classList={{ "sim-bc-readonly": !bc.serializable }}>
                          <div class="sim-bc-main">
                            <span class="sim-bc-type">{BC_LABELS[bc.type]}</span>
                            <code title={describeSelection(bc.nodes)}>
                              {describeSelection(bc.nodes)}
                            </code>
                          </div>
                          <Show
                            when={bc.serializable}
                            fallback={<small class="sim-note">edit in code</small>}
                          >
                            <Show when={bcValue(bc) !== null}>
                              <Show
                                when={bc.type === "traction"}
                                fallback={
                                  <input
                                    type="number"
                                    value={bcValue(bc) as number}
                                    disabled={solving() !== null}
                                    onChange={(event) => {
                                      const value = parse(event.currentTarget.value);
                                      if (value !== null) {
                                        void patch(setBcValueRequest(study, bcIndex(), value));
                                      }
                                    }}
                                    title={BC_LABELS[bc.type]}
                                    data-testid={`simulate-bc-value-${study.name}-${bcIndex()}`}
                                  />
                                }
                              >
                                <span class="sim-vector">
                                  <Index each={bcValue(bc) as number[]}>
                                    {(component, index) => (
                                      <input
                                        type="number"
                                        value={component()}
                                        disabled={solving() !== null}
                                        onChange={(event) => {
                                          const value = parse(event.currentTarget.value);
                                          if (value === null) return;
                                          const vector = [...(bcValue(bc) as number[])];
                                          vector[index] = value;
                                          void patch(
                                            setBcValueRequest(study, bcIndex(), vector),
                                          );
                                        }}
                                        title={`Traction ${AXIS_LABELS[index]}`}
                                      />
                                    )}
                                  </Index>
                                </span>
                              </Show>
                            </Show>
                            <button
                              type="button"
                              class="sim-delete"
                              onClick={() => void patch(deleteBcRequest(study, bcIndex()))}
                              title="Remove this boundary condition"
                              aria-label="Remove boundary condition"
                              data-testid={`simulate-bc-delete-${study.name}-${bcIndex()}`}
                            >
                              ×
                            </button>
                          </Show>
                        </li>
                      )}
                    </For>
                  </ul>

                  <Show
                    when={building() === study.index}
                    fallback={
                      <button
                        type="button"
                        class="sim-add-bc"
                        onClick={() => openBuilder(study)}
                        data-testid={`simulate-add-bc-${study.name}`}
                      >
                        + Boundary condition
                      </button>
                    }
                  >
                    <div class="sim-builder" data-testid="simulate-builder">
                      <div class="sim-builder-row">
                        <label>
                          <span>Type</span>
                          <select
                            value={draft().bcType}
                            onChange={(event) =>
                              setDraft({
                                ...draft(),
                                bcType: event.currentTarget
                                  .value as BcDraft["bcType"],
                              })
                            }
                            data-testid="simulate-builder-type"
                          >
                            <For each={bcTypesFor(study.kind)}>
                              {(type) => <option value={type}>{BC_LABELS[type]}</option>}
                            </For>
                          </select>
                        </label>
                        <label>
                          <span>Select</span>
                          <select
                            value={draft().selectionKind}
                            onChange={(event) =>
                              setDraft({
                                ...draft(),
                                selectionKind: event.currentTarget
                                  .value as BuilderSelectionKind,
                              })
                            }
                            data-testid="simulate-builder-selection"
                          >
                            <For each={SELECTION_KINDS}>
                              {(kind) => <option value={kind.value}>{kind.label}</option>}
                            </For>
                          </select>
                        </label>
                      </div>

                      <Show when={draft().selectionKind === "side"}>
                        <div class="sim-sides" data-testid="simulate-builder-sides">
                          <For each={SIDES}>
                            {(side) => (
                              <button
                                type="button"
                                classList={{ active: draft().side === side }}
                                onClick={() => setDraft({ ...draft(), side })}
                              >
                                {side}
                              </button>
                            )}
                          </For>
                        </div>
                      </Show>
                      <Show when={draft().selectionKind === "box"}>
                        {vectorRow("minCorner", "Min")}
                        {vectorRow("maxCorner", "Max")}
                      </Show>
                      <Show when={draft().selectionKind === "sphere"}>
                        {vectorRow("center", "Center")}
                        <label class="sim-builder-vector">
                          <span>Radius</span>
                          <input
                            type="number"
                            step="0.1"
                            value={draft().radius}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) setDraft({ ...draft(), radius: value });
                            }}
                          />
                        </label>
                      </Show>
                      <Show when={draft().selectionKind === "halfspace"}>
                        {vectorRow("point", "Point")}
                        {vectorRow("normal", "Normal")}
                      </Show>

                      <Show when={draft().bcType === "dirichlet" || draft().bcType === "heat_flux"}>
                        <label class="sim-builder-vector">
                          <span>{BC_LABELS[draft().bcType]}</span>
                          <input
                            type="number"
                            value={draft().value}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) setDraft({ ...draft(), value });
                            }}
                            data-testid="simulate-builder-value"
                          />
                        </label>
                      </Show>
                      <Show when={draft().bcType === "traction"}>
                        <label class="sim-builder-vector">
                          <span>Vector</span>
                          <Index each={[0, 1, 2]}>
                            {(component) => (
                              <input
                                type="number"
                                step="0.1"
                                value={draft().vector[component()]}
                                onChange={(event) =>
                                  setVector("vector", component(), event.currentTarget.value)
                                }
                                title={`Traction ${AXIS_LABELS[component()]}`}
                              />
                            )}
                          </Index>
                        </label>
                      </Show>

                      <div class="sim-builder-actions">
                        <button
                          type="button"
                          onClick={() => void submitBc(study)}
                          data-testid="simulate-builder-add"
                        >
                          Add
                        </button>
                        <button type="button" onClick={() => setBuilding(null)}>
                          Cancel
                        </button>
                      </div>
                    </div>
                  </Show>
                </Show>

                <button
                  type="button"
                  class="sim-run"
                  disabled={solving() !== null || unavailable() || study.bcs.length === 0}
                  onClick={() => void solve(study)}
                  title={
                    study.bcs.length > 0
                      ? "Mesh the scene and run this study"
                      : "Add at least one boundary condition first"
                  }
                  data-testid={`simulate-run-${study.name}`}
                >
                  {solving() === study.name ? "Meshing + solving…" : "Solve"}
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>

      <div class="sim-row sim-add-study">
        <button
          type="button"
          onClick={() => void patch(addStudyRequest("thermal"))}
          data-testid="simulate-add-thermal"
        >
          + Thermal study
        </button>
        <button
          type="button"
          onClick={() => void patch(addStudyRequest("elastic"))}
          data-testid="simulate-add-elastic"
        >
          + Elastic study
        </button>
      </div>

      <Show when={result()}>
        {(current) => (
          <div class="sim-legend" data-testid="simulate-legend">
            <small>
              {current().name} · {current().field.replaceAll("_", " ")}
            </small>
            <div class="sim-ramp" style={{ background: rampCss() }} />
            <div class="sim-legend-values">
              <span>{formatScalar(current().range[0])}</span>
              <span>{formatScalar(current().range[1])}</span>
            </div>
          </div>
        )}
      </Show>

      <Show when={result()}>
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
            onInput={(event) => applySlice({ fraction: Number(event.currentTarget.value) })}
            data-testid="simulate-slice-fraction"
          />
        </div>
      </Show>

      <Show when={unavailable()}>
        <p class="sim-note" data-testid="simulate-unavailable">
          FEM solves need the optional jax-fem extra on the server. Install it
          with <code> pip install jaxcad[fem]</code> and restart the playground.
          Study declarations still compile and stay editable without it.
        </p>
      </Show>
      <Show when={error() && !unavailable()}>
        <p class="sim-error" data-testid="simulate-error">
          {error()}
        </p>
      </Show>
    </aside>
  );
}
