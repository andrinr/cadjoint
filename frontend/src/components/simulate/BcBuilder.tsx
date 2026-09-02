/**
 * The add-a-boundary-condition builder, opened inside one study's card.
 *
 * A BC is a node selection plus a value, and both halves are drafted here
 * before anything is written: the selection kind chooses which geometry
 * fields are shown (a named side, a box, a sphere, a half-space) and the BC
 * type chooses whether the value is a scalar or a traction vector. Nothing
 * commits until "Add", which turns the draft into an `add_study_bc` patch.
 *
 * The viewport is the other way in: while "Pick in viewport" is armed, a
 * click on the mesh proposes a sphere and a shift-drag proposes a box. Those
 * proposals only pre-fill this draft — the user still confirms here.
 */

import { For, Show } from "solid-js";
import { simView } from "../../state";
import {
  BC_LABELS,
  bcTypesFor,
  type BcDraft,
  type BuilderSelectionKind,
} from "../../studies";
import { AXIS_LABELS, NumberField, VectorField } from "../ui";
import type { SimulateController } from "./controller";
import type { StudyPayload } from "../../types";

const SIDES = ["+x", "-x", "+y", "-y", "+z", "-z"] as const;
const SELECTION_KINDS: { value: BuilderSelectionKind; label: string }[] = [
  { value: "side", label: "Side" },
  { value: "box", label: "Box" },
  { value: "sphere", label: "Sphere" },
  { value: "halfspace", label: "Half-space" },
];

export interface BcBuilderProps {
  sim: SimulateController;
  study: StudyPayload;
}

export function BcBuilder(props: BcBuilderProps) {
  const sim = () => props.sim;
  const draft = () => sim().draft();

  /** One of the draft's three-component geometry fields. */
  const vectorRow = (
    key: "minCorner" | "maxCorner" | "center" | "point" | "normal",
    label: string,
  ) => (
    <VectorField
      label={label}
      value={draft()[key]}
      step="0.1"
      testId={(component) => `simulate-builder-${key}-${component}`}
      onCommit={(component, value) => sim().setVector(key, component, value)}
    />
  );

  return (
    <div class="sim-builder" data-testid="simulate-builder">
      <div class="sim-builder-row">
        <label>
          <span>Type</span>
          <select
            value={draft().bcType}
            onChange={(event) =>
              sim().setDraft({
                ...draft(),
                bcType: event.currentTarget.value as BcDraft["bcType"],
              })
            }
            data-testid="simulate-builder-type"
          >
            <For each={bcTypesFor(props.study.kind)}>
              {(type) => <option value={type}>{BC_LABELS[type]}</option>}
            </For>
          </select>
        </label>
        <label>
          <span>Select</span>
          <select
            value={draft().selectionKind}
            onChange={(event) =>
              sim().setDraft({
                ...draft(),
                selectionKind: event.currentTarget.value as BuilderSelectionKind,
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

      {/* Viewport picking: while armed and a mesh is shown, a click proposes
          Nodes.sphere at the picked point and a shift-drag rectangle
          proposes Nodes.box. */}
      <Show when={simView()}>
        <button
          type="button"
          class="sim-pick"
          classList={{ active: sim().picking() }}
          onClick={() => sim().setPicking(!sim().picking())}
          title="Click the mesh to propose a sphere; shift-drag for a box"
          data-testid="simulate-builder-pick"
        >
          {sim().picking() ? "Picking… click the mesh" : "Pick in viewport"}
        </button>
      </Show>

      <Show when={draft().selectionKind === "side"}>
        <div class="sim-sides" data-testid="simulate-builder-sides">
          <For each={SIDES}>
            {(side) => (
              <button
                type="button"
                classList={{ active: draft().side === side }}
                onClick={() => sim().setDraft({ ...draft(), side })}
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
        <NumberField
          class="sim-builder-vector"
          label="Radius"
          step="0.1"
          value={draft().radius}
          testId="simulate-builder-radius"
          onCommit={(radius) => sim().setDraft({ ...draft(), radius })}
        />
      </Show>
      <Show when={draft().selectionKind === "halfspace"}>
        {vectorRow("point", "Point")}
        {vectorRow("normal", "Normal")}
      </Show>

      <Show when={draft().bcType === "dirichlet" || draft().bcType === "heat_flux"}>
        <NumberField
          class="sim-builder-vector"
          label={BC_LABELS[draft().bcType]}
          value={draft().value}
          testId="simulate-builder-value"
          onCommit={(value) => sim().setDraft({ ...draft(), value })}
        />
      </Show>
      <Show when={draft().bcType === "traction"}>
        <VectorField
          label="Vector"
          value={draft().vector}
          step="0.1"
          axisTitle={(component) => `Traction ${AXIS_LABELS[component]}`}
          onCommit={(component, value) => sim().setVector("vector", component, value)}
        />
      </Show>

      <div class="sim-builder-actions">
        <button
          type="button"
          onClick={() => void sim().submitBc(props.study)}
          data-testid="simulate-builder-add"
        >
          Add
        </button>
        <button type="button" onClick={sim().closeBuilder}>
          Cancel
        </button>
      </div>
    </div>
  );
}
