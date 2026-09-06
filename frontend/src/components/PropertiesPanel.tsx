/**
 * The properties window: the selected construction element, argument by
 * argument, with every number and choice wired straight back to the source.
 *
 * The rows are the arguments the program *wrote*. A literal the contract
 * table lists gets a number box or a triplet; a material gets a choice among
 * the program's named materials; everything else — an expression, a
 * reference, a plane derived from a face — is shown as the source text it
 * is, read-only, because the patch layer would refuse to rewrite it and the
 * window never invents a request. Which rows appear is decided by the
 * `elements` payload of the last compile, so the window can only offer edits
 * the server already said it accepts.
 */

import { For, Show, createMemo } from "solid-js";
import {
  elements,
  inspected,
  materials,
  nodes,
  selection,
  setInspected,
} from "../state";
import {
  elementTitle,
  formatArgument,
  kindLabel,
  resolveSections,
  runtimePlaneRows,
} from "../properties";
import { NumberInput } from "./ui";
import type { ConstructionArgument, ConstructionElement, PatchAddress } from "../types";

export interface PropertiesPanelProps {
  /** Rewrite one literal argument at the published address. */
  onSetValue: (address: PatchAddress, value: number | number[]) => Promise<void>;
  /** Assign a named material at the published address. */
  onAssignMaterial: (address: PatchAddress, material: string) => Promise<void>;
}

/** How a number box steps: one part in a hundred of the value's own scale. */
function stepFor(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0 || !Number.isFinite(magnitude)) return "0.01";
  return String(Math.pow(10, Math.floor(Math.log10(magnitude)) - 2));
}

export function PropertiesPanel(props: PropertiesPanelProps) {
  const sections = createMemo(() =>
    resolveSections(elements(), nodes(), selection(), inspected()),
  );

  const testId = (element: ConstructionElement, argument: string, suffix = "") =>
    `prop-${element.kind}-${argument}${suffix}`;

  const numberRow = (element: ConstructionElement, argument: ConstructionArgument) => {
    const patch = argument.patch!;
    const value = argument.value as number;
    return (
      <label class="properties-row">
        <span>{argument.name}</span>
        <NumberInput
          value={value}
          step={stepFor(value)}
          testId={testId(element, argument.name)}
          title={
            argument.parameter
              ? `${argument.name} — follows the parameter ${argument.parameter}`
              : `${argument.name}`
          }
          onCommit={(next) => void props.onSetValue(patch, next)}
        />
        <Show when={argument.parameter}>
          <small class="properties-parameter" title="This value is a named parameter">
            {argument.parameter}
          </small>
        </Show>
      </label>
    );
  };

  const vectorRow = (element: ConstructionElement, argument: ConstructionArgument) => {
    const patch = argument.patch!;
    const value = argument.value as number[];
    return (
      <label class="properties-row properties-vector">
        <span>{argument.name}</span>
        <For each={[0, 1, 2]}>
          {(component) => (
            <NumberInput
              value={value[component]}
              step={stepFor(value[component])}
              testId={testId(element, argument.name, `-${component}`)}
              title={`${argument.name} ${"XYZ"[component]}`}
              onCommit={(next) => {
                const vector = [...value];
                vector[component] = next;
                void props.onSetValue(patch, vector);
              }}
            />
          )}
        </For>
        <Show when={argument.parameter}>
          <small class="properties-parameter" title="This value is a named parameter">
            {argument.parameter}
          </small>
        </Show>
      </label>
    );
  };

  const materialRow = (element: ConstructionElement, argument: ConstructionArgument) => {
    const patch = argument.patch!;
    return (
      <label class="properties-row">
        <span>material</span>
        <select
          data-testid={testId(element, "material")}
          value={String(argument.value ?? "")}
          onChange={(event) => void props.onAssignMaterial(patch, event.currentTarget.value)}
        >
          <For each={materials()}>
            {(material) => (
              <option value={material.name} selected={material.name === argument.value}>
                {material.name}
              </option>
            )}
          </For>
        </select>
      </label>
    );
  };

  const readOnlyRow = (element: ConstructionElement, argument: ConstructionArgument) => (
    <div
      class="properties-row properties-readonly"
      title={
        argument.kind === "expression"
          ? "An expression in the source: edit it in the code"
          : argument.kind === "reference"
            ? "A reference to another object in the program"
            : undefined
      }
    >
      <span>{argument.name}</span>
      <code data-testid={testId(element, argument.name)}>{formatArgument(argument)}</code>
    </div>
  );

  const row = (element: ConstructionElement, argument: ConstructionArgument) => {
    const patch = argument.patch;
    if (patch?.op === "assign_material") return materialRow(element, argument);
    if (patch?.op === "set_value" && Array.isArray(argument.value)) {
      return vectorRow(element, argument);
    }
    if (patch?.op === "set_value" && typeof argument.value === "number") {
      return numberRow(element, argument);
    }
    return readOnlyRow(element, argument);
  };

  return (
    <aside class="properties-panel" data-testid="properties-panel">
      <header>
        <span>
          <small>selection</small>
          Properties
        </span>
        <Show when={inspected()}>
          <button
            type="button"
            class="properties-follow"
            title="Go back to following the viewport selection"
            onClick={() => setInspected(null)}
          >
            follow selection
          </button>
        </Show>
      </header>
      <Show
        when={sections().length > 0}
        fallback={
          <p class="properties-empty" data-testid="properties-empty">
            Select an object in the viewport or the object tree to see its kind and the
            arguments it was written with. Numbers and choices edit the source; expressions
            are shown as written.
          </p>
        }
      >
        <For each={sections()}>
          {(element) => (
            <section
              class="properties-section"
              data-testid={`properties-${element.kind}`}
              data-element={element.id}
            >
              <h3>
                <small>{kindLabel(element)}</small>
                <b>{elementTitle(element)}</b>
                <i title="Where the call sits in scene.py">line {element.line}</i>
              </h3>
              <Show when={element.kind === "plane" && element.call !== "SketchPlane"}>
                <p class="properties-note" data-testid="properties-plane-derived">
                  Derived from other geometry: this plane follows its parent and cannot be
                  dragged. Edit the reference in the code to move it.
                </p>
              </Show>
              <For each={element.arguments}>{(argument) => row(element, argument)}</For>
              <For each={runtimePlaneRows(element, nodes())}>
                {(runtime) => (
                  <div class="properties-row properties-readonly" title="Evaluated at compile">
                    <span>{runtime.name}</span>
                    <code data-testid={testId(element, runtime.name)}>{runtime.text}</code>
                  </div>
                )}
              </For>
            </section>
          )}
        </For>
      </Show>
    </aside>
  );
}
