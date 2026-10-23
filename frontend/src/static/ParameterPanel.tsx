/**
 * The demo's parameter panel: every free design parameter of the scene, with
 * a slider and a number field per component, grouped by the construction
 * object whose handle moves it.
 *
 * A parameter some constraint holds is listed too, read-only, with the
 * constraint that holds it: the page has no constraint solver, so it cannot
 * move such a value without breaking the constraint.
 */
import { For, Index, Show, createMemo } from "solid-js";
import { hover, selection, setHover, setSelection } from "../state";
import { PanelHeader, Section } from "../components/ui";
import type { ConstructionNode, DemoParameter } from "../types";
import { clampComponent, editable, type Values } from "./parameters";

export interface ParameterPanelProps {
  parameters: readonly DemoParameter[];
  /**
   * The scene's construction tree as shipped: the groups' names and order.
   * Not the live tree, which every edit replaces: groups keyed on it would
   * remount every row, the slider under the pointer included, mid-drag.
   */
  construction: readonly ConstructionNode[];
  values: Values;
  /** How many parameters differ from the shipped scene. */
  edited: number;
  onChange: (name: string, value: number[]) => void;
  /** A slider is held (true) or let go (false). */
  onGesture: (active: boolean) => void;
  onReset: () => void;
}

const AXES2 = ["u", "v"];
const AXES3 = ["x", "y", "z"];

/** The component's name: u/v in a sketch plane, x/y/z in the world. */
function axisLabel(parameter: DemoParameter, index: number): string | null {
  if (parameter.components === 1) return null;
  return (parameter.components === 2 ? AXES2 : AXES3)[index] ?? String(index);
}

/** As many decimals as the step has, so the field shows what the slider moves by. */
const decimals = (step: number) => Math.max(0, Math.min(6, -Math.floor(Math.log10(step))));

function ParameterRow(props: {
  parameter: DemoParameter;
  value: readonly number[];
  owned: boolean;
  onChange: (value: number[]) => void;
  onGesture: (active: boolean) => void;
}) {
  const locked = () => !editable(props.parameter);
  const set = (index: number, component: number) => {
    const next = [...props.value];
    next[index] = clampComponent(props.parameter, index, component);
    props.onChange(next);
  };
  return (
    <div
      class="demo-parameter"
      classList={{ locked: locked(), owned: props.owned }}
      data-testid={`parameter-${props.parameter.name}`}
      onMouseEnter={() => props.parameter.owner && setHover({ nodeId: props.parameter.owner, vertexIndex: null })}
      onMouseLeave={() => props.parameter.owner && setHover(null)}
    >
      <div class="demo-parameter-head">
        <code>{props.parameter.name}</code>
        <Show when={props.parameter.unit}>
          <small>{props.parameter.unit}</small>
        </Show>
        <Show when={locked()}>
          <small class="demo-lock" title={props.parameter.drivenBy ?? undefined}>
            constraint-driven
          </small>
        </Show>
      </div>
      <Index each={props.value}>
        {(component, index) => (
          <label class="demo-component">
            <span>{axisLabel(props.parameter, index)}</span>
            <input
              type="range"
              min={props.parameter.lower[index]}
              max={props.parameter.upper[index]}
              step={props.parameter.step}
              value={component()}
              disabled={locked()}
              aria-label={`${props.parameter.name} ${axisLabel(props.parameter, index) ?? ""}`.trim()}
              data-testid={`slider-${props.parameter.name}-${index}`}
              onInput={(event) => set(index, Number(event.currentTarget.value))}
              onPointerDown={() => props.onGesture(true)}
              onPointerUp={() => props.onGesture(false)}
              onPointerCancel={() => props.onGesture(false)}
              onLostPointerCapture={() => props.onGesture(false)}
            />
            <input
              type="number"
              min={props.parameter.lower[index]}
              max={props.parameter.upper[index]}
              step={props.parameter.step}
              value={component().toFixed(decimals(props.parameter.step))}
              disabled={locked()}
              data-testid={`field-${props.parameter.name}-${index}`}
              onChange={(event) => {
                const value = Number(event.currentTarget.value);
                if (Number.isFinite(value)) set(index, value);
                else event.currentTarget.value = component().toFixed(decimals(props.parameter.step));
              }}
            />
          </label>
        )}
      </Index>
      <Show when={locked()}>
        <small class="demo-lock-reason">
          Held by {props.parameter.drivenBy}. Moving it needs the constraint solver, which runs in Python, so it
          stays at its shipped value here.
        </small>
      </Show>
    </div>
  );
}

export function ParameterPanel(props: ParameterPanelProps) {
  /** Parameters grouped by the object whose handle moves them, in scene order. */
  const groups = createMemo(() => {
    const names = new Map(props.construction.map((node) => [node.id, node.name ?? node.id]));
    const order = new Map(props.construction.map((node, index) => [node.id, index]));
    // A parameter no handle moves is grouped by its name's stem, so
    // `sketch0.v0` … `sketch0.v15` sit together rather than in one long list.
    const keyOf = (p: DemoParameter) => p.owner ?? `stem-${p.name.includes(".") ? p.name.split(".")[0] : "scene"}`;
    const groups = new Map<string, DemoParameter[]>();
    for (const parameter of props.parameters) {
      const key = keyOf(parameter);
      groups.set(key, [...(groups.get(key) ?? []), parameter]);
    }
    const rank = (key: string) => (key.startsWith("stem-") ? -1 : (order.get(key) ?? 0));
    return [...groups.entries()]
      .sort(([a], [b]) => rank(a) - rank(b))
      .map(([key, parameters]) => {
        const owner = key.startsWith("stem-") ? null : key;
        const stem = key.slice(5);
        return {
          key,
          owner,
          title: owner !== null ? (names.get(owner) ?? owner) : stem === "scene" ? "Scene" : stem,
          parameters,
          free: parameters.filter(editable).length,
        };
      });
  });
  const free = () => props.parameters.filter(editable).length;
  const active = (owner: string | null) =>
    owner !== null && (selection()?.nodeId === owner || hover()?.nodeId === owner);

  return (
    <aside class="demo-parameters" data-testid="parameter-panel">
      <PanelHeader kicker={`${free()} free · ${props.parameters.length - free()} held`} title="Parameters">
        <button
          type="button"
          disabled={props.edited === 0}
          onClick={() => props.onReset()}
          title="Back to the values the scene ships with"
          data-testid="parameters-reset"
        >
          Reset{props.edited ? ` · ${props.edited}` : ""}
        </button>
      </PanelHeader>
      <div class="demo-parameter-groups">
        <For each={groups()}>
          {(group) => (
            <Section
              title={group.title}
              count={`${group.free}/${group.parameters.length}`}
              testId={`parameter-group-${group.key}`}
              actions={
                <Show when={group.owner}>
                  <button
                    type="button"
                    class="demo-select"
                    onClick={() => setSelection({ nodeId: group.owner!, vertexIndex: null })}
                    title="Select it in the viewport"
                  >
                    select
                  </button>
                </Show>
              }
            >
              <For each={group.parameters}>
                {(parameter) => (
                  <ParameterRow
                    parameter={parameter}
                    value={props.values[parameter.name] ?? parameter.value}
                    owned={active(group.owner)}
                    onChange={(value) => props.onChange(parameter.name, value)}
                    onGesture={props.onGesture}
                  />
                )}
              </For>
            </Section>
          )}
        </For>
      </div>
    </aside>
  );
}
