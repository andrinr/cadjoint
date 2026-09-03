/**
 * Numeric form primitives shared by every panel.
 *
 * The panels are all thin editors over Python literals, so they all need the
 * same input: a number box that commits only finite values and leaves the
 * text alone otherwise. This module owns that behaviour (`parseNumber`), the
 * bare box (`NumberInput`), the labelled row (`NumberField`), and the
 * three-component triplet row (`VectorField`) used for positions, bounds and
 * resolutions.
 *
 * Nothing here knows about meshes, studies or render settings — callers pass
 * the value and receive a committed number. Styling lives in styles.css under
 * the classes these emit (`sim-builder-vector`, and the panels' bare labels),
 * so a design pass has one place to change the look of every numeric editor.
 */

import { Index, type JSX } from "solid-js";

/** Axis suffixes for the triplet rows' per-component tooltips. */
export const AXIS_LABELS = ["X", "Y", "Z"] as const;

/** Commit only finite values; anything else leaves the model untouched. */
export const parseNumber = (raw: string): number | null => {
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
};

export interface NumberInputProps {
  value: number;
  step?: string;
  min?: string;
  max?: string;
  disabled?: boolean;
  title?: string;
  testId?: string;
  /** Called on change with the parsed value; never with NaN. */
  onCommit: (value: number) => void;
}

/** A bare number box: parses on change, commits only finite values. */
export function NumberInput(props: NumberInputProps) {
  return (
    <input
      type="number"
      step={props.step}
      min={props.min}
      max={props.max}
      value={props.value}
      disabled={props.disabled}
      title={props.title}
      data-testid={props.testId}
      onChange={(event) => {
        const value = parseNumber(event.currentTarget.value);
        if (value !== null) props.onCommit(value);
      }}
    />
  );
}

export interface NumberFieldProps extends NumberInputProps {
  /** Text in the leading <span>; the argument name in most panels. */
  label: string;
  /** Extra class on the wrapping <label> (panels use "sim-builder-vector"). */
  class?: string;
}

/** A labelled number box — the standard "argument = value" panel row. */
export function NumberField(props: NumberFieldProps) {
  return (
    <label class={props.class}>
      <span>{props.label}</span>
      <NumberInput
        value={props.value}
        step={props.step}
        min={props.min}
        max={props.max}
        disabled={props.disabled}
        title={props.title}
        testId={props.testId}
        onCommit={props.onCommit}
      />
    </label>
  );
}

export interface VectorFieldProps {
  label: string;
  value: readonly number[];
  /** Defaults to the panels' triplet-row class. */
  class?: string;
  step?: string;
  disabled?: boolean;
  /** Per-component tooltip; defaults to "<label> X/Y/Z". */
  axisTitle?: (component: number) => string;
  /** Per-component data-testid; omitted when absent. */
  testId?: (component: number) => string;
  onCommit: (component: number, value: number) => void;
}

/** A labelled row of three number boxes (position, bounds, resolution, …). */
export function VectorField(props: VectorFieldProps) {
  const title = (component: number) =>
    props.axisTitle
      ? props.axisTitle(component)
      : `${props.label} ${AXIS_LABELS[component]}`;
  return (
    <label class={props.class ?? "sim-builder-vector"}>
      <span>{props.label}</span>
      <Index each={[0, 1, 2]}>
        {(component) => (
          <NumberInput
            value={props.value[component()]}
            step={props.step}
            disabled={props.disabled}
            title={title(component())}
            testId={props.testId?.(component())}
            onCommit={(value) => props.onCommit(component(), value)}
          />
        )}
      </Index>
    </label>
  );
}

export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  /** The dense variant used inside cards and result rows. */
  compact?: boolean;
  /** Extra classes appended after "switch"/"switch compact". */
  class?: string;
  disabled?: boolean;
  title?: string;
  testId?: string;
  /** Label content; rendered inside the switch's <span>. */
  children: JSX.Element;
}

/** The checkbox-backed toggle every panel uses for on/off settings. */
export function ToggleSwitch(props: ToggleSwitchProps) {
  const className = () =>
    ["switch", props.compact ? "compact" : null, props.class ?? null]
      .filter((part): part is string => part !== null)
      .join(" ");
  return (
    <label class={className()} title={props.title}>
      <input
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(event) => props.onChange(event.currentTarget.checked)}
        data-testid={props.testId}
      />
      <span>{props.children}</span>
    </label>
  );
}
