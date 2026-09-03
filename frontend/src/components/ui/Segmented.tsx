/**
 * The segmented control: a row of mutually exclusive buttons.
 *
 * Every panel that offers a small closed choice — shading mode, shadow
 * quality, element type, field to display, which view owns the viewport —
 * draws the same `.segmented` strip with an `active` button. This is that
 * strip, so the design pass has one component to restyle instead of six
 * hand-written copies.
 *
 * The active button is decided by comparing each option's `value` with the
 * control's `value`, so callers pass a signal read (`value={mode()}`) and the
 * highlight tracks it.
 */

import { For, type JSX } from "solid-js";

export interface SegmentedOption<T> {
  value: T;
  label: JSX.Element;
  title?: string;
  testId?: string;
  disabled?: boolean;
}

export interface SegmentedProps<T> {
  options: readonly SegmentedOption<T>[];
  /** The selected value; `null` selects nothing. */
  value: T | null;
  onSelect: (value: T) => void;
  /** Extra classes appended after "segmented". */
  class?: string;
  testId?: string;
  /** Disables every button; an option may still disable itself. */
  disabled?: boolean;
}

export function Segmented<T>(props: SegmentedProps<T>) {
  return (
    <div
      class={props.class ? `segmented ${props.class}` : "segmented"}
      data-testid={props.testId}
    >
      <For each={props.options}>
        {(option) => (
          <button
            type="button"
            classList={{ active: option.value === props.value }}
            disabled={option.disabled ?? props.disabled}
            title={option.title}
            onClick={() => props.onSelect(option.value)}
            data-testid={option.testId}
          >
            {option.label}
          </button>
        )}
      </For>
    </div>
  );
}
