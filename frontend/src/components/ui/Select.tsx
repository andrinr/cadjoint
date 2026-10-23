/**
 * A select whose open list is drawn like the app's menus (K5); a native
 * `<select>` opens the OS menu, which no token reaches.
 *
 * Keyboard: Enter, Space or ArrowDown opens; arrows move; Enter picks; Escape
 * closes and stops propagation (so the viewer's Escape ladder does not fire);
 * a letter jumps to the next option starting with it.
 */

import { For, Show, createMemo, createSignal, type JSX } from "solid-js";
import { createDismiss } from "./dismiss";
import "./select.css";

export interface SelectOption {
  value: string;
  label?: JSX.Element;
  /** Text used for type-ahead and the closed field when `label` is markup. */
  text?: string;
  disabled?: boolean;
}

export interface SelectProps {
  value: string;
  options: readonly SelectOption[];
  onChange: (value: string) => void;
  /** Shown in the closed field when no option matches `value`. */
  placeholder?: string;
  title?: string;
  disabled?: boolean;
  testId?: string;
  class?: string;
  "aria-label"?: string;
}

const textOf = (option: SelectOption) =>
  option.text ?? (typeof option.label === "string" ? option.label : option.value);

export function Select(props: SelectProps) {
  const [open, setOpen] = createSignal(false);
  const [cursor, setCursor] = createSignal(0);
  let root: HTMLDivElement | undefined;
  let trigger: HTMLButtonElement | undefined;

  const current = createMemo(() => props.options.find((option) => option.value === props.value));
  const indexOfValue = () =>
    Math.max(0, props.options.findIndex((option) => option.value === props.value));

  const close = (refocus = true) => {
    setOpen(false);
    if (refocus) trigger?.focus();
  };
  const show = () => {
    if (props.disabled) return;
    setCursor(indexOfValue());
    setOpen(true);
  };
  const pick = (option: SelectOption) => {
    if (option.disabled) return;
    close();
    if (option.value !== props.value) props.onChange(option.value);
  };
  const move = (step: number) => {
    const count = props.options.length;
    if (!count) return;
    let next = cursor();
    for (let tries = 0; tries < count; tries++) {
      next = (next + step + count) % count;
      if (!props.options[next].disabled) break;
    }
    setCursor(next);
  };

  createDismiss({ open, close: () => close(false), within: () => root });

  const onKeyDown = (event: KeyboardEvent) => {
    if (!open()) {
      if (event.key === "Enter" || event.key === " " || event.key === "ArrowDown") {
        event.preventDefault();
        show();
      }
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      move(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const option = props.options[cursor()];
      if (option) pick(option);
    } else if (event.key === "Tab") {
      close(false);
    } else if (event.key.length === 1) {
      const letter = event.key.toLowerCase();
      const count = props.options.length;
      for (let step = 1; step <= count; step++) {
        const index = (cursor() + step) % count;
        if (textOf(props.options[index]).toLowerCase().startsWith(letter)) {
          setCursor(index);
          break;
        }
      }
    }
  };

  return (
    <div
      ref={root}
      class={props.class ? `ui-select ${props.class}` : "ui-select"}
      classList={{ open: open() }}
    >
      <button
        ref={trigger}
        type="button"
        class="ui-select-field"
        disabled={props.disabled}
        title={props.title}
        aria-label={props["aria-label"]}
        aria-haspopup="listbox"
        aria-expanded={open()}
        data-testid={props.testId}
        data-value={props.value}
        onClick={() => (open() ? close() : show())}
        onKeyDown={onKeyDown}
      >
        <span>{current()?.label ?? current()?.value ?? props.placeholder ?? "—"}</span>
      </button>
      <Show when={open()}>
        <ul class="ui-select-list" role="listbox" onKeyDown={onKeyDown}>
          <For each={props.options}>
            {(option, index) => (
              <li
                role="option"
                aria-selected={option.value === props.value}
                aria-disabled={option.disabled}
                classList={{ cursor: index() === cursor(), disabled: !!option.disabled }}
                data-testid={props.testId ? `${props.testId}-option-${option.value}` : undefined}
                onPointerEnter={() => !option.disabled && setCursor(index())}
                onClick={(event) => {
                  // A field is often wrapped in its <label>; left uncancelled,
                  // the click would reach the label and re-open the list.
                  event.preventDefault();
                  pick(option);
                }}
              >
                <i>{option.value === props.value ? "✓" : ""}</i>
                <span>{option.label ?? option.value}</span>
              </li>
            )}
          </For>
        </ul>
      </Show>
    </div>
  );
}
