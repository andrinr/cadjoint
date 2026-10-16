/**
 * A select that obeys the design language all the way down.
 *
 * A native `<select>` can be styled closed but not open: its list is the
 * operating system's menu, in the system font with the system highlight, and
 * no token reaches it. This one is a field (the sheet, a rule-strong edge, the
 * value in mono as the program writes it, an ink chevron) that opens a list
 * drawn like every other menu here (K5): 24px options in one column, the
 * current one checked in the leading column, hover a step down to
 * surface-bar-alt.
 *
 * Keyboard: Enter, Space or ArrowDown opens; ArrowUp/ArrowDown move; Enter
 * picks; Escape closes without changing anything (and stops there, so the
 * viewer's Escape ladder does not also fire); typing a letter jumps to the
 * next option starting with it.
 */

import { For, Show, createMemo, createSignal, onCleanup, type JSX } from "solid-js";

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
    Math.max(
      0,
      props.options.findIndex((option) => option.value === props.value),
    );

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

  const onPointerDown = (event: PointerEvent) => {
    if (open() && root && !root.contains(event.target as Node)) close(false);
  };
  window.addEventListener("pointerdown", onPointerDown, true);
  onCleanup(() => window.removeEventListener("pointerdown", onPointerDown, true));

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
        <span>{current() ? (current()!.label ?? current()!.value) : (props.placeholder ?? "—")}</span>
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
