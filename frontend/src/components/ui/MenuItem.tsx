/**
 * One row of a dropdown menu: the mark column, the label, and an optional
 * key or glyph hung right. Every other attribute (role, aria-*, handlers,
 * test id) passes through to the button.
 */

import { Show, splitProps, type JSX } from "solid-js";

export interface MenuItemProps extends JSX.ButtonHTMLAttributes<HTMLButtonElement> {
  label: JSX.Element;
  /** The leading column: ✓, •, or nothing. */
  mark?: JSX.Element;
  /** Right-aligned: the key that runs it, or a glyph such as ›. */
  aside?: JSX.Element;
  "data-testid"?: string;
}

export function MenuItem(props: MenuItemProps) {
  const [local, rest] = splitProps(props, ["label", "mark", "aside"]);
  return (
    <button type="button" role="menuitem" {...rest}>
      <i class="menu-check">{local.mark}</i>
      <span>{local.label}</span>
      <Show when={local.aside}>
        <small>{local.aside}</small>
      </Show>
    </button>
  );
}
