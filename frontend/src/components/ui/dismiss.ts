import { onCleanup } from "solid-js";

export interface DismissOptions {
  open: () => boolean;
  close: () => void;
  /** The element a press inside of does not dismiss. */
  within: () => Element | undefined;
  /** Also close on Escape, claimed in the capture phase so the viewer's Escape ladder does not fire too. */
  escape?: boolean;
}

/** Close a popover on a press outside it (and optionally on Escape). */
export function createDismiss(options: DismissOptions): void {
  const onPointerDown = (event: PointerEvent) => {
    const root = options.within();
    if (options.open() && root && !root.contains(event.target as Node)) options.close();
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key !== "Escape" || !options.open()) return;
    event.preventDefault();
    event.stopPropagation();
    options.close();
  };
  window.addEventListener("pointerdown", onPointerDown, true);
  if (options.escape) window.addEventListener("keydown", onKeyDown, true);
  onCleanup(() => {
    window.removeEventListener("pointerdown", onPointerDown, true);
    window.removeEventListener("keydown", onKeyDown, true);
  });
}
