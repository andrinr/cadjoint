/**
 * Window and dialog heads.
 *
 * `PanelHeader` is every dock window's head: the kicker (eyebrow) first, the
 * title second, then any actions. `DialogHeader` is a modal's title bar with
 * its Close button.
 */

import { type JSX } from "solid-js";

export interface PanelHeaderProps {
  kicker: JSX.Element;
  title: JSX.Element;
  /** Actions after the title stack. */
  children?: JSX.Element;
}

export function PanelHeader(props: PanelHeaderProps) {
  return (
    <header>
      <span>
        <small>{props.kicker}</small>
        {props.title}
      </span>
      {props.children}
    </header>
  );
}

export interface DialogHeaderProps {
  title: JSX.Element;
  onClose: () => void;
  closeDisabled?: boolean;
}

export function DialogHeader(props: DialogHeaderProps) {
  return (
    <header>
      <span>{props.title}</span>
      <button type="button" onClick={() => props.onClose()} disabled={props.closeDisabled}>
        Close
      </button>
    </header>
  );
}
