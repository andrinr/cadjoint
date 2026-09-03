/**
 * Card chrome shared by the mesh, study and optimization lists.
 *
 * All three panels render the same object: a `.sim-study` list item whose
 * head carries a kind chip, the declaration's name, and a delete button that
 * removes it from the source. `CardHeader` is that head; `SectionHead` is the
 * "title + inline add button" bar above a list.
 *
 * Only chrome belongs here — the body of each card stays with the feature
 * that owns it, because that is where the domain rules live.
 */

import { type JSX } from "solid-js";

export interface CardListProps {
  testId?: string;
  children: JSX.Element;
}

/** The list a panel's declaration cards live in. */
export function CardList(props: CardListProps) {
  return (
    <ul class="sim-studies" data-testid={props.testId}>
      {props.children}
    </ul>
  );
}

export interface CardProps {
  testId?: string;
  children: JSX.Element;
}

/** One declaration card: a mesh, a study, or an optimization. */
export function Card(props: CardProps) {
  return (
    <li class="sim-study" data-testid={props.testId}>
      {props.children}
    </li>
  );
}

export interface CardHeaderProps {
  /** Chip text: the declaration's kind ("mesh", "thermal", "adam", …). */
  kind: string;
  /** Modifier class on the chip, appended after "sim-kind". */
  kindClass: string;
  name: string;
  onDelete: () => void;
  deleteTitle: string;
  deleteAriaLabel: string;
  deleteTestId: string;
}

/** Kind chip + name + delete, the head of every declaration card. */
export function CardHeader(props: CardHeaderProps) {
  return (
    <div class="sim-study-head">
      <span class={`sim-kind ${props.kindClass}`}>{props.kind}</span>
      <strong>{props.name}</strong>
      <button
        type="button"
        class="sim-delete"
        onClick={props.onDelete}
        title={props.deleteTitle}
        aria-label={props.deleteAriaLabel}
        data-testid={props.deleteTestId}
      >
        ×
      </button>
    </div>
  );
}

export interface SectionHeadProps {
  title: string;
  testId?: string;
  /** Trailing controls, typically an inline "+ Thing" button. */
  children?: JSX.Element;
}

/** A panel section title with its inline actions. */
export function SectionHead(props: SectionHeadProps) {
  return (
    <div class="sim-section-head" data-testid={props.testId}>
      <b>{props.title}</b>
      {props.children}
    </div>
  );
}

export interface StatRowProps {
  /** Extra classes appended after "sim-stats". */
  class?: string;
  testId?: string;
  children: JSX.Element;
}

/** One line of "label value" readouts. */
export function StatRow(props: StatRowProps) {
  return (
    <div
      class={props.class ? `sim-stats ${props.class}` : "sim-stats"}
      data-testid={props.testId}
    >
      {props.children}
    </div>
  );
}

export interface StatProps {
  label: string;
  value: JSX.Element;
}

/** A single "label <b>value</b>" readout inside a StatRow. */
export function Stat(props: StatProps) {
  return (
    <span>
      {props.label} <b>{props.value}</b>
    </span>
  );
}
