/**
 * The numbered section: this UI's one way of dividing a panel.
 *
 * A section is a zone of a drawing, so it is announced the way a drawing
 * announces a zone — a filled cell carrying its number, then the name in
 * tracked uppercase, then whatever the section counts, right-hung against the
 * rule that closes it. That is the whole idiom, and it exists here rather than
 * in each panel so that changing it is one edit.
 *
 * The number is not a prop. It comes from a CSS counter (`cj-section`, reset
 * per panel in styles.css) incremented by every section head in DOM order,
 * which means a section cannot be given the wrong number, reordering the
 * panel renumbers it, and — the reason it is done this way — a panel that
 * still writes its own section markup is picked up by the same selectors
 * without a line changing in that panel's file.
 *
 * `count` is the right-hung figure the reference puts at the end of the rule:
 * how many things the section holds, or the unit its readouts are in.
 */

import { Show, type JSX } from "solid-js";

export interface SectionHeadingProps {
  /** The section's name; rendered tracked and uppercase. */
  title: string;
  /** Right-hung figure: a count, a unit, a revision. */
  count?: JSX.Element;
  testId?: string;
  /** Trailing controls, typically an inline "+ Thing" button. */
  children?: JSX.Element;
}

/** The head of a numbered section: `01 · OBJECTIVE · 2`. */
export function SectionHeading(props: SectionHeadingProps) {
  return (
    <div class="section-head" data-testid={props.testId}>
      <b>{props.title}</b>
      <Show when={props.count !== undefined}>
        <span class="section-count">{props.count}</span>
      </Show>
      {props.children}
    </div>
  );
}

export interface SectionProps {
  title: string;
  count?: JSX.Element;
  testId?: string;
  /** Trailing controls in the head. */
  actions?: JSX.Element;
  children: JSX.Element;
}

/** A numbered section: its head, then its body, closed by a rule. */
export function Section(props: SectionProps) {
  return (
    <section class="section" data-testid={props.testId}>
      <SectionHeading title={props.title} count={props.count}>
        {props.actions}
      </SectionHeading>
      {props.children}
    </section>
  );
}
