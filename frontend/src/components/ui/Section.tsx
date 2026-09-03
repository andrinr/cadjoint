/**
 * The section: this UI's one way of dividing a panel.
 *
 * Its name in tracked uppercase, then whatever the section counts, right-hung
 * against the rule that closes it. That is the whole idiom, and it exists here
 * rather than in each panel so that changing it is one edit.
 *
 * It used to open with a filled accent cell carrying a number, from a CSS
 * counter. The number is gone: a drawing numbers its zones so that a written
 * reference can point at one, and nothing in this app ever pointed at a
 * section by number, which left the loudest block on the panel carrying no
 * information. There is nothing to remove from this file's markup — the
 * numbers were a `::before` in `styles.css` and never a prop, which is what
 * made deleting them one edit.
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

/** The head of a section: `OBJECTIVE · 2`. */
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

/** A section: its head, then its body, closed by a rule. */
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
