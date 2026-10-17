import { For } from "solid-js";
import { TEXT_ROLES, type TextRole } from "../../tokens";
import { Plate, Rows, Rule, v, type SectionDef } from "../kit";

const JOBS: Record<TextRole, string> = {
  caps: "labels, kickers, tabs, buttons, menu titles",
  body: "names, values 2700.0, units, list rows, the editor",
  title: "a panel's title, and nothing else",
  prose: "Sentences: hints, help, empty states, dialog text.",
};

const ROLES = Object.entries(TEXT_ROLES) as [TextRole, (typeof TEXT_ROLES)[TextRole]][];

const READINGS = [
  ["peak", "412.80"],
  ["mean", "338.07"],
  ["min", "293.21"],
  ["Δ", "119.59"],
] as const;

export const type: SectionDef = {
  id: "type",
  title: "Type",
  body: () => (
    <>
      <Rule
        id="T0"
        why="Roles that would look alike are one role; hierarchy comes from a role and an ink, never from one adjusted property. test/tokens.test.ts fails on text set outside a role."
        examples={
          <Plate label="the four roles, as the app sets them">
            <div class="dl-type">
              <For each={ROLES}>
                {([role, spec]) => (
                  <div>
                    <code>{role}</code>
                    <code>
                      {spec.size} · {spec.weight}
                    </code>
                    <span class={`t-${role}`}>{JOBS[role]}</span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        Text is set in one of Y4's four roles and no other way. A role is a whole style: size, weight, leading, tracking,
        case and figures together.
      </Rule>
      <Rule
        id="T1"
        why="Between 9 and 13px a 1px step is about 10%, just above the visible threshold, so the roles skip 10px. Only capitals are tracked, more as the size falls."
      >
        The raw scale in tokens.ts (six sizes, four weights, six trackings) is what the roles are cut from. Components never
        use it directly.
      </Rule>
      <Rule
        id="T2"
        why="A program's string is a value, so it is quoted, not restyled: uppercasing it would change what it says."
        examples={
          <>
            <Plate kind="do" label="furniture tracked, names verbatim">
              <div class="dl-row-of">
                <span class="dl-tag">thermal</span>
                <span class="t-body">sink-conduction</span>
              </div>
            </Plate>
            <Plate kind="dont" label="the program's name restyled">
              <div class="dl-row-of">
                <span class="dl-tag">thermal</span>
                <span
                  style={{
                    "font-weight": "var(--weight-semibold)",
                    "text-transform": "uppercase",
                    "letter-spacing": "var(--tracking-xs)",
                  }}
                >
                  sink-conduction
                </span>
              </div>
            </Plate>
          </>
        }
      >
        Uppercase is for furniture only; strings from the program (<code>fin comb</code>, <code>sink-conduction</code>)
        are set exactly as written (A8).
      </Rule>
      <Rule
        id="T3"
        why="Numbers change while you watch them: tabular figures keep the row still, a shared right edge shows magnitude, the decimal point becomes a column."
        examples={
          <>
            <Plate kind="do" label="tabular, right-hung, unit after">
              <Rows rows={READINGS.map(([label, value]) => [label, value, "K"] as const)} />
            </Plate>
            <Plate kind="dont" label="proportional, left-aligned, unit in label">
              <div class="dl-stack" style={{ gap: "var(--gap)", "font-family": "var(--font-sans)" }}>
                <For each={READINGS}>
                  {([label, value]) => (
                    <div>
                      <span style={{ color: v("ink-3") }}>{label} (K): </span>
                      <b class="t-prose">{value}</b>
                    </div>
                  )}
                </For>
              </div>
            </Plate>
          </>
        }
      >
        Values are tabular, right-aligned on a shared column. The unit follows the value in <code>ink-3</code>, on the
        same baseline.
      </Rule>
    </>
  ),
};
