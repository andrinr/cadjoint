import { For } from "solid-js";
import { FlyoutSpecimen } from "../FlyoutSpecimen";
import { InputDemo, SelectSpecimen } from "../demos";
import { Chip, MenuItem, MiniPanel, Plate, Rows, Rule, Status, v, type SectionDef } from "../kit";

const BUTTONS = [
  ["primary", "primary", "Solve"],
  ["button", "", "Mesh"],
  ["quiet", "quiet", "Reset"],
  ["destructive", "destructive", "Delete study"],
  ["add", "add", "+ Study"],
] as const;

export const components: SectionDef = {
  id: "components",
  title: "Components",
  intro: "Compositions of the primitives (P1), arranged by L1–L7, set in the text roles (T0).",
  body: () => (
    <>
      <h3>Panel</h3>
      <Rule
        id="K1"
        why="Docked, not floated over the viewport, unless the user floats it."
        examples={
          <Plate label="tab strip, header, rows" ground>
            <MiniPanel kicker="library" title="Materials" tabs={["Materials", "Optimize"]}>
              <Rows
                free="free parameters"
                rows={[
                  ["aluminium", "1 applied"],
                  ["copper", "1 applied"],
                  ["free parameters", "17"],
                  ["density", "2700", "kg/m³", "aluminium_6061 · 293 K"],
                ]}
              />
            </MiniPanel>
          </Plate>
        }
      >
        A panel is a tab strip, then a sheet with an inset. The body opens with a kind kicker and the title; rows are
        label, value, unit on one baseline, and a second line hangs under the value.
      </Rule>

      <h3>Buttons</h3>
      <Rule
        id="K2"
        why="One primary per panel, so the accent stays the current action."
        examples={
          <Plate label="variants and states">
            <table class="dl-table">
              <thead>
                <tr>
                  <th>variant</th>
                  <th>rest</th>
                  <th>pressed</th>
                  <th>disabled</th>
                </tr>
              </thead>
              <tbody>
                <For each={BUTTONS}>
                  {([name, cls, label]) => (
                    <tr>
                      <td>{name}</td>
                      <td>
                        <button type="button" class={`dl-btn ${cls}`}>
                          {label}
                        </button>
                      </td>
                      <td>
                        <button type="button" class={`dl-btn ${cls} pressed`}>
                          {label}
                        </button>
                      </td>
                      <td>
                        <button type="button" class={`dl-btn ${cls}`} disabled title="Nothing to solve: add a study first">
                          {label}
                        </button>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        Primary is the current block. A button is the sheet with a <code>rule-strong</code> edge; hover is the step,
        pressed an <code>ink</code> edge. A disabled button keeps its shape, loses its edge contrast, and says why in its
        tooltip.
      </Rule>

      <h3>Inputs</h3>
      <Rule
        id="K3"
        why="Invalid text is kept as typed so it can be fixed, and nothing is written (F9)."
        examples={
          <Plate kind="try" label="type a non-number into base width and press Enter">
            <InputDemo />
          </Plate>
        }
      >
        An input is the sheet with a <code>rule-strong</code> edge and right-hung body text; a select is the same field
        with an ink chevron. Focus is the active line; invalid is a <code>danger</code> edge; an expression is read-only
        and pending (E4).
      </Rule>

      <h3>Chips and badges</h3>
      <Rule
        id="K4"
        examples={
          <Plate label="kind, value, violated, count, free count">
            <div class="dl-row-of">
              <span class="dl-tag">thermal</span>
              <span class="dl-tag">elastic</span>
              <Chip>distance · P1–P2 = 1.8</Chip>
              <Chip violated>equal · P1P15–P2P4 ≠ 0.02</Chip>
              <span class="dl-badge" title="3 constraints">
                3
              </span>
              <span class="dl-badge free" title="17 free parameters">
                17
              </span>
            </div>
          </Plate>
        }
      >
        A kind chip is the kind block. A value chip is the sheet with a <code>rule</code> edge, the value as written, and
        a × to remove it. A count badge is <code>ink</code> on the step; the free-parameter count is the one badge with a
        hue.
      </Rule>

      <Rule
        id="K4b"
        why="A feature that runs on the approximate path in the public tier must say so where it is chosen, not degrade silently."
        examples={
          <Plate label="tier chip: private tier, public tier, the reason under a disabled control">
            <div class="dl-row-of">
              <span class="tier-tag">Exact surfaces · private tier</span>
              <span class="tier-tag tier-public">Faceted · public tier</span>
            </div>
            <small class="tier-reason">
              STEP export is faceted in the public tier: triangles from dual contouring.
            </small>
          </Plate>
        }
      >
        A tier chip is a value chip that states what a feature is in this install: what you get, then where it comes
        from. The public form has a dashed edge, still the one 1px weight and no hue. Where the exact path is missing,
        one line of <code>ink-3</code> under the control gives the server's reason (<code>cadjoint.tier.message</code>),
        and a control that cannot run is disabled rather than hidden.
      </Rule>

      <h3>Menus</h3>
      <Rule
        id="K5"
        why="A fixed leading column keeps every label aligned whether or not it carries a check or a dot."
        examples={
          <Plate label="a menu, opened from the bar" ground menu>
            <div class="dl-menubar">
              <button type="button">File</button>
              <button type="button">Edit</button>
              <div class="dl-menubar-item">
                <button type="button" class="open" aria-expanded="true">
                  Window
                </button>
                <div class="dl-menu dl-dropdown" role="menu">
                  <MenuItem mark="●" label="Model mode" keys="Esc" />
                  <MenuItem label="Sketch mode" keys="M" />
                  <hr />
                  <MenuItem mark="✓" label="Editor" />
                  <MenuItem label="Processes" />
                  <MenuItem
                    label="Float results"
                    disabled
                    title="Results has nothing to show until a study is solved"
                  />
                  <hr />
                  <MenuItem label="Reset layout" />
                </div>
              </div>
              <button type="button">Help</button>
            </div>
          </Plate>
        }
      >
        A menu is the sheet with a <code>rule-strong</code> edge and one column of cells, shortcuts right-hung in{" "}
        <code>ink-3</code>. Titles are caps; items are commands in body, one line each; section labels are kickers. The
        open title is the current block.
      </Rule>

      <h3>Notices</h3>
      <Rule
        id="K6"
        why="A timer decides when you have read something; feedback stays where its cause is until the cause is gone."
        examples={
          <>
            <Plate kind="do" label="in place, in the program's words">
              <div class="dl-notice error">
                <b>Error</b>
                <p>
                  Line 118: <code>Solidd</code> is not defined. Did you mean <code>Solid</code>?
                </p>
                <button type="button" class="dl-btn quiet h-sm">
                  Go to line
                </button>
                <details>
                  <summary>Traceback</summary>
                  <pre>NameError: name 'Solidd' is not defined</pre>
                </details>
              </div>
            </Plate>
            <Plate kind="dont" label="a toast with a traceback">
              <div
                style={{
                  padding: "var(--inset)",
                  background: v("ink"),
                  color: v("surface-float"),
                  "border-radius": "6px",
                  "font-size": "var(--text-2xs)",
                }}
              >
                Traceback (most recent call last): File "worker.py", line 88, in compile … NameError: name 'Solidd' is not
                defined
                <div style={{ "margin-top": "var(--gap)", color: v("ink-3") }}>disappears in 4 s</div>
              </div>
            </Plate>
          </>
        }
      >
        No toasts on a timer. Feedback goes in place, in the status line (the whole program), or in the viewport's
        notice bar (the whole picture). It names the line, says what to do, and keeps the traceback behind a disclosure.
      </Rule>

      <h3>Status and empty states</h3>
      <Rule
        id="K7"
        examples={
          <>
            <Plate label="the status line">
              <div class="dl-stack">
                <Status>compiled · rev 14 · 1.2 s</Status>
                <Status stale>out of date · compiling rev 15</Status>
              </div>
            </Plate>
            <Plate label="an empty state">
              <div class="dl-empty">
                <p>
                  No results yet. <code>sink-conduction</code> has not been solved.
                </p>
                <div>
                  <button type="button" class="dl-btn primary h-sm">
                    Solve
                  </button>
                </div>
              </div>
            </Plate>
          </>
        }
      >
        The status line says whether the screen matches the code. An empty state names the missing precondition in the
        program's terms and offers the action.
      </Rule>

      <h3>Flyouts and submenus</h3>
      <Rule
        id="K8a"
        why="The notch says where the flyout came from and the fill which cell owns it; the placement is A12."
        examples={
          <Plate label="a tool group, open" ground>
            <FlyoutSpecimen />
          </Plate>
        }
      >
        <b>A tool group opens as a flyout:</b> a separate sheet a section out from the rail, its cells on the parent's
        row at the parent's size, a notch pointing at the parent, the parent marked by its fill.
      </Rule>
      <Rule
        id="K8"
        why="A submenu is a flyout for a menu item: same row, says where it came from (A12)."
        examples={
          <Plate label="a menu with a submenu open beside its parent" ground>
            <div class="dl-row-of" style={{ "align-items": "flex-start", gap: "0" }}>
              <div class="dl-menu" style={{ width: "calc(var(--unit) * 52)" }}>
                <MenuItem label="Create" keys="›" />
                <MenuItem label="Shape" keys="›" hover />
                <MenuItem label="Constrain" keys="›" />
              </div>
              <div class="dl-menu" style={{ width: "calc(var(--unit) * 44)", "margin-top": "calc(var(--unit) * 6)" }}>
                <MenuItem label="Extrude" keys="E" />
                <MenuItem label="Revolve" />
                <MenuItem label="Loft" />
              </div>
            </div>
          </Plate>
        }
      >
        A submenu opens to the side of its parent item, top edge on the parent's row, with a › where the parent's
        shortcut would be.
      </Rule>

      <h3>Platform controls</h3>
      <Rule
        id="K9"
        why="A native select's open list is the operating system's menu: no token reaches it, so it is outside A3 and A5."
        examples={
          <Plate kind="try" label="the playground's Select">
            <SelectSpecimen />
          </Plate>
        }
      >
        A platform control whose look the tokens cannot reach (a select's list, a colour picker, a date field) is rebuilt
        from the primitives, not themed.
      </Rule>
    </>
  ),
};
