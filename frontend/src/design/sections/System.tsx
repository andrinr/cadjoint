import { For } from "solid-js";
import { Plate, Rule, type SectionDef } from "../kit";

/**
 * The module as this section states it. The numbers are written here, not
 * read from tokens.ts, so the page states the system rather than quoting it.
 */
const SYSTEM = {
  heights: [
    { name: "tag", px: 16, formula: "L", job: "kind chips, marks that are not pressed" },
    { name: "cell", px: 24, formula: "L + 2 × gap", job: "the one interactive height: buttons, fields, menu items, rows, chips" },
    { name: "bar", px: 32, formula: "L + 2 × inset", job: "strips: tab strips, the tool rail, the viewport controls" },
    { name: "toolbar", px: 40, formula: "bar + 2 × gap", job: "the app toolbar, which holds bars" },
  ],
  distances: [
    { name: "gap", px: 4, formula: "u", job: "between cells of one group; text to its cell's edge, vertically" },
    { name: "inset", px: 8, formula: "2u", job: "inside a sheet; a cell's text to its edge, horizontally" },
    { name: "section", px: 16, formula: "4u = L", job: "between sections; the viewport's furniture from its edges" },
  ],
  type: [
    { role: "caps", size: 11, weight: 600, job: "every label of the instrument: tabs, buttons, column heads, kickers" },
    { role: "body", size: 12, weight: 400, job: "names, values 2700.0, units, menu items, the editor" },
    { role: "title", size: 15, weight: 600, job: "a panel's title, on a 20px line box" },
    { role: "prose", size: 12, weight: 400, job: "sentences, in sans, on a 20px line box" },
  ],
} as const;

const SYS_ROWS = [
  ["mesh", "sink-mesh", ""],
  ["conductivity", "401.0", "W/mK"],
  ["source", "12.5", "W"],
] as const;

export const system: SectionDef = {
  id: "system",
  title: "System",
  intro:
    "One unit and one line box; every other size is derived from them, none is chosen per component. The Variations drawer tries the alternatives.",
  body: () => (
    <>
      <Rule
        id="Y1"
        why="One number to remember (4) and one to see (16): a layout built from them lines up because nothing else is available."
        examples={
          <Plate label="the unit and the line box">
            <div class="dl-sys-row">
              <div class="dl-sys-unit" />
              <span class="t-body">u = 4px</span>
              <div class="dl-sys-line">
                <span class="t-body">a line of text, in its 16px box</span>
              </div>
              <span class="t-body">L = 4u = 16px</span>
            </div>
          </Plate>
        }
      >
        <b>u = 4px</b>, the unit. <b>L = 16px</b>, the line box: every single line of text sits in one, and an icon is
        one line box square.
      </Rule>
      <Rule
        id="Y2"
        why="Each height is a line box plus one distance on each side, so every height holds text on the same baseline."
        examples={
          <Plate label="the heights, built from the line box" ground>
            <div class="dl-sys-heights">
              <For each={SYSTEM.heights}>
                {(h) => (
                  <div class="dl-sys-height">
                    <div class="dl-sys-box" style={{ height: `${h.px}px` }}>
                      <span class="dl-sys-text" />
                    </div>
                    <code>{h.px}</code>
                    <span class="t-body">{h.name}</span>
                    <span class="t-body" style={{ color: "var(--ink-3)" }}>
                      {h.formula}
                    </span>
                    <span class="t-prose">{h.job}</span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>tag</b> 16 · <b>cell</b> 24 · <b>bar</b> 32 · <b>toolbar</b> 40. Anything clickable is a cell or larger (24
        is the WCAG 2.2 minimum target).
      </Rule>
      <Rule
        id="Y3"
        why="The distances double, so a 2px nudge or a 12px stack is a hesitation between two of them, and neither exists."
        examples={
          <Plate label="the three distances">
            <div class="dl-ruler">
              <For each={SYSTEM.distances}>
                {(d) => (
                  <div>
                    <code>{d.name}</code>
                    <b class="t-body">{d.px}px</b>
                    <span class="dl-sys-dist">
                      <i style={{ width: `${d.px * 4}px` }} />
                      <span class="t-prose">
                        {d.formula} · {d.job}
                      </span>
                    </span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>gap</b> 4 · <b>inset</b> 8 · <b>section</b> 16.
      </Rule>
      <Rule
        id="Y4"
        why="A value and a name are one role: a row of label, value and unit is told apart by case and ink, not by size."
        examples={
          <Plate label="four roles, three sizes, on the line box">
            <div class="dl-sys-type">
              <For each={SYSTEM.type}>
                {(t) => (
                  <div>
                    <code>{t.role}</code>
                    <code>
                      {t.size} · {t.weight}
                    </code>
                    <span
                      class={`dl-sys-sample ${t.role}`}
                      style={{ "font-size": `${t.size}px`, "font-weight": String(t.weight) }}
                    >
                      {t.job}
                    </span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>caps</b> 11 · <b>body</b> 12 · <b>title</b> 15 · <b>prose</b> 12 sans.
      </Rule>
      <Rule
        id="Y5"
        why="Every edge of the panel falls on the 4px grid behind it."
        examples={
          <Plate label="a panel, built from nothing but the module">
            <div class="dl-unit-grid" style={{ padding: "var(--section)" }}>
              <div class="dl-sys-panel">
                <div class="dl-sys-tabs">
                  <span class="active">studies</span>
                  <span>meshes</span>
                </div>
                <div class="dl-sys-body">
                  <div class="dl-sys-head">
                    <span class="dl-sys-kind">fem</span>
                    <span class="dl-sys-title">studies</span>
                  </div>
                  <div class="dl-sys-rows">
                    <For each={SYS_ROWS}>
                      {([label, value, unit]) => (
                        <div>
                          <span class="dl-sys-label">{label}</span>
                          <span class={unit ? "dl-sys-val num" : "dl-sys-val"}>{value}</span>
                          {unit ? <span class="dl-sys-unit-t">{unit}</span> : <span />}
                        </div>
                      )}
                    </For>
                  </div>
                  <div class="dl-sys-actions">
                    <span class="dl-sys-cell">mesh</span>
                    <span class="dl-sys-cell primary">solve</span>
                  </div>
                </div>
              </div>
            </div>
          </Plate>
        }
      >
        A panel: a <b>bar</b> of tabs, an <b>inset</b>, a head on a title line box, a <b>section</b>, <b>cell</b> rows,
        a <b>section</b>, a row of cells a <b>gap</b> apart.
      </Rule>
    </>
  ),
};
