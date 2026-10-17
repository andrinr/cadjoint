import { For, type JSX } from "solid-js";
import { MenuItem, Plate, Rows, Rule, type SectionDef } from "../kit";

const FORMULAS: [string, string, () => JSX.Element][] = [
  ["sheet", "plane-sheet + line + inset", () => <div class="dl-sheet-demo" />],
  [
    "rule",
    "line, full width inside the sheet's inset",
    () => <div style={{ width: "calc(var(--unit) * 40)", "border-top": "var(--line)" }} />,
  ],
  [
    "cell",
    "a cell height + caps text; a line when it can be pressed, a block when current",
    () => (
      <span class="l-cluster">
        <button type="button" class="dl-btn h-sm">
          Mesh
        </button>
        <button type="button" class="dl-btn h-sm primary">
          Run
        </button>
      </span>
    ),
  ],
  [
    "field",
    "a sheet at cell height + body text, right-hung",
    () => <input class="dl-input" value="0.75" style={{ width: "calc(var(--unit) * 24)" }} />,
  ],
  [
    "chip",
    "tag height + body text on a sheet with a line; or a kind block with caps text",
    () => (
      <span class="l-cluster">
        <span class="dl-chip">fix · P1</span>
        <span class="dl-tag">thermal</span>
      </span>
    ),
  ],
  [
    "row",
    "l-row: caps label | body value | body unit, on the panel tracks",
    () => (
      <div style={{ width: "calc(var(--unit) * 64)" }}>
        <Rows rows={[["density", "2700", "kg/m³"]]} />
      </div>
    ),
  ],
  [
    "menu",
    "sheet + 24px body rows + a check column + a key column",
    () => (
      <div class="dl-menu" style={{ width: "calc(var(--unit) * 40)" }}>
        <MenuItem mark="✓" label="Editor" />
      </div>
    ),
  ],
];

export const primitives: SectionDef = {
  id: "primitives",
  title: "Primitives",
  intro: "Each primitive is a formula of the building blocks (B0); every component is a composition of primitives.",
  body: () => (
    <Rule
      id="P1"
      why="Two things that look alike are the same primitive, so they behave alike; a component that seems to need an eighth is the place to ask whether it is new."
      examples={
        <Plate label="each primitive, and what it is made of" ground>
          <table class="dl-table">
            <thead>
              <tr>
                <th>primitive</th>
                <th>made of</th>
                <th>specimen</th>
              </tr>
            </thead>
            <tbody>
              <For each={FORMULAS}>
                {([name, recipe, specimen]) => (
                  <tr>
                    <td>{name}</td>
                    <td class="prose">{recipe}</td>
                    <td>{specimen()}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Plate>
      }
    >
      <b>sheet</b>, <b>rule</b>, <b>cell</b>, <b>field</b>, <b>chip</b>, <b>row</b>, <b>menu</b>. None has a variant
      that carries no meaning.
    </Rule>
  ),
};
