import { For, type JSX } from "solid-js";
import { Plate, Rule, type SectionDef } from "../kit";

/** Each kind of string, an example, and how it is set; the example is set that way. */
const STRINGS: [string, string, string, JSX.CSSProperties?][] = [
  [
    "furniture",
    "MESH · SOLVE · BOUNDARY CONDITION",
    "uppercase tracked mono",
    { "text-transform": "uppercase", "letter-spacing": "var(--tracking-xs)", "font-size": "var(--text-xs)" },
  ],
  ["commands in a list", "Save as… · Show generated WGSL", "mono, sentence case, one line"],
  ["program strings", "fin comb · sink-conduction · base_l", "verbatim mono"],
  ["sentences", "Drag a swatch onto an object", "sans, sentence case, no period on one clause", { "font-family": "var(--font-sans)" }],
  ["quantities", "1.00 m · 412.8 K · 2700 kg/m³", "SI, unit as a symbol after the value"],
  ["empty", "—", "an em dash; never 0, N/A or blank"],
];

export const language: SectionDef = {
  id: "language",
  title: "Language",
  body: () => (
    <Rule
      id="W1"
      why="Voice and case follow A8 and T2; what this rule adds is quantities and the empty value."
      examples={
        <Plate label="six kinds of string">
          <table class="dl-table">
            <thead>
              <tr>
                <th>kind</th>
                <th>example</th>
                <th>set as</th>
              </tr>
            </thead>
            <tbody>
              <For each={STRINGS}>
                {([kind, example, how, style]) => (
                  <tr>
                    <td>{kind}</td>
                    <td style={style ?? { "font-variant-numeric": "tabular-nums" }}>{example}</td>
                    <td class="prose">{how}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Plate>
      }
    >
      Each kind of string has one setting. Quantities are SI, the unit a symbol after the value; an empty value is an em
      dash.
    </Rule>
  ),
};
