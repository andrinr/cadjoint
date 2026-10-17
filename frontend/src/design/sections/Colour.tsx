import { For } from "solid-js";
import { BC_TYPE_COLORS, PROPOSAL_COLOR, cssColor, fieldRampCss, qualityRampCss } from "../../simColors";
import { CHROME, COLOR_ROLES } from "../../tokens";
import { sdfRampCss } from "../../viewer/sdfRamp";
import { DONT, Legend, Line, MiniPanel, Plate, Ratio, Rows, Rule, Seg, Swatch, v, type SectionDef } from "../kit";

const BC = Object.entries(BC_TYPE_COLORS) as [string, readonly [number, number, number]][];

/** The data roles: hues that live in the viewport and its legends only. */
const DATA_ROLES = [
  { role: "field", ramp: () => fieldRampCss(), token: "viridis", means: "a solved nodal field", carrier: "a legend with labelled ends" },
  { role: "quality", ramp: () => qualityRampCss(), token: "magma", means: "mesh quality", carrier: "a legend with labelled ends" },
  {
    role: "distance",
    ramp: () => sdfRampCss(),
    token: "sdf ramp",
    means: "signed distance: inside violet, outside ochre",
    carrier: "the zero isoline",
  },
  {
    role: "boundary",
    ramp: () => `linear-gradient(90deg, ${BC.map(([, c]) => cssColor(c)).join(", ")})`,
    token: "bc-*",
    means: "the kind of each boundary condition",
    carrier: "a swatch beside its name",
    zone: "viewport, BC list",
  },
  {
    role: "proposal",
    ramp: () => cssColor(PROPOSAL_COLOR),
    token: "proposal",
    means: "the live selection preview while building a BC",
    carrier: "exists only while building",
    zone: "viewport",
  },
];

/** The viridis ends and middle, for the C7 plates. */
const VIRIDIS = [
  [0, [0.267, 0.005, 0.329]],
  [0.5, [0.128, 0.567, 0.551]],
  [1, [0.993, 0.906, 0.144]],
] as const;

const FieldPatch = (props: { id: string; hover: "outline" | "tint" }) => (
  <svg viewBox="0 0 200 80" style={{ width: "100%" }}>
    <defs>
      <linearGradient id={props.id} x1="0" x2="1">
        <For each={VIRIDIS}>{([offset, rgb]) => <stop offset={String(offset)} stop-color={cssColor(rgb)} />}</For>
      </linearGradient>
    </defs>
    <rect x="10" y="10" width="180" height="60" fill={`url(#${props.id})`} />
    {props.hover === "outline" ? (
      <rect x="70" y="10" width="60" height="60" fill="none" stroke={v("ink")} stroke-width="2" />
    ) : (
      <rect x="70" y="10" width="60" height="60" fill={DONT.hoverTint} fill-opacity="0.45" />
    )}
  </svg>
);

const ROWS = [["sketch", "fin comb"]] as const;

export const colour: SectionDef = {
  id: "colour",
  title: "Colour",
  intro: "Paper, three inks, three rules, and a closed table of roles. Ratios are measured live on the ground and the sheet.",
  body: () => (
    <>
      <h3>The ground (A1, A7)</h3>
      <Rule id="C1" why="Hover is a step down because on paper pressure darkens; the seam to the viewport is a rule, not a step.">
        Chrome and viewport share one ground, with two sheets above it and no third: a surface within 3 ΔOKLab of
        another is that surface.
      </Rule>
      <div class="dl-swatches">
        <Swatch token="surface-base" job="the ground: page and viewport" ink="ink" />
        <Swatch token="surface-bar-alt" job="sub-bars, and hover" ink="ink" />
        <Swatch token="surface-float" job="the sheet: panels, popovers, dialogs" ink="ink" />
      </div>

      <h3>Ink and rules (A5)</h3>
      <Rule id="C2" why="An opacity fade has no contrast ratio that can be asserted, so text is never faded.">
        Three inks and no fourth. Three rules, one weight.
      </Rule>
      <div class="dl-swatches">
        <Swatch token="ink" job="text, selection" need={4.5} />
        <Swatch token="ink-2" job="secondary text, icons, hover" need={4.5} />
        <Swatch token="ink-3" job="labels, units, construction" need={4.5} />
        <Swatch token="rule" job="hairlines inside a panel" need={[1.4, 2.8]} />
        <Swatch token="rule-strong" job="control edges, the seam" need={3} />
        <Swatch token="rule-heavy" job="viewport frame, active line" need={3} />
      </div>

      <h3>Roles (A3, A4)</h3>
      <Rule id="C3" why="A new hue needs a new row here first. Data roles live in the viewport and its legends only.">
        The table is closed. Each role names what it means, what else carries it, and where it may appear.
      </Rule>
      <table class="dl-table">
        <thead>
          <tr>
            <th>role</th>
            <th>token</th>
            <th>means</th>
            <th>carrier</th>
            <th>zone</th>
            <th class="num">ground</th>
          </tr>
        </thead>
        <tbody>
          <For each={COLOR_ROLES}>
            {(role) => (
              <tr>
                <td>
                  <i class="dl-dot" style={{ background: v(role.token!) }} />
                  {role.role}
                </td>
                <td>
                  {role.token} <small style={{ color: v("ink-3") }}>{CHROME[role.token!]}</small>
                </td>
                <td class="prose">{role.means}</td>
                <td class="prose">{role.carrier}</td>
                <td>{role.zones.join(", ")}</td>
                <td class="num">
                  <Ratio fg={role.token!} bg="surface-base" need={role.token === "accent" ? undefined : 3} />
                  {role.token === "accent" ? <small style={{ color: v("ink-3") }}> fill only</small> : null}
                </td>
              </tr>
            )}
          </For>
          <For each={DATA_ROLES}>
            {(row) => (
              <tr>
                <td>
                  <i class="dl-dot" style={{ background: row.ramp() }} />
                  {row.role}
                </td>
                <td>{row.token}</td>
                <td class="prose">{row.means}</td>
                <td class="prose">{row.carrier}</td>
                <td>{row.zone ?? "viewport, legend"}</td>
                <td />
              </tr>
            )}
          </For>
        </tbody>
      </table>
      <div class="dl-examples">
        <Plate label="the data roles, as legends">
          <Legend name="field" from="293.2 K" to="412.8 K" ramp={fieldRampCss()} />
          <Legend name="quality" from="0.00" to="1.00" ramp={qualityRampCss()} />
          <Legend name="distance" from="−0.25 m" to="+0.25 m" ramp={sdfRampCss()} />
          <div style={{ "padding-top": "var(--inset)" }}>
            <For each={BC}>
              {([name, color]) => (
                <span class="dl-bc">
                  <i style={{ background: cssColor(color) }} />
                  {name}
                </span>
              )}
            </For>
          </div>
        </Plate>
      </div>

      <h3>No hue for these</h3>
      <Rule
        id="C4"
        why="Selection has to read over any field value, and only the heaviest ink does. A constraint holds geometry still, and stillness has no hue."
      >
        Selection is <code>ink</code>, hover <code>ink-2</code>, construction geometry <code>ink-3</code> (
        <code>rule</code> when locked). Constraints are <code>ink</code>; a violated one is <code>danger</code>.
      </Rule>

      <h3>The accent (A3, A5)</h3>
      <Rule
        id="C5"
        why="Under near-black type it clears AA; as text or a hairline on paper it fails even 3:1. One composition passes, so it is the only one."
        examples={
          <>
            <Plate kind="do" label="a filled block">
              <div class="dl-row-of">
                <button type="button" class="dl-btn primary">
                  Run
                </button>
                <Seg active="Sketch" />
              </div>
              <p class="dl-note">
                ink-on-accent on accent: <Ratio fg="ink-on-accent" bg="accent" need={4.5} />
              </p>
            </Plate>
            <Plate kind="dont" label="accent as ink or outline">
              <div class="dl-row-of">
                <span style={{ color: v("accent"), "font-weight": "var(--weight-semibold)" }}>SELECTION</span>
                <span style={{ padding: "var(--inset)", border: `var(--hairline) solid ${v("accent")}` }}>aluminium</span>
              </div>
              <p class="dl-note">
                accent on ground: <Ratio fg="accent" bg="surface-base" need={3} />
              </p>
            </Plate>
          </>
        }
      >
        The accent means <i>current</i> (B3) and is only ever a fill under <code>ink-on-accent</code> type, one amber
        that does not change under the pointer. <code>accent-press</code> is the one accent mark on paper, where the fill
        fails 3:1.
      </Rule>
      <Rule
        id="C6"
        why="If kickers, chips and outlines also take the accent, it stops meaning anything."
        examples={
          <>
            <Plate kind="do" label="one current thing per region">
              <MiniPanel kicker="selection" title="Properties">
                <Rows rows={ROWS} />
              </MiniPanel>
            </Plate>
            <Plate kind="dont" label="accent as a label of kind">
              <MiniPanel kicker="selection" title="Properties" kickerFill="accent">
                <Rows rows={ROWS} />
              </MiniPanel>
            </Plate>
          </>
        }
      >
        A screen holds few accent blocks: the mode, the focused panel's primary action, the selection, the armed tool.
      </Rule>

      <h3>Zones</h3>
      <Rule
        id="C7"
        why="A chrome hue inside the viewport could be read as a value, and a recoloured field pixel is a wrong reading."
        examples={
          <>
            <Plate kind="do" label="hover over a field is an outline" ground>
              <FieldPatch id="dl-zone-do" hover="outline" />
            </Plate>
            <Plate kind="dont" label="hover tints the reading" ground>
              <FieldPatch id="dl-zone-dont" hover="tint" />
            </Plate>
          </>
        }
      >
        Chrome hue stays outside the viewport; inside it only data and direction are coloured. A ramp in a panel is a
        legend at most 8px tall with both ends labelled. Hover and selection over a field are outlines, and no text sits
        on a data-dependent backdrop.
      </Rule>

      <h3>The editor</h3>
      <Rule
        id="C9"
        why="Syntax is not a role, so it has no hue (A3); the values are the subject (A1), so they carry the weight."
        examples={
          <>
            <Plate kind="do" label="values carry the line; one hue for free">
              <pre class="dl-code">
                <Line n={79}>
                  <span class="kwarg">density</span>
                  <span class="punct">=</span>
                  <span class="num">8940.0</span>
                  <span class="punct">,</span>
                </Line>
                <Line n={116}>
                  base_l <span class="punct">=</span> Vector2<span class="punct">(</span>
                  <span class="kwarg">value</span>
                  <span class="punct">=[</span>
                  <span class="num">-0.9</span>
                  <span class="punct">,</span> <span class="num">0.0</span>
                  <span class="punct">],</span> <span class="free">free=True</span>
                  <span class="punct">)</span>
                </Line>
                <Line n={117}>
                  <span class="com"># the deck stays level</span>
                </Line>
              </pre>
            </Plate>
            <Plate kind="dont" label="one ink for everything">
              <pre class="dl-code">
                <Line n={79}>density=8940.0,</Line>
                <Line n={116}>base_l = Vector2(value=[-0.9, 0.0], free=True)</Line>
                <Line n={117}># the deck stays level</Line>
              </pre>
            </Plate>
          </>
        }
      >
        Numbers, booleans and variable references are <code>ink</code>; declarations, calls, types, keywords and
        strings <code>ink-2</code>; argument names, punctuation and comments <code>ink-3</code>, comments italic. One
        weight throughout. The editor's hues are its roles:{" "}
        <code>free=True</code> in <code>param</code>, diagnostics in <code>danger</code> and <code>warn</code> with a
        gutter shape, and the accent behind the span a viewport edit is rewriting.
      </Rule>

      <Rule id="C8" why="ok and axis-y are both green, traction and axis-x both red: they never label the same thing in the same place.">
        Two roles may share a hue family only across zones or modes.
      </Rule>
    </>
  ),
};
