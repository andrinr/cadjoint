/**
 * A drawing of the viewport as the guide specifies it: the instrument face
 * the renderer paints, reproduced in SVG with the same tokens so each overlay
 * rule has a picture beside it.
 *
 * It is a specimen, not the renderer: geometry is a fixed front view of a fin
 * comb sketch, and every state the overlay table names (rest, hover,
 * selected, locked, hidden; free and held vertices) appears on it once.
 */

import { For, type JSX } from "solid-js";
import { fieldRampCss } from "../simColors";
import { v } from "./kit";

type Point = [number, number];

/**
 * A front view (looking along +Y) of a three-fin comb on a base plate, in
 * viewBox units. Fins and gaps are even, so the specimen reads as a part and
 * not as a puzzle.
 */
const COMB: Point[] = [
  [160, 250], [480, 250], [480, 222], [444, 222], [444, 110], [408, 110],
  [408, 222], [338, 222], [338, 110], [302, 110], [302, 222], [232, 222],
  [232, 110], [196, 110], [196, 222], [160, 222],
];

/** Which vertices are held (fixed by a constraint); the rest are free. */
const HELD = new Set([0, 1]);
const LOCKED_VERTEX = 0;
const SELECTED_VERTEX = 8;
/** Edge i runs from COMB[i] to COMB[i + 1]. */
const LOCKED_EDGE = 0;
const SELECTED_EDGE = 7;
const HOVER_EDGE = 12;
/** A construction axis through the middle fin: hidden inside the part, seen above it. */
const AXIS_SEEN: [Point, Point] = [[320, 78], [320, 110]];
const AXIS_HIDDEN: [Point, Point] = [[320, 110], [320, 250]];

/** The skeleton's state ladder (V1), or the ink one over a field (V2). */
function edgeTone(i: number, field: boolean): string {
  if (i === SELECTED_EDGE) return "ink";
  if (i === HOVER_EDGE) return field ? "ink-2" : "construction-strong";
  if (i === LOCKED_EDGE) return field ? "rule-strong" : "construction-locked";
  return field ? "ink-3" : "construction";
}

/** A stroke cased in sheet, as `fs_edge` draws it. */
function CasedLine(props: { a: Point; b: Point; tone: string; width: number }) {
  return (
    <g>
      <line
        x1={props.a[0]}
        y1={props.a[1]}
        x2={props.b[0]}
        y2={props.b[1]}
        stroke={v("surface-float")}
        stroke-opacity="0.75"
        stroke-width={props.width + 1.6}
      />
      <line x1={props.a[0]} y1={props.a[1]} x2={props.b[0]} y2={props.b[1]} stroke={v(props.tone)} stroke-width={props.width} />
    </g>
  );
}

/** World z = 0 and x = 0 on screen, where the graticule's axes run. */
const ORIGIN: Point = [272, 250];
const MINOR = 16;
const MAJOR_EVERY = 5;

/**
 * The graticule on the plane the camera faces: in a front view that is XZ,
 * so it is a square grid, a firmer line every fifth, and the two world axes
 * a step above that.
 */
function Graticule(props: { width: number; height: number }) {
  const lines: JSX.Element[] = [];
  const tone = (index: number) =>
    index === 0 ? "graticule-axis" : index % MAJOR_EVERY === 0 ? "graticule-axis" : "graticule-line";
  const opacity = (index: number) => (index === 0 ? 0.9 : index % MAJOR_EVERY === 0 ? 0.6 : 0.3);
  for (let i = -Math.ceil(ORIGIN[0] / MINOR); ORIGIN[0] + i * MINOR <= props.width; i++) {
    const x = ORIGIN[0] + i * MINOR;
    lines.push(
      <line x1={x} y1={0} x2={x} y2={props.height} stroke={v(tone(i))} stroke-opacity={opacity(i)} stroke-width="1" />,
    );
  }
  for (let j = -Math.ceil((props.height - ORIGIN[1]) / MINOR); ORIGIN[1] - j * MINOR >= 0; j++) {
    const y = ORIGIN[1] - j * MINOR;
    lines.push(
      <line x1={0} y1={y} x2={props.width} y2={y} stroke={v(tone(j))} stroke-opacity={opacity(j)} stroke-width="1" />,
    );
  }
  return <g shape-rendering="crispEdges">{lines}</g>;
}

function Handle(props: { at: Point; held: boolean; selected: boolean; locked?: boolean; field?: boolean }) {
  const [x, y] = props.at;
  // Every shape sits in a sheet casing, so it reads over any shade of the part.
  const casing = { stroke: v("surface-float"), "stroke-opacity": "0.75", "stroke-width": "1.5" };
  if (props.selected) return <circle cx={x} cy={y} r="5.5" fill={v("ink")} {...casing} />;
  if (props.held) {
    const tone = props.field
      ? props.locked
        ? "rule-strong"
        : "ink-3"
      : props.locked
        ? "construction-locked"
        : "construction";
    return <rect x={x - 3.5} y={y - 3.5} width="7" height="7" fill={v(tone)} {...casing} />;
  }
  return (
    <g>
      <circle cx={x} cy={y} r="5" fill={v("surface-float")} />
      <circle cx={x} cy={y} r="3.4" fill={v("surface-float")} stroke={v("param")} stroke-width="1.6" />
    </g>
  );
}

/** A dimension above two points: extension lines, the line, the value on a tab. */
function Dimension(props: { from: Point; to: Point; offset: number; text: string; violated?: boolean }) {
  const [x1, y1] = props.from;
  const [x2, y2] = props.to;
  const y = Math.min(y1, y2) - props.offset;
  const tone = props.violated ? v("danger") : v("ink-3");
  const width = props.text.length * 6.4 + 12;
  const mid = (x1 + x2) / 2;
  return (
    <g>
      <line x1={x1} y1={y1 - 6} x2={x1} y2={y - 4} stroke={tone} stroke-width="1" />
      <line x1={x2} y1={y2 - 6} x2={x2} y2={y - 4} stroke={tone} stroke-width="1" />
      <line x1={x1} y1={y} x2={x2} y2={y} stroke={tone} stroke-width="1" />
      <rect
        x={mid - width / 2}
        y={y - 20}
        width={width}
        height="16"
        fill={v("surface-float")}
        stroke={props.violated ? v("danger") : v("rule-strong")}
        stroke-width="1"
      />
      <text
        x={mid}
        y={y - 8.5}
        text-anchor="middle"
        fill={props.violated ? v("danger-ink") : v("ink")}
        style={{ font: "var(--weight-medium) var(--text-2xs) var(--font-mono)" }}
      >
        {props.text}
      </text>
    </g>
  );
}

/** Y points away from a front view: drawn as the "into the page" mark. */
function AxisAway(props: { at: Point; r: number; letter: string; labelDx: number }) {
  const [x, y] = props.at;
  const d = props.r * 0.6;
  return (
    <g stroke={v("axis-y")} stroke-width="1.5" fill="none">
      <circle cx={x} cy={y} r={props.r} />
      <line x1={x - d} y1={y - d} x2={x + d} y2={y + d} />
      <line x1={x + d} y1={y - d} x2={x - d} y2={y + d} />
      <text
        x={x + props.labelDx}
        y={y + 3}
        fill={v("axis-y")}
        stroke="none"
        style={{ font: "var(--weight-semibold) var(--text-3xs) var(--font-mono)" }}
      >
        {props.letter}
      </text>
    </g>
  );
}

/** The translate gizmo at the selection: X and Z in the view plane, Y away. */
function Gizmo(props: { at: Point }) {
  const [x, y] = props.at;
  const arrow = (dx: number, dy: number, token: string, letter: string, lx: number, ly: number) => (
    <g>
      <line x1={x} y1={y} x2={x + dx} y2={y + dy} stroke={v(token)} stroke-width="2" />
      <path
        d={dx ? `M${x + dx + 7},${y} l-8,-4 v8 z` : `M${x},${y + dy - 7} l-4,8 h8 z`}
        fill={v(token)}
      />
      <text
        x={x + dx + lx}
        y={y + dy + ly}
        text-anchor="middle"
        fill={v(token)}
        style={{ font: "var(--weight-semibold) var(--text-2xs) var(--font-mono)" }}
      >
        {letter}
      </text>
    </g>
  );
  return (
    <g>
      {arrow(44, 0, "axis-x", "X", 16, 4)}
      {arrow(0, -44, "axis-z", "Z", 0, -12)}
      <AxisAway at={[x + 14, y + 14]} r={5} letter="Y" labelDx={9} />
    </g>
  );
}

function Triad(props: { at: Point }) {
  const [x, y] = props.at;
  const axis = (dx: number, dy: number, token: string, letter: string) => (
    <g>
      <line x1={x} y1={y} x2={x + dx} y2={y + dy} stroke={v(token)} stroke-width="1.5" />
      <text
        x={x + dx * 1.5}
        y={y + dy * 1.5 + 3}
        text-anchor="middle"
        fill={v(token)}
        style={{ font: "var(--weight-semibold) var(--text-3xs) var(--font-mono)" }}
      >
        {letter}
      </text>
    </g>
  );
  return (
    <g>
      {axis(20, 0, "axis-x", "X")}
      {axis(0, -20, "axis-z", "Z")}
      <AxisAway at={[x, y]} r={4} letter="Y" labelDx={-14} />
    </g>
  );
}

export function ViewportSpecimen(props: { field?: boolean }) {
  const edges = COMB.map((point, index) => [point, COMB[(index + 1) % COMB.length]] as const);
  const path = `M${COMB.map((p) => p.join(",")).join("L")}Z`;
  return (
    <div class="dl-viewport">
      <svg viewBox="0 0 640 340" role="img" aria-label="Viewport specimen">
        <defs>
          <linearGradient id="dl-field" x1="0" y1="1" x2="0" y2="0">
            <For each={fieldStops()}>
              {(stop) => <stop offset={stop.offset} stop-color={stop.color} />}
            </For>
          </linearGradient>
        </defs>
        <Graticule width={640} height={340} />
        <g transform="translate(-48 0)">
        {/* The sketch plane: the ground a sketch is drawn on, so ink-3, not the skeleton's hue. */}
        <rect
          x="128"
          y="64"
          width="352"
          height="208"
          fill={v("ink-3")}
          fill-opacity="0.08"
          stroke={v("ink-3")}
          stroke-width="1"
        />
        <text
          x="128"
          y="58"
          fill={v("ink-2")}
          style={{ font: "var(--weight-semibold) var(--text-3xs) var(--font-mono)", "letter-spacing": "var(--tracking-3xs)" }}
        >
          FIN COMB · XZ
        </text>
        {props.field ? (
          <g>
            <path d={path} fill="url(#dl-field)" />
            {/* Hover over a field is an outline; the pixels keep their value. */}
            <path d="M302,222L338,222L338,110L302,110Z" fill="none" stroke={v("ink")} stroke-width="2" />
          </g>
        ) : (
          <path d={path} fill={v("ink")} fill-opacity="0.14" />
        )}
        {/* Behind the part the skeleton is a hidden line: no casing, dashed, at 45 %. */}
        <line
          x1={AXIS_HIDDEN[0][0]}
          y1={AXIS_HIDDEN[0][1]}
          x2={AXIS_HIDDEN[1][0]}
          y2={AXIS_HIDDEN[1][1]}
          stroke={v(props.field ? "ink-3" : "construction")}
          stroke-opacity="0.45"
          stroke-width="1.5"
          stroke-dasharray="5 3"
        />
        <CasedLine a={AXIS_SEEN[0]} b={AXIS_SEEN[1]} tone={props.field ? "ink-3" : "construction"} width={1.5} />
        <For each={edges}>
          {([a, b], index) => {
            const i = index();
            return <CasedLine a={a} b={b} tone={edgeTone(i, Boolean(props.field))} width={i === SELECTED_EDGE ? 2 : 1.5} />;
          }}
        </For>
        <Dimension from={[196, 110]} to={[232, 110]} offset={14} text="0.30 ≠ 0.32" violated />
        <Dimension from={[408, 110]} to={[444, 110]} offset={14} text="0.30" />
        <For each={COMB}>
          {(point, index) =>
            // The derived vertex has no handle (V1): only its skeleton shows.
            index() === LOCKED_VERTEX ? null : (
              <Handle
                at={point}
                held={HELD.has(index())}
                selected={index() === SELECTED_VERTEX}
                field={props.field}
              />
            )
          }
        </For>
        {props.field ? null : <Gizmo at={COMB[SELECTED_VERTEX]} />}
        </g>
        <Triad at={[598, 56]} />
        <rect x="0.5" y="0.5" width="639" height="339" fill="none" stroke={v("rule-heavy")} />
      </svg>
      <div class="readout">
        GRID <b>100 mm</b> · XZ &nbsp; VIEW <b>FRONT</b>
      </div>
      {props.field ? (
        <div class="legend">
          <span>293.2 K</span>
          <i style={{ background: fieldRampCss() }} />
          <span>412.8 K</span>
        </div>
      ) : null}
      <div class="titleblock">
        <span>FPS</span>
        <span>{props.field ? "60" : "idle"}</span>
        <span>CPU</span>
        <span>{props.field ? "1.8 ms/frame" : "—"}</span>
        <span>RES</span>
        <span>1696 × 1104 px</span>
        <span>NODES</span>
        <span>12</span>
      </div>
      <div class="hint">
        {props.field
          ? "Pick BC: click the mesh → sphere · Shift-drag → box"
          : "Distance: choose a second point in the same sketch"}
      </div>
    </div>
  );
}

/** The field ramp as SVG gradient stops, from the same polynomial. */
function fieldStops(): { offset: string; color: string }[] {
  const css = fieldRampCss(8);
  const colors = css.slice(css.indexOf(",") + 1, -1).split(/,\s*(?=rgb)/);
  return colors.map((color, index) => ({
    offset: `${(index / (colors.length - 1)) * 100}%`,
    color: color.trim(),
  }));
}
