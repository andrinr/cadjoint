/**
 * A drawing of the viewport as the guide specifies it: the instrument face
 * the renderer paints, reproduced in SVG with the same tokens so each overlay
 * rule has a picture beside it.
 *
 * It is a specimen, not the renderer: geometry is a fixed front view of a fin
 * comb sketch, and every state the overlay table names (rest, hover,
 * selected, locked; free and held vertices) appears on it once.
 */

import { For, type JSX } from "solid-js";
import { fieldRampCss } from "../simColors";
import { v } from "./kit";

type Point = [number, number];

const COMB: Point[] = [
  [200, 250], [440, 250], [440, 200], [400, 200], [400, 130], [370, 130],
  [370, 200], [330, 200], [330, 130], [300, 130], [300, 200], [240, 200],
  [240, 130], [210, 130], [210, 200], [200, 200],
];

/** Which vertices are held (fixed by a constraint); the rest are free. */
const HELD = new Set([0, 1]);
const SELECTED_VERTEX = 8;
const HOVER_EDGE = 11;
const SELECTED_EDGE = 4;

/** Graticule: two families of world lines on the floor plane, clipped. */
function Graticule(props: { width: number; height: number }) {
  const lines: JSX.Element[] = [];
  const spacing = 32;
  for (let x = -props.height * 2; x < props.width + props.height * 2; x += spacing) {
    const major = Math.round(x / spacing) % 5 === 0;
    lines.push(
      <line
        x1={x}
        y1={props.height}
        x2={x + props.height * 1.6}
        y2={0}
        stroke={v(major ? "graticule-axis" : "graticule-line")}
        stroke-width="1"
      />,
      <line
        x1={x}
        y1={0}
        x2={x + props.height * 1.6}
        y2={props.height}
        stroke={v(major ? "graticule-axis" : "graticule-line")}
        stroke-width="1"
      />,
    );
  }
  return <g opacity="0.55">{lines}</g>;
}

function Handle(props: { at: Point; held: boolean; selected: boolean; locked?: boolean }) {
  const [x, y] = props.at;
  if (props.selected) return <circle cx={x} cy={y} r="5.5" fill={v("ink")} />;
  if (props.held) {
    return (
      <rect
        x={x - 3.5}
        y={y - 3.5}
        width="7"
        height="7"
        fill={v(props.locked ? "rule" : "ink-3")}
      />
    );
  }
  return (
    <circle cx={x} cy={y} r="4" fill={v("surface-float")} stroke={v("ink-2")} stroke-width="1.5" />
  );
}

/** A dimension: extension ticks, a line, and the value on an opaque tab. */
function Dimension(props: { from: Point; to: Point; offset: number; text: string; violated?: boolean }) {
  const [x1, y1] = props.from;
  const [x2, y2] = props.to;
  const y = Math.min(y1, y2) - props.offset;
  const tone = props.violated ? v("danger") : v("ink-3");
  const width = props.text.length * 7 + 10;
  return (
    <g>
      <line x1={x1} y1={y1 - 4} x2={x1} y2={y - 4} stroke={tone} stroke-width="1" />
      <line x1={x2} y1={y2 - 4} x2={x2} y2={y - 4} stroke={tone} stroke-width="1" />
      <line x1={x1} y1={y} x2={x2} y2={y} stroke={tone} stroke-width="1" />
      <rect
        x={(x1 + x2) / 2 - width / 2}
        y={y - 9}
        width={width}
        height="18"
        fill={v("surface-float")}
        stroke={props.violated ? v("danger") : v("rule-strong")}
        stroke-width="1"
      />
      <text
        x={(x1 + x2) / 2}
        y={y + 4}
        text-anchor="middle"
        fill={props.violated ? v("danger-ink") : v("ink")}
        style={{ font: "var(--weight-medium) var(--text-2xs) var(--font-mono)" }}
      >
        {props.text}
      </text>
    </g>
  );
}

function Gizmo(props: { at: Point }) {
  const [x, y] = props.at;
  const arrow = (dx: number, dy: number, token: string, letter: string) => (
    <g>
      <line x1={x} y1={y} x2={x + dx} y2={y + dy} stroke={v(token)} stroke-width="2" />
      <circle cx={x + dx} cy={y + dy} r="3" fill={v(token)} />
      <text
        x={x + dx * 1.22}
        y={y + dy * 1.22 + 4}
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
      {arrow(56, 0, "axis-x", "X")}
      {arrow(34, -28, "axis-y", "Y")}
      {arrow(0, -56, "axis-z", "Z")}
    </g>
  );
}

function Triad(props: { at: Point }) {
  const [x, y] = props.at;
  const axis = (dx: number, dy: number, token: string, letter: string) => (
    <g>
      <line x1={x} y1={y} x2={x + dx} y2={y + dy} stroke={v(token)} stroke-width="1.5" />
      <text
        x={x + dx * 1.45}
        y={y + dy * 1.45 + 3}
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
      {axis(18, 8, "axis-x", "X")}
      {axis(12, -10, "axis-y", "Y")}
      {axis(0, -20, "axis-z", "Z")}
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
        <g transform="translate(-70 -24)">
        {/* The sketch plane: construction, so ink-3 at rest. */}
        <rect
          x="170"
          y="104"
          width="300"
          height="168"
          fill={v("ink-3")}
          fill-opacity="0.08"
          stroke={v("ink-3")}
          stroke-width="1"
        />
        <text
          x="174"
          y="98"
          fill={v("ink-2")}
          style={{ font: "var(--weight-semibold) var(--text-3xs) var(--font-mono)", "letter-spacing": "var(--tracking-3xs)" }}
        >
          FIN COMB · XZ
        </text>
        {props.field ? (
          <g>
            <path d={path} fill="url(#dl-field)" />
            {/* Hover over a field is an outline; the pixels keep their value. */}
            <path d="M300,200L330,200L330,130L300,130Z" fill="none" stroke={v("ink")} stroke-width="2" />
          </g>
        ) : (
          <path d={path} fill={v("ink")} fill-opacity="0.06" />
        )}
        <For each={edges}>
          {([a, b], index) => {
            const i = index();
            const tone = i === SELECTED_EDGE ? "ink" : i === HOVER_EDGE ? "ink-2" : i === 0 ? "rule" : "ink-3";
            const width = i === SELECTED_EDGE ? 2 : 1.5;
            return (
              <line
                x1={a[0]}
                y1={a[1]}
                x2={b[0]}
                y2={b[1]}
                stroke={v(tone)}
                stroke-width={width}
                stroke-dasharray={i === 0 ? "4 3" : undefined}
              />
            );
          }}
        </For>
        <Dimension from={[370, 130]} to={[400, 130]} offset={18} text="0.30" />
        <Dimension from={[210, 130]} to={[240, 130]} offset={18} text="0.30 ≠ 0.32" violated />
        <For each={COMB}>
          {(point, index) => (
            <Handle
              at={point}
              held={HELD.has(index())}
              selected={index() === SELECTED_VERTEX}
              locked={index() === 0}
            />
          )}
        </For>
        {props.field ? null : <Gizmo at={COMB[SELECTED_VERTEX]} />}
        </g>
        <Triad at={[596, 52]} />
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
        <span>SCENE</span>
        <span>heat_sink</span>
        <span>STUDY</span>
        <span>{props.field ? "sink-conduction" : "—"}</span>
        <span>REV</span>
        <span>14</span>
      </div>
      <div class="hint">
        <span class="dl-tag">{props.field ? "SIMULATE" : "SKETCH"}</span>
        {props.field
          ? "Hover a face to probe it · Click to pick it for a boundary condition"
          : "Drag a free vertex · G moves the selection · Esc steps back"}
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
