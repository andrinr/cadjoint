/**
 * Layout, drawn the way the people who use this tool read a layout: as a
 * dimensioned drawing. Every number on these drawings comes from the system
 * (§1); the drawings are generated, not illustrated, so a change to the
 * module redraws them.
 */

import { For, type JSX } from "solid-js";

const S = {
  unit: 4,
  line: 16,
  tag: 16,
  cell: 24,
  bar: 32,
  toolbar: 40,
  gap: 4,
  inset: 8,
  section: 16,
  title: 20,
  label: 96,
  unitTrack: 40,
  action: 24,
} as const;

const ink = (token: string) => `var(--${token})`;
const MONO = "var(--weight-medium) 9px var(--font-mono)";

/** A dimension line with end ticks and its value, horizontal or vertical. */
function Dim(props: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  text: string;
  side?: -1 | 1;
}): JSX.Element {
  const vertical = props.x1 === props.x2;
  const off = (props.side ?? -1) * 10;
  const ax1 = vertical ? props.x1 + off : props.x1;
  const ay1 = vertical ? props.y1 : props.y1 + off;
  const ax2 = vertical ? props.x2 + off : props.x2;
  const ay2 = vertical ? props.y2 : props.y2 + off;
  const mx = (ax1 + ax2) / 2;
  const my = (ay1 + ay2) / 2;
  const tick = (x: number, y: number) =>
    vertical ? (
      <line x1={x - 3} y1={y} x2={x + 3} y2={y} stroke={ink("ink-3")} stroke-width="1" />
    ) : (
      <line x1={x} y1={y - 3} x2={x} y2={y + 3} stroke={ink("ink-3")} stroke-width="1" />
    );
  return (
    <g>
      {/* extension lines */}
      <line
        x1={props.x1}
        y1={props.y1}
        x2={ax1}
        y2={ay1}
        stroke={ink("rule")}
        stroke-width="1"
      />
      <line
        x1={props.x2}
        y1={props.y2}
        x2={ax2}
        y2={ay2}
        stroke={ink("rule")}
        stroke-width="1"
      />
      <line x1={ax1} y1={ay1} x2={ax2} y2={ay2} stroke={ink("ink-3")} stroke-width="1" />
      {tick(ax1, ay1)}
      {tick(ax2, ay2)}
      <text
        x={vertical ? mx + (props.side ?? -1) * 4 : mx}
        y={vertical ? my + 3 : my - 4}
        text-anchor={vertical ? ((props.side ?? -1) < 0 ? "end" : "start") : "middle"}
        fill={ink("ink-2")}
        style={{ font: MONO }}
      >
        {props.text}
      </text>
    </g>
  );
}

/** A box: an edge, a plane, an optional caption inside. */
function Box(props: {
  x: number;
  y: number;
  w: number;
  h: number;
  plane?: string;
  line?: string;
  caption?: string;
  captionColor?: string;
  dashed?: boolean;
}): JSX.Element {
  return (
    <g>
      <rect
        x={props.x + 0.5}
        y={props.y + 0.5}
        width={props.w - 1}
        height={props.h - 1}
        fill={props.plane ? ink(props.plane) : "none"}
        stroke={props.line ? ink(props.line) : "none"}
        stroke-width="1"
        stroke-dasharray={props.dashed ? "3 2" : undefined}
      />
      {props.caption ? (
        <text
          x={props.x + S.inset}
          y={props.y + props.h / 2 + 3}
          fill={ink(props.captionColor ?? "ink-3")}
          style={{ font: MONO, "text-transform": "uppercase", "letter-spacing": "0.12em" }}
        >
          {props.caption}
        </text>
      ) : null}
    </g>
  );
}

/**
 * L1: the five levels, nested, each separated from its siblings in exactly
 * one way.
 */
export function LevelsDrawing() {
  const W = 520;
  // desk → window → sheet → block → list → row
  return (
    <svg viewBox={`0 0 ${W + 160} 300`} class="dl-drawing" role="img" aria-label="Levels">
      {/* desk: two windows sharing an edge */}
      <Box x={20} y={20} w={250} h={260} plane="plane-ground" line="line-edge" />
      <Box x={270} y={20} w={250} h={260} plane="plane-ground" line="rule-strong" />
      {/* window head bars */}
      <Box x={20} y={20} w={250} h={S.bar} plane="plane-ground" line="rule-strong" caption="window · bar 32" />
      <Box x={270} y={20} w={250} h={S.bar} plane="plane-ground" line="rule-strong" caption="window" />
      {/* sheet with inset */}
      <Box x={20} y={52} w={250} h={228} plane="plane-sheet" line="rule-strong" />
      <Box x={20 + S.inset} y={52 + S.inset} w={250 - 2 * S.inset} h={228 - 2 * S.inset} line="rule" dashed />
      {/* blocks a section apart */}
      <Box x={28} y={60} w={234} h={88} line="ink-3" caption="block" />
      <Box x={28} y={60 + 88 + S.section} w={234} h={96} line="ink-3" caption="block" />
      {/* a ruled list inside the second block */}
      <For each={[0, 1, 2]}>
        {(i) => (
          <line
            x1={28}
            x2={262}
            y1={60 + 88 + S.section + 24 + i * S.cell}
            y2={60 + 88 + S.section + 24 + i * S.cell}
            stroke={ink("rule")}
            stroke-width="1"
          />
        )}
      </For>
      <Dim x1={262} y1={148} x2={262} y2={164} text="section 16" side={1} />
      <Dim x1={20} y1={290} x2={28} y2={290} text="inset 8" />
      <Dim x1={28} y1={188} x2={28} y2={212} text="row 24" />
      {/* legend */}
      <g style={{ font: MONO }}>
        <For
          each={[
            ["desk", "windows share an edge"],
            ["window", "a bar of tabs, an edge"],
            ["sheet", "an inset all round"],
            ["block", "a section of space apart"],
            ["list", "a rule between rows"],
            ["row", "a cell high"],
          ]}
        >
          {([level, how], i) => (
            <g>
              <text x={540} y={40 + i() * 40} fill={ink("ink")} style={{ font: MONO, "text-transform": "uppercase", "letter-spacing": "0.12em" }}>
                {level}
              </text>
              <text x={540} y={54 + i() * 40} fill={ink("ink-3")} style={{ font: MONO }}>
                {how}
              </text>
            </g>
          )}
        </For>
      </g>
    </svg>
  );
}

/** L2 + L3: one panel on the panel grid, with its tracks and its rhythm. */
export function PanelAnatomy() {
  const W = 320;
  const x0 = 150;
  const y0 = 30;
  const inner = W - 2 * S.inset;
  const value = inner - S.label - S.unitTrack - S.action - 3 * S.inset;
  const tracks = [
    { name: "label", w: S.label },
    { name: "value", w: value },
    { name: "unit", w: S.unitTrack },
    { name: "action", w: S.action },
  ];
  let tx = x0 + S.inset;
  const trackX = tracks.map((t) => {
    const x = tx;
    tx += t.w + S.inset;
    return { ...t, x };
  });
  const head = S.tag + S.gap + S.title; // 40
  const yHead = y0 + S.bar + S.inset;
  const yRows = yHead + head + S.section;
  const rows = ["mesh", "conductivity", "source", "domain"];
  const yActions = yRows + rows.length * S.cell + S.section;
  const H = yActions + S.cell + S.inset - y0;
  return (
    <svg viewBox={`0 0 ${x0 + W + 140} ${y0 + H + 64}`} class="dl-drawing" role="img" aria-label="Panel anatomy">
      {/* track bands */}
      <For each={trackX}>
        {(t, i) => (
          <g>
            <rect
              x={t.x}
              y={y0 + S.bar}
              width={t.w}
              height={H - S.bar}
              fill={ink("param")}
              fill-opacity="0.06"
            />
            <text x={t.x + t.w / 2} y={y0 + H + 30 + (i() % 2) * 12} text-anchor="middle" fill={ink("param-ink")} style={{ font: MONO }}>
              {t.name} {t.name === "value" ? "1fr" : t.w}
            </text>
          </g>
        )}
      </For>
      {/* the sheet and its tab bar */}
      <Box x={x0} y={y0} w={W} h={H} line="rule-strong" />
      <Box x={x0} y={y0} w={W} h={S.bar} plane="plane-ground" line="rule-strong" caption="studies · meshes" captionColor="ink-2" />
      {/* head: kind tag + title */}
      <rect x={x0 + S.inset} y={yHead} width={32} height={S.tag} fill={ink("ink")} />
      <text x={x0 + S.inset + 4} y={yHead + 11} fill={ink("surface-float")} style={{ font: MONO, "letter-spacing": "0.12em" }}>
        FEM
      </text>
      <text x={x0 + S.inset} y={yHead + S.tag + S.gap + 15} fill={ink("ink")} style={{ font: "var(--weight-semibold) 15px var(--font-mono)", "letter-spacing": "0.04em" }}>
        STUDIES
      </text>
      {/* rows on the tracks */}
      <For each={rows}>
        {(name, i) => {
          const y = yRows + i() * S.cell;
          return (
            <g>
              <line x1={x0 + S.inset} x2={x0 + W - S.inset} y1={y + S.cell} y2={y + S.cell} stroke={ink("rule")} stroke-width="1" />
              <text x={trackX[0].x} y={y + 15} fill={ink("ink-3")} style={{ font: MONO, "text-transform": "uppercase", "letter-spacing": "0.12em" }}>
                {name}
              </text>
              <text x={trackX[1].x + trackX[1].w} y={y + 16} text-anchor="end" fill={ink("ink")} style={{ font: "var(--weight-medium) 12px var(--font-mono)" }}>
                {["sink-mesh", "401.0", "12.5", "body"][i()]}
              </text>
              <text x={trackX[2].x} y={y + 16} fill={ink("ink-3")} style={{ font: "12px var(--font-mono)" }}>
                {["", "W/mK", "W", ""][i()]}
              </text>
              <rect x={trackX[3].x} y={y} width={S.action} height={S.cell} fill="none" stroke={ink("rule")} stroke-width="1" />
            </g>
          );
        }}
      </For>
      {/* actions, end-aligned in the value…action span */}
      <Box x={x0 + W - S.inset - 56} y={yActions} w={56} h={S.cell} plane="block-current" caption="solve" captionColor="ink-on-accent" />
      <Box x={x0 + W - S.inset - 56 - S.gap - 52} y={yActions} w={52} h={S.cell} line="rule-strong" caption="mesh" captionColor="ink-2" />
      {/* vertical rhythm */}
      <Dim x1={x0} y1={y0} x2={x0} y2={y0 + S.bar} text="bar 32" />
      <Dim x1={x0} y1={y0 + S.bar} x2={x0} y2={yHead} text="inset 8" />
      <Dim x1={x0} y1={yHead} x2={x0} y2={yHead + head} text="head 40 = 16+4+20" />
      <Dim x1={x0} y1={yHead + head} x2={x0} y2={yRows} text="section 16" />
      <Dim x1={x0} y1={yRows} x2={x0} y2={yRows + S.cell} text="row 24" />
      <Dim x1={x0} y1={yActions - S.section} x2={x0} y2={yActions} text="section 16" />
      <Dim x1={x0} y1={yActions} x2={x0} y2={yActions + S.cell} text="cell 24" />
      {/* the right edge: total */}
      <Dim x1={x0 + W} y1={y0} x2={x0 + W} y2={y0 + H} text={`${H} = 4 × ${H / 4}`} side={1} />
      <Dim x1={x0} y1={y0 + H} x2={x0 + S.inset} y2={y0 + H} text="8" side={1} />
    </svg>
  );
}

/** L5: the desk, its tracks and rows, at the default proportions. */
export function DeskDrawing() {
  const W = 660;
  const scale = 0.5;
  const editor = 460 * scale;
  const column = 320 * scale;
  const rowsH = [S.bar, S.toolbar];
  const deskH = 220;
  const tray = S.cell;
  const H = rowsH[0] + rowsH[1] + deskH + tray;
  const x0 = 70;
  const y0 = 20;
  const viewport = W - editor - column;
  return (
    <svg viewBox={`0 0 ${x0 + W + 40} ${y0 + H + 50}`} class="dl-drawing" role="img" aria-label="Desk">
      <Box x={x0} y={y0} w={W} h={rowsH[0]} plane="plane-ground" line="rule-strong" caption="menu bar · bar 32" />
      <Box x={x0} y={y0 + rowsH[0]} w={W} h={rowsH[1]} plane="plane-ground" line="rule-strong" caption="toolbar · 40" />
      <Box x={x0} y={y0 + rowsH[0] + rowsH[1]} w={editor} h={deskH} plane="plane-sheet" line="rule-strong" caption="scene.py" />
      <Box x={x0 + editor} y={y0 + rowsH[0] + rowsH[1]} w={viewport} h={deskH} plane="plane-ground" line="rule-heavy" caption="viewport" />
      <Box x={x0 + editor + viewport} y={y0 + rowsH[0] + rowsH[1]} w={column} h={deskH} plane="plane-sheet" line="rule-strong" caption="column" />
      <Box x={x0} y={y0 + H - tray} w={W} h={tray} plane="plane-ground" line="rule-strong" caption="tray · cell 24" />
      <Dim x1={x0} y1={y0 + H} x2={x0 + editor} y2={y0 + H} text="460 = 115u" side={1} />
      <Dim x1={x0 + editor} y1={y0 + H} x2={x0 + editor + viewport} y2={y0 + H} text="the rest" side={1} />
      <Dim x1={x0 + editor + viewport} y1={y0 + H} x2={x0 + W} y2={y0 + H} text="320 = 80u" side={1} />
      <Dim x1={x0} y1={y0} x2={x0} y2={y0 + rowsH[0]} text="32" />
      <Dim x1={x0} y1={y0 + rowsH[0]} x2={x0} y2={y0 + rowsH[0] + rowsH[1]} text="40" />
      <Dim x1={x0} y1={y0 + H - tray} x2={x0} y2={y0 + H} text="24" />
    </svg>
  );
}

/** L6: the viewport's anchor grid; the centre is the part's. */
export function AnchorsDrawing() {
  const W = 520;
  const H = 300;
  const x0 = 40;
  const y0 = 20;
  const m = S.section;
  const anchors: [string, number, number, "start" | "middle" | "end"][] = [
    ["tool rail ↓", x0 + m, y0 + m, "start"],
    ["readout →", x0 + m + 110, y0 + m, "start"],
    ["projection · render · cube ↓", x0 + W - m, y0 + m, "end"],
    ["legend ↑", x0 + m, y0 + H - m, "start"],
    ["hint ↑", x0 + W / 2, y0 + H - m, "middle"],
    ["performance ↑", x0 + W - m, y0 + H - m, "end"],
  ];
  return (
    <svg viewBox={`0 0 ${x0 + W + 40} ${y0 + H + 30}`} class="dl-drawing" role="img" aria-label="Viewport anchors">
      <Box x={x0} y={y0} w={W} h={H} plane="plane-ground" line="rule-heavy" />
      {/* the margin */}
      <Box x={x0 + m} y={y0 + m} w={W - 2 * m} h={H - 2 * m} line="rule" dashed />
      {/* thirds */}
      <For each={[1, 2]}>
        {(i) => (
          <g>
            <line x1={x0 + (W * i) / 3} x2={x0 + (W * i) / 3} y1={y0 + m} y2={y0 + H - m} stroke={ink("rule")} stroke-dasharray="2 3" />
            <line y1={y0 + (H * i) / 3} y2={y0 + (H * i) / 3} x1={x0 + m} x2={x0 + W - m} stroke={ink("rule")} stroke-dasharray="2 3" />
          </g>
        )}
      </For>
      <rect x={x0 + W / 3} y={y0 + H / 3} width={W / 3} height={H / 3} fill={ink("param")} fill-opacity="0.06" />
      <text x={x0 + W / 2} y={y0 + H / 2 + 3} text-anchor="middle" fill={ink("param-ink")} style={{ font: MONO, "text-transform": "uppercase", "letter-spacing": "0.12em" }}>
        the part
      </text>
      <For each={anchors}>
        {([name, x, y, anchor]) => (
          <g>
            <circle cx={x} cy={y} r="2" fill={ink("ink")} />
            <text
              x={anchor === "end" ? x - 6 : anchor === "middle" ? x : x + 6}
              y={y < y0 + H / 2 ? y + 14 : y - 8}
              text-anchor={anchor}
              fill={ink("ink-2")}
              style={{ font: MONO }}
            >
              {name}
            </text>
          </g>
        )}
      </For>
      <Dim x1={x0} y1={y0 + H} x2={x0 + m} y2={y0 + H} text="section 16" side={1} />
    </svg>
  );
}
