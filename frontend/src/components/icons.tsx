/**
 * Inline SVG icons for the tool rail and top bar.
 *
 * Inline rather than an icon font or sprite sheet: the playground ships under a
 * strict CSP with no external requests, and these are a handful of glyphs.
 *
 * ── The grid contract (design language I1) ───────────────────────────────
 * Every icon draws on a 16×16 viewBox, inherits `currentColor`, and obeys
 * five rules so a rail of them reads as one set, drawn by the same pen as
 * the hairlines around it:
 *
 *   1. One stroke weight on screen: 1.25 device px. `stroke-width` is scaled
 *      against the 16px box, so an icon rendered at another size still puts
 *      1.25px on the line instead of growing or thinning with the box.
 *   2. No fills. A point is a small hollow circle (`<Pt>`), a grabbable
 *      handle a small hollow square (`<Grip>`); the lines that meet them stop
 *      at their rim (`link`) rather than running through the hole.
 *   3. Square caps and miter joins, always; no rounded rectangles.
 *   4. Drawn on the half pixel: every horizontal or vertical stroke sits on
 *      an x.5 coordinate, so at 16px it covers whole device pixels.
 *   5. Artwork stays inside roughly 1.5 → 14.5 on both axes.
 */

import type { JSX } from "solid-js";

type IconProps = { size?: number };

/** The box the artwork is drawn in, and the default rendered size. */
const BOX = 16;
/** Stroke weight in device px, at any rendered size. */
const STROKE = 1.25;
/** Radius of a point node. */
const POINT = 1.5;
/** Half side of a handle square — a point you can grab. */
const GRIP = 1.5;

const base = (size: number): JSX.SvgSVGAttributes<SVGSVGElement> => ({
  width: size,
  height: size,
  viewBox: `0 0 ${BOX} ${BOX}`,
  fill: "none",
  stroke: "currentColor",
  // Hold the rendered line at STROKE device px across icon sizes.
  "stroke-width": (STROKE * BOX) / size,
  "stroke-linecap": "square" as const,
  "stroke-linejoin": "miter" as const,
});

/** A hollow point. */
const Pt = (props: { x: number; y: number }) => <circle cx={props.x} cy={props.y} r={POINT} />;

/** A hollow grabbable handle, centred on whole units so its sides land on x.5. */
const Grip = (props: { x: number; y: number }) => (
  <rect x={props.x - GRIP} y={props.y - GRIP} width={GRIP * 2} height={GRIP * 2} />
);

type Node = "point" | "grip" | null;
type P = readonly [number, number];

/**
 * How far to pull a line end back from a node's centre so its square cap
 * stops on the node's stroke: to the rim along the line's direction, plus
 * half a stroke for the cap.
 */
function pullback(node: Node, ux: number, uy: number): number {
  if (node === "point") return POINT + STROKE / 2;
  if (node === "grip") return GRIP / Math.max(Math.abs(ux), Math.abs(uy)) + STROKE / 2;
  return 0;
}

const n = (v: number) => +v.toFixed(3);

/** A polyline through `pts` whose ends stop short of the nodes they meet. */
function link(pts: readonly P[], start: Node = null, end: Node = null): string {
  const out = pts.map((p) => [p[0], p[1]] as [number, number]);
  const trim = (i: number, j: number, node: Node) => {
    const dx = out[j][0] - out[i][0];
    const dy = out[j][1] - out[i][1];
    const len = Math.hypot(dx, dy);
    const d = pullback(node, dx / len, dy / len);
    out[i] = [out[i][0] + (dx / len) * d, out[i][1] + (dy / len) * d];
  };
  trim(0, 1, start);
  trim(out.length - 1, out.length - 2, end);
  return "M" + out.map((p) => `${n(p[0])} ${n(p[1])}`).join("L");
}

/** A closed polygon whose every vertex is a node: one trimmed edge per side. */
function ring(pts: readonly P[], node: Node): string {
  return pts.map((p, i) => link([p, pts[(i + 1) % pts.length]], node, node)).join("");
}

/** The isometric cube shared by the box, iso-view and model-mode icons. */
const CUBE = "M8.5 1.5L14.5 5V12L8.5 15.5L2.5 12V5Z" + "M2.5 5L8.5 8.5L14.5 5M8.5 8.5V15.5";

export function CursorIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M3.5 2.5L8 13.5L9.5 9L14 7.5Z" />
    </svg>
  );
}

const POLYGON: readonly P[] = [
  [8, 2.5],
  [13.5, 6.5],
  [11, 13.5],
  [5, 13.5],
  [2.5, 6.5],
];

export function PolygonIcon(props: IconProps) {
  const [top, right, br, bl, left] = POLYGON;
  return (
    <svg {...base(props.size ?? BOX)}>
      <path
        d={
          link([left, top], "point", "point") +
          link([top, right], "point", "point") +
          link([right, br, bl, left], "point", "point")
        }
      />
      <Pt x={top[0]} y={top[1]} />
      <Pt x={right[0]} y={right[1]} />
      <Pt x={left[0]} y={left[1]} />
    </svg>
  );
}

/**
 * Sketch on a face: a plane seen at an angle, with a profile drawn on it.
 *
 * The face is the trapezoid — a flat face seen in perspective — and the open
 * polyline with a node at each end is the same mark PolygonIcon uses for a
 * sketch. The tool is the composition of those two, and so is its glyph.
 */
export function FaceSketchIcon(props: IconProps) {
  const a: P = [5.5, 10];
  const b: P = [10.5, 10];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M1.5 13.5L3.5 3.5H12.5L14.5 13.5Z" />
      <path d={link([a, [8, 6], b], "point", "point")} />
      <Pt x={a[0]} y={a[1]} />
      <Pt x={b[0]} y={b[1]} />
    </svg>
  );
}

export function PointAddIcon(props: IconProps) {
  const a: P = [3, 12.5];
  const b: P = [6, 3.5];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={link([a, b], "point", "point") + link([b, [12.5, 7.5]], "point")} />
      <Pt x={a[0]} y={a[1]} />
      <Pt x={b[0]} y={b[1]} />
      <path d="M11.5 10V15M9 12.5H14" />
    </svg>
  );
}

export function BoxIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={CUBE} />
    </svg>
  );
}

export function SphereIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <circle cx="8.5" cy="8.5" r="6.5" />
      <ellipse cx="8.5" cy="8.5" rx="6.5" ry="2.5" />
    </svg>
  );
}

export function CylinderIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M2.5 4.5A6 2 0 0 1 14.5 4.5V12.5A6 2 0 0 1 2.5 12.5Z" />
      <path d="M2.5 4.5A6 2 0 0 0 14.5 4.5" />
    </svg>
  );
}

export function MoveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M8.5 3V14M3 8.5H14" />
      <path d="M6.5 4L8.5 2L10.5 4M6.5 13L8.5 15L10.5 13" />
      <path d="M4 6.5L2 8.5L4 10.5M13 6.5L15 8.5L13 10.5" />
    </svg>
  );
}

export function RotateIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M14 8.5A5.5 5.5 0 1 1 12.4 4.6" />
      <path d="M13.5 2V5.5H10" />
    </svg>
  );
}

export function ScaleIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <rect x="1.5" y="9.5" width="5" height="5" />
      <path d="M8.5 7.5L13.5 2.5M9.5 2.5H13.5V6.5" />
    </svg>
  );
}

export function DisplayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M1.5 8.5C3.5 5.2 5.8 3.5 8.5 3.5S13.5 5.2 15.5 8.5C13.5 11.8 11.2 13.5 8.5 13.5S3.5 11.8 1.5 8.5Z" />
      <circle cx="8.5" cy="8.5" r="2.25" />
    </svg>
  );
}

export function PlayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M4.5 2.5L13.5 8L4.5 13.5Z" />
    </svg>
  );
}

export function ResetIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M3 8.5A5.5 5.5 0 1 0 4.6 4.6" />
      <path d="M3.5 2V5.5H7" />
    </svg>
  );
}

export function CodeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M5.5 4.5L1.5 8.5L5.5 12.5M10.5 4.5L14.5 8.5L10.5 12.5" />
    </svg>
  );
}

export function ObjectSelectIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M8.5 2.5L13.5 5.5V11.5L8.5 14.5L3.5 11.5V5.5Z" />
      <path d="M3.5 5.5L8.5 8.5L13.5 5.5" />
    </svg>
  );
}

export function VertexSelectIcon(props: IconProps) {
  const pts: readonly P[] = [
    [3, 13],
    [6, 3],
    [13, 9],
  ];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={ring(pts, "grip")} />
      {pts.map((p) => (
        <Grip x={p[0]} y={p[1]} />
      ))}
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M2.5 4.5H13.5M6.5 4.5V2.5H9.5V4.5M3.5 4.5L4.5 14.5H11.5L12.5 4.5" />
      <path d="M6.5 7.5V11.5M9.5 7.5V11.5" />
    </svg>
  );
}

/** A profile through its held vertices: the construction skeleton. */
export function SkeletonIcon(props: IconProps) {
  const a: P = [3.5, 13.5];
  const b: P = [3.5, 3.5];
  const c: P = [13.5, 3.5];
  const d: P = [13.5, 9.5];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={link([a, b], "grip", "grip") + link([b, c], "grip", "grip") + link([c, d], "grip", "grip")} />
      <Grip x={a[0]} y={a[1]} />
      <Grip x={b[0]} y={b[1]} />
      <Grip x={c[0]} y={c[1]} />
      <Grip x={d[0]} y={d[1]} />
    </svg>
  );
}

/** Converging lines: a perspective frustum. */
export function PerspectiveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M2.5 2.5L13.5 5.5V11.5L2.5 14.5Z" />
      <path d="M2.5 8.5H13.5" />
    </svg>
  );
}

/** Parallel lines: an orthographic box. */
export function OrthographicIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M2.5 3.5H13.5V13.5H2.5Z" />
      <path d="M2.5 8.5H13.5" />
    </svg>
  );
}

/** Solid cube: the model editing mode. */
export function ModelModeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={CUBE} />
    </svg>
  );
}

/** Pencil over a corner: the sketch editing mode. */
export function SketchModeIcon(props: IconProps) {
  const a: P = [2.5, 13.5];
  const b: P = [2.5, 7.5];
  const c: P = [7.5, 2.5];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={link([a, b], "point", "point") + link([b, c], "point", "point")} />
      <path d="M7.5 14.5L8.5 11.5L13 7L15 9L10.5 13.5Z" />
      <Pt x={a[0]} y={a[1]} />
      <Pt x={b[0]} y={b[1]} />
      <Pt x={c[0]} y={c[1]} />
    </svg>
  );
}

/** Field lines through a part: the simulate editing mode. */
export function SimulateModeIcon(props: IconProps) {
  const wave = (y: number) => `M1.5 ${y}C3.5 ${y - 1.6} 5.5 ${y - 1.6} 8 ${y}S12.5 ${y + 1.6} 14.5 ${y}`;
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={wave(4) + wave(8.5) + wave(13)} />
    </svg>
  );
}

/** Profile pushed upward: extrude. */
export function ExtrudeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M2.5 11.5L8.5 14.5L14.5 11.5L8.5 8.5Z" />
      <path d="M8.5 6.5V3M6 4L8.5 1.5L11 4" />
    </svg>
  );
}

/** Profile swung around an axis: revolve. */
export function RevolveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M8.5 1.5V15" stroke-dasharray="1 2.5" />
      <path d="M10.5 2.5H13.5V6.5H10.5Z" />
      <path d="M3.5 11A5 2.5 0 0 0 13.5 11V10.5" />
      <path d="M11.5 11.5L13.5 9.5L15.5 11.5" />
    </svg>
  );
}

/** Two sections joined by walls: loft. */
export function LoftIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <ellipse cx="8.5" cy="3.5" rx="4" ry="1.5" />
      <ellipse cx="8.5" cy="12.5" rx="6" ry="2" />
      <path d="M4.5 3.5L2.5 12.5M12.5 3.5L14.5 12.5" />
    </svg>
  );
}

/** Two points with a dimension between them: distance constraint. */
export function DistanceIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <Pt x={2.5} y={11.5} />
      <Pt x={13.5} y={11.5} />
      <path d="M2.5 2.5V6.5M13.5 2.5V6.5M2.5 4.5H13.5" />
    </svg>
  );
}

/** Level points: horizontal constraint. */
export function HorizontalIcon(props: IconProps) {
  const a: P = [3, 8.5];
  const b: P = [13, 8.5];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={link([a, b], "point", "point")} />
      <Pt x={a[0]} y={a[1]} />
      <Pt x={b[0]} y={b[1]} />
    </svg>
  );
}

/** Stacked points: vertical constraint. */
export function VerticalIcon(props: IconProps) {
  const a: P = [8.5, 3];
  const b: P = [8.5, 13];
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={link([a, b], "point", "point")} />
      <Pt x={a[0]} y={a[1]} />
      <Pt x={b[0]} y={b[1]} />
    </svg>
  );
}

/** Two points collapsing into one: coincident constraint. */
export function CoincidentIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <Pt x={8.5} y={8.5} />
      <circle cx="8.5" cy="8.5" r="4.5" />
      <path d="M1.5 8.5H2.5M14.5 8.5H15.5" />
    </svg>
  );
}

/** Two edges that never meet: parallel constraint. */
export function ParallelIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M6.5 2.5L3.5 13.5M12.5 2.5L9.5 13.5" />
    </svg>
  );
}

/** Edges at a right angle: perpendicular constraint. */
export function PerpendicularIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d="M3.5 2.5V13.5H14.5" />
      <path d="M3.5 9.5H7.5V13.5" />
    </svg>
  );
}

/** The cube's three hidden edges, meeting at the far corner behind the near one. */
const CUBE_HIDDEN = "M8.5 8.5L2.5 12M8.5 8.5L14.5 12M8.5 8.5V1.5";

/** X-Ray: the solid's outline with its hidden edges shown through, dashed. */
export function XrayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={CUBE} />
      <path d={CUBE_HIDDEN} stroke-dasharray="1.5 1.5" stroke-linecap="butt" />
    </svg>
  );
}

/** Studio: the lit solid, its shadow side hatched. */
export function StudioIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={CUBE} />
      <path d="M10.5 8.6V13.6M12.5 7.4V12.4" />
    </svg>
  );
}

/** Wire: every edge, hidden or not, drawn alike. */
export function WireIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <path d={CUBE + CUBE_HIDDEN} />
    </svg>
  );
}

/** A point pinned to ground: the Fix (anchor) constraint. */
export function FixIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? BOX)}>
      <circle cx="8.5" cy="4.5" r="1.5" />
      <path d="M8.5 6.625V10.5" />
      <path d="M3.5 10.5H13.5" />
      <path d="M4.5 13.5L6.5 11.5M7.5 13.5L9.5 11.5M10.5 13.5L12.5 11.5" />
    </svg>
  );
}
