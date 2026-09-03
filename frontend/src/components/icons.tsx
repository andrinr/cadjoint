/**
 * Inline SVG icons for the tool rail and top bar.
 *
 * Inline rather than an icon font or sprite sheet: the playground ships under a
 * strict CSP with no external requests, and these are a handful of glyphs.
 *
 * ── The grid contract ────────────────────────────────────────────────────
 * Every icon draws on a 24×24 viewBox, inherits `currentColor`, and obeys
 * four rules so a rail of them reads as one set:
 *
 *   1. One stroke weight on screen. `strokeWidth` is scaled against the
 *      reference size, so a 16px icon and an 18px icon put the same number
 *      of device pixels on the hairline instead of the smaller one looking
 *      thinner than its neighbours.
 *   2. Round caps and joins, always.
 *   3. Solid nodes come from `<Dot>` / `<Handle>`, never a hand-written
 *      circle: a filled circle that also inherits the stroke renders a
 *      radius bigger than it claims, which is why the old dots varied in
 *      weight from icon to icon.
 *   4. One corner radius (`CORNER`) on rectangles.
 *
 * Artwork stays inside roughly 3 → 21 on both axes, so nothing sits optically
 * larger than the rest at the same nominal size.
 */

import type { JSX } from "solid-js";

type IconProps = { size?: number };

/** The size the stroke weight is tuned for; other sizes compensate. */
const REFERENCE_SIZE = 18;
/** Stroke weight in viewBox units at the reference size. */
const REFERENCE_STROKE = 1.6;
/** Radius of a point node. */
const DOT = 1.7;
/** Radius of a draggable handle — a point you can grab, so it reads bigger. */
const HANDLE = 2.3;
/** The one corner radius rectangles use. */
const CORNER = 1;

const base = (size: number): JSX.SvgSVGAttributes<SVGSVGElement> => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  // Hold the rendered hairline constant across icon sizes.
  "stroke-width": (REFERENCE_STROKE * REFERENCE_SIZE) / size,
  "stroke-linecap": "round" as const,
  "stroke-linejoin": "round" as const,
});

/** A solid point. `stroke="none"` keeps it at the radius it declares. */
const Dot = (props: { cx: number; cy: number }) => (
  <circle cx={props.cx} cy={props.cy} r={DOT} fill="currentColor" stroke="none" />
);

/** A solid grabbable handle, one step larger than a point. */
const Handle = (props: { cx: number; cy: number }) => (
  <circle cx={props.cx} cy={props.cy} r={HANDLE} fill="currentColor" stroke="none" />
);

export function CursorIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M5 3l6.5 16 2.2-6.3 6.3-2.2z" />
    </svg>
  );
}

export function PolygonIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 3l8 6-3 10H7L4 9z" />
      <Dot cx={12} cy={3} />
      <Dot cx={20} cy={9} />
      <Dot cx={4} cy={9} />
    </svg>
  );
}

/**
 * Sketch on a face: a plane seen at an angle, with a profile drawn on it.
 *
 * The face is the parallelogram — the same read as the box's top facet, so
 * the two icons agree about what "a flat face" looks like — and the open
 * polyline with a node at each end is the same mark PolygonIcon uses for a
 * sketch. The tool is the composition of those two, and so is its glyph.
 */
export function FaceSketchIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M3.5 13.5l8-4.8 9 3.2-8 4.8z" />
      <path d="M8.6 13.1l3.4-2 3.9 1.4" />
      <Dot cx={8.6} cy={13.1} />
      <Dot cx={15.9} cy={12.5} />
    </svg>
  );
}

export function PointAddIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M4 17L9 6l9 6" />
      <Dot cx={4} cy={17} />
      <Dot cx={9} cy={6} />
      <path d="M17.5 15v6M14.5 18h6" />
    </svg>
  );
}

export function BoxIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 2.6l8 4.2v9L12 20l-8-4.2v-9z" />
      <path d="M4 6.8l8 4.2 8-4.2" />
      <path d="M12 11v9" />
    </svg>
  );
}

export function SphereIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <circle cx="12" cy="12" r="8.5" />
      <ellipse cx="12" cy="12" rx="8.5" ry="3.4" />
    </svg>
  );
}

export function CylinderIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <ellipse cx="12" cy="6" rx="7" ry="3" />
      <path d="M5 6v12M19 6v12" />
      <path d="M5 18c0 1.7 3.1 3 7 3s7-1.3 7-3" />
    </svg>
  );
}

export function MoveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 3v18M3 12h18" />
      <path d="M12 3l-2.4 2.4M12 3l2.4 2.4M12 21l-2.4-2.4M12 21l2.4-2.4" />
      <path d="M3 12l2.4-2.4M3 12l2.4 2.4M21 12l-2.4-2.4M21 12l-2.4 2.4" />
    </svg>
  );
}

export function RotateIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20 3.5V8h-4.5" />
    </svg>
  );
}

export function ScaleIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M5 19L19 5M10 5h9v9" />
      <rect x="3.5" y="15.5" width="5" height="5" rx={CORNER} />
      <rect x="15.5" y="3.5" width="5" height="5" rx={CORNER} />
    </svg>
  );
}

export function DisplayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M2.5 12s3.6-6 9.5-6 9.5 6 9.5 6-3.6 6-9.5 6-9.5-6-9.5-6z" />
      <circle cx="12" cy="12" r="2.6" />
    </svg>
  );
}

export function PlayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M7 4.5l12 7.5-12 7.5z" />
    </svg>
  );
}

export function ResetIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M4 12a8 8 0 1 0 2.6-5.9" />
      <path d="M4 3.5V8h4.5" />
    </svg>
  );
}

export function CodeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M9 7l-5 5 5 5M15 7l5 5-5 5" />
    </svg>
  );
}

export function TraceIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2.1 2.1M16.9 16.9L19 19M19 5l-2.1 2.1M7.1 16.9L5 19" />
    </svg>
  );
}

export function ObjectSelectIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 3.2l7.2 3.8v8L12 18.8 4.8 15V7z" />
      <path d="M4.8 7l7.2 3.8L19.2 7" />
    </svg>
  );
}

export function VertexSelectIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M5 18L9.5 6l9 5.5z" />
      <Handle cx={5} cy={18} />
      <Handle cx={9.5} cy={6} />
      <Handle cx={18.5} cy={11.5} />
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M4 6.5h16M9.5 6.5V4.5h5v2M6.5 6.5l1 13h9l1-13" />
      <path d="M10.5 10v6M13.5 10v6" />
    </svg>
  );
}

/** Converging lines: a perspective frustum. */
export function PerspectiveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M4 4.5l16 3.2v8.6L4 19.5z" />
      <path d="M4 4.5v15M20 7.7v8.6" />
    </svg>
  );
}

/** Parallel lines: an orthographic box. */
export function OrthographicIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <rect x="4" y="5" width="16" height="14" rx={CORNER} />
      <path d="M4 9h16M8 5v14" />
    </svg>
  );
}

/** A corner-on cube: the isometric home view. */
export function IsoIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 3l8 4.6v8.8L12 21l-8-4.6V7.6z" />
      <path d="M4 7.6l8 4.6 8-4.6M12 12.2V21" />
    </svg>
  );
}

/** Solid cube: the model editing mode. */
export function ModelModeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 3l7.5 4.3v8.4L12 20l-7.5-4.3V7.3z" />
      <path d="M4.5 7.3l7.5 4.3 7.5-4.3M12 11.6V20" />
    </svg>
  );
}

/** Pencil over a corner: the sketch editing mode. */
export function SketchModeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M4 20V9l6-5" />
      <path d="M13.2 15.6l6.4-6.4 2.1 2.1-6.4 6.4-2.9.8z" />
      <Dot cx={4} cy={20} />
      <Dot cx={4} cy={9} />
      <Dot cx={10} cy={4} />
    </svg>
  );
}

/** Field lines through a part: the simulate editing mode. */
export function SimulateModeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M3 8c3-3.4 6-3.4 9 0s6 3.4 9 0" />
      <path d="M3 13c3-3.4 6-3.4 9 0s6 3.4 9 0" />
      <path d="M3 18c3-3.4 6-3.4 9 0s6 3.4 9 0" />
    </svg>
  );
}

/** Profile pushed upward: extrude. */
export function ExtrudeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M5 17.5l7 3.5 7-3.5-7-3.5z" />
      <path d="M12 12V4M9 7l3-3 3 3" />
    </svg>
  );
}

/** Profile swung around an axis: revolve. */
export function RevolveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 3v18" stroke-dasharray="2.4 2.4" />
      <path d="M14.5 6.5h4v5h-4z" />
      <path d="M5 16.5c0 1.9 3.1 3.5 7 3.5s7-1.6 7-3.5" />
      <path d="M17.6 18.9l1.4 1.1-1.8.6" />
    </svg>
  );
}

/** Two sections joined by walls: loft. */
export function LoftIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <ellipse cx="12" cy="5.5" rx="6" ry="2.4" />
      <ellipse cx="12" cy="18.5" rx="8" ry="2.8" />
      <path d="M6 5.8L4 18.2M18 5.8l2 12.4" />
    </svg>
  );
}

/** Two points with a dimension between them: distance constraint. */
export function DistanceIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <Dot cx={4.5} cy={12} />
      <Dot cx={19.5} cy={12} />
      <path d="M7 12h10M9 9.5L7 12l2 2.5M15 9.5l2 2.5-2 2.5" />
    </svg>
  );
}

/** Level points: horizontal constraint. */
export function HorizontalIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M4 12h16" />
      <Dot cx={7} cy={12} />
      <Dot cx={17} cy={12} />
    </svg>
  );
}

/** Stacked points: vertical constraint. */
export function VerticalIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M12 4v16" />
      <Dot cx={12} cy={7} />
      <Dot cx={12} cy={17} />
    </svg>
  );
}

/** Two points collapsing into one: coincident constraint. */
export function CoincidentIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <Dot cx={12} cy={12} />
      <circle cx="12" cy="12" r="5.5" />
      <path d="M3.5 12h3M17.5 12h3" />
    </svg>
  );
}

/** Two edges that never meet: parallel constraint. */
export function ParallelIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M8 4L5 20M19 4l-3 16" />
    </svg>
  );
}

/** Edges at a right angle: perpendicular constraint. */
export function PerpendicularIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? REFERENCE_SIZE)}>
      <path d="M6 4v16M6 20h14" />
      <path d="M6 14h6v6" />
    </svg>
  );
}
