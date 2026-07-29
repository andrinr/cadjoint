/**
 * Inline SVG icons for the tool rail and top bar.
 *
 * Inline rather than an icon font or sprite sheet: the playground ships under a
 * strict CSP with no external requests, and these are a handful of glyphs.
 * Every icon draws on a 24×24 grid and inherits `currentColor`.
 */

import type { JSX } from "solid-js";

type IconProps = { size?: number };

const base = (size: number): JSX.SvgSVGAttributes<SVGSVGElement> => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  "stroke-width": 1.6,
  "stroke-linecap": "round" as const,
  "stroke-linejoin": "round" as const,
});

export function CursorIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M5 3l6.5 16 2.2-6.3 6.3-2.2z" />
    </svg>
  );
}

export function PolygonIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M12 3l8 6-3 10H7L4 9z" />
      <circle cx="12" cy="3" r="1.6" fill="currentColor" />
      <circle cx="20" cy="9" r="1.6" fill="currentColor" />
      <circle cx="4" cy="9" r="1.6" fill="currentColor" />
    </svg>
  );
}

export function BoxIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M12 2.6l8 4.2v9L12 20l-8-4.2v-9z" />
      <path d="M4 6.8l8 4.2 8-4.2" />
      <path d="M12 11v9" />
    </svg>
  );
}

export function SphereIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <circle cx="12" cy="12" r="8.5" />
      <ellipse cx="12" cy="12" rx="8.5" ry="3.4" />
    </svg>
  );
}

export function CylinderIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <ellipse cx="12" cy="6" rx="7" ry="3" />
      <path d="M5 6v12M19 6v12" />
      <path d="M5 18c0 1.7 3.1 3 7 3s7-1.3 7-3" />
    </svg>
  );
}

export function MoveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M12 3v18M3 12h18" />
      <path d="M12 3l-2.4 2.4M12 3l2.4 2.4M12 21l-2.4-2.4M12 21l2.4-2.4" />
      <path d="M3 12l2.4-2.4M3 12l2.4 2.4M21 12l-2.4-2.4M21 12l-2.4 2.4" />
    </svg>
  );
}

export function RotateIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20 3.5V8h-4.5" />
    </svg>
  );
}

export function DisplayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M2.5 12s3.6-6 9.5-6 9.5 6 9.5 6-3.6 6-9.5 6-9.5-6-9.5-6z" />
      <circle cx="12" cy="12" r="2.6" />
    </svg>
  );
}

export function PlayIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M7 4.5l12 7.5-12 7.5z" />
    </svg>
  );
}

export function ResetIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M4 12a8 8 0 1 0 2.6-5.9" />
      <path d="M4 3.5V8h4.5" />
    </svg>
  );
}

export function CodeIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M9 7l-5 5 5 5M15 7l5 5-5 5" />
    </svg>
  );
}

export function TraceIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2.1 2.1M16.9 16.9L19 19M19 5l-2.1 2.1M7.1 16.9L5 19" />
    </svg>
  );
}

export function ObjectSelectIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M12 3.2l7.2 3.8v8L12 18.8 4.8 15V7z" />
      <path d="M4.8 7l7.2 3.8L19.2 7" />
    </svg>
  );
}

export function VertexSelectIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M5 18L9.5 6l9 5.5z" />
      <circle cx="5" cy="18" r="2.3" fill="currentColor" stroke="none" />
      <circle cx="9.5" cy="6" r="2.3" fill="currentColor" stroke="none" />
      <circle cx="18.5" cy="11.5" r="2.3" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 18)}>
      <path d="M4 6.5h16M9.5 6.5V4.5h5v2M6.5 6.5l1 13h9l1-13" />
      <path d="M10.5 10v6M13.5 10v6" />
    </svg>
  );
}

/** Converging lines: a perspective frustum. */
export function PerspectiveIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 16)}>
      <path d="M4 4.5l16 3.2v8.6L4 19.5z" />
      <path d="M4 4.5v15M20 7.7v8.6" />
    </svg>
  );
}

/** Parallel lines: an orthographic box. */
export function OrthographicIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 16)}>
      <rect x="4" y="5" width="16" height="14" rx="1" />
      <path d="M4 9h16M8 5v14" />
    </svg>
  );
}

/** A corner-on cube: the isometric home view. */
export function IsoIcon(props: IconProps) {
  return (
    <svg {...base(props.size ?? 16)}>
      <path d="M12 3l8 4.6v8.8L12 21l-8-4.6V7.6z" />
      <path d="M4 7.6l8 4.6 8-4.6M12 12.2V21" />
    </svg>
  );
}
