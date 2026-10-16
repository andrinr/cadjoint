/**
 * Fix: a point pinned to ground — the anchor constraint's rail glyph.
 *
 * Drawn to the icon set's grid contract (icons.tsx, design language I1): a
 * 16px box, a 1.25px stroke held across sizes, no fills, square caps, miter
 * joins, horizontals on the half pixel. Kept in its own file because the
 * shared set is owned elsewhere; it can move into icons.tsx unchanged.
 */

const BOX = 16;
const STROKE = 1.25;

export function FixIcon(props: { size?: number }) {
  const size = props.size ?? BOX;
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${BOX} ${BOX}`}
      fill="none"
      stroke="currentColor"
      stroke-width={(STROKE * BOX) / size}
      stroke-linecap="square"
      stroke-linejoin="miter"
    >
      {/* The point, its stem, and the ground it is pinned to. */}
      <circle cx="8.5" cy="4.5" r="1.5" />
      <path d="M8.5 6.625V10.5" />
      <path d="M3.5 10.5H13.5" />
      <path d="M4.5 13.5L6.5 11.5M7.5 13.5L9.5 11.5M10.5 13.5L12.5 11.5" />
    </svg>
  );
}
