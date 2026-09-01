/**
 * The objective sparkline drawn by every optimization view.
 *
 * The optimize card draws it twice (live progress and finished history) and
 * the trajectory player draws it again with a cursor on top, so the svg box,
 * its aspect and its polyline live here once. Extra marks (the player's
 * cursor line and point) are passed as children and render inside the same
 * viewBox.
 */

import { type JSX } from "solid-js";
import { sparklinePoints } from "../../optimize";

export const SPARK_WIDTH = 220;
export const SPARK_HEIGHT = 44;

export interface SparklineProps {
  values: readonly number[];
  ariaLabel: string;
  testId?: string;
  /** Overlay marks drawn in the same viewBox, above the polyline. */
  children?: JSX.Element;
}

export function Sparkline(props: SparklineProps) {
  return (
    <div class="opt-spark">
      <svg
        viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={props.ariaLabel}
        data-testid={props.testId}
      >
        <polyline points={sparklinePoints(props.values, SPARK_WIDTH, SPARK_HEIGHT)} />
        {props.children}
      </svg>
    </div>
  );
}
