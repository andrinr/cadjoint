/**
 * The mesh inspection readout: counts, quality bands, and the element
 * histogram.
 *
 * Two places in the Meshes tab show it — under a declared SimMesh card, and
 * under the "generate this study's implicit mesh" fallback — and they differ
 * only in whether a histogram is available, so both render this. The numbers
 * come straight from /api/mesh_inspect; the histogram is binned client-side
 * from the per-vertex quality scalars.
 */

import { For, Show } from "solid-js";
import { formatScalar } from "../../simulation";
import { Stat, StatRow } from "../ui";
import type { Histogram } from "../../meshes";
import type { MeshInspectInfo, QualitySummary } from "../../types";

const HISTOGRAM_WIDTH = 216;
const HISTOGRAM_HEIGHT = 42;

/** min / mean / max, or an em dash where the metric was not computed. */
const quality = (summary: QualitySummary | undefined) =>
  summary
    ? `${formatScalar(summary.min)} / ${formatScalar(summary.mean)} / ${formatScalar(summary.max)}`
    : "–";

export interface MeshStatsProps {
  info: MeshInspectInfo;
  /** Element-quality bins; omitted where the panel shows no histogram. */
  histogram?: Histogram | null;
}

export function MeshStats(props: MeshStatsProps) {
  return (
    <div class="sim-inspect" data-testid="mesh-stats">
      <StatRow>
        <Stat label="nodes" value={props.info.nodes} />
        <Stat label="elements" value={props.info.elements} />
        <Show when={props.info.method}>
          <Stat label="method" value={props.info.method} />
        </Show>
      </StatRow>
      <StatRow>
        <Stat label="jacobian" value={quality(props.info.quality.scaled_jacobian)} />
      </StatRow>
      <StatRow>
        <Stat label="aspect" value={quality(props.info.quality.aspect_ratio)} />
      </StatRow>
      <Show when={props.histogram}>
        {(bins) => (
          <svg
            class="sim-histogram"
            viewBox={`0 0 ${HISTOGRAM_WIDTH} ${HISTOGRAM_HEIGHT}`}
            preserveAspectRatio="none"
            role="img"
            aria-label="Element quality histogram"
            data-testid="mesh-histogram"
          >
            <For each={bins().counts}>
              {(count, index) => {
                const width = HISTOGRAM_WIDTH / bins().counts.length;
                const height =
                  bins().peak > 0 ? (count / bins().peak) * (HISTOGRAM_HEIGHT - 2) : 0;
                return (
                  <rect
                    x={index() * width + 0.5}
                    y={HISTOGRAM_HEIGHT - height}
                    width={Math.max(width - 1, 0.5)}
                    height={height}
                  />
                );
              }}
            </For>
          </svg>
        )}
      </Show>
      <Show when={props.histogram}>
        {(bins) => (
          <div class="sim-legend-values">
            <span>{formatScalar(bins().min)}</span>
            <span>{formatScalar(bins().max)}</span>
          </div>
        )}
      </Show>
    </div>
  );
}
