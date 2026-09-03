/**
 * The graticule's text: the gain readout and the title block.
 *
 * The ruled faceplate itself is drawn on the GPU (`viewer/graticule.wgsl`), so
 * that it can sit *under* the geometry. Its two labels cannot — they are type,
 * and type belongs to the browser — so they live here, over the canvas, and
 * read the same camera the shader does.
 *
 * Top-left, never moving: what one division is worth and where the camera is
 * standing, and — while a distance-field slice is up — the two contour
 * intervals that slice is ruled at. Bottom-right, ASME Y14.100's corner: the
 * title block, five rows of what is on screen, with an em dash wherever the app
 * cannot fill a field. The gain and the title block appear and disappear with
 * the grid — one instrument, one switch; the contour row follows the slice,
 * because that is what it describes.
 */

import { Show, createMemo } from "solid-js";
import { dirty, sceneName, simView, studies } from "../../state";
import {
  GRID_MAJOR_EVERY,
  MM_PER_UNIT,
  formatDistance,
  formatGain,
  gainOf,
  gridPlane,
  octant,
  viewLabel,
} from "../../viewer/graticule";
import { orthoHeightFor } from "../../viewer/math";
import { isSliceView, slicePosition, type SdfView } from "../../viewer/display";
import { sdfRampCss } from "../../viewer/sdfRamp";
import type { Projection } from "../../viewer/math";

export interface GraticuleCamera {
  distance: number;
  yaw: number;
  pitch: number;
  projection: Projection;
}

export interface GraticuleProps {
  show: boolean;
  camera: GraticuleCamera;
  /** Which distance-field view is on; `solid` shows no second row. */
  sdfView: SdfView;
  /** The cutting plane, for the SLICE field. */
  sdfAxis: 0 | 1 | 2;
  sdfFraction: number;
}

/** One title-block row; an absent value is stated, never omitted. */
function Row(props: { label: string; value: string | null }) {
  return (
    <>
      <dt>{props.label}</dt>
      <dd>{props.value ?? "—"}</dd>
    </>
  );
}

export function Graticule(props: GraticuleProps) {
  const gain = createMemo(() => {
    const { mm, calibrated } = gainOf(props.camera.distance, props.camera.projection);
    return formatGain(mm, calibrated);
  });

  /** The study whose result the viewport is showing, if any. */
  const study = createMemo(() => {
    const name = simView()?.studyName;
    if (!name) return null;
    return studies().find((entry) => entry.name === name) ?? null;
  });

  /**
   * The contour intervals the slice is ruled at.
   *
   * Stated, not implied. The contours are drawn at the floor grid's own rung
   * and at a fifth of it — the same subdivision the grid uses — so both are
   * exact numbers rather than "some lines", and both come off the same 1-2-5
   * ladder the GRID field beside this one is already printing. The ramp
   * saturates at two major intervals, which is the third number here.
   */
  const contours = createMemo(() => {
    const { mm } = gainOf(props.camera.distance, props.camera.projection);
    return {
      major: formatGain(mm, true),
      minor: formatGain(mm / GRID_MAJOR_EVERY, true),
      full: formatGain(mm * 2, true),
    };
  });

  /**
   * The depth ramp's two ends, in world units.
   *
   * Half a framed height either side of the orbit target — the same range
   * `depth_tone` in `_webgpu.py` maps, restated here rather than guessed, so
   * the printed numbers are the ones the pixels were made from.
   */
  const depthRange = createMemo(() => {
    const frame = orthoHeightFor(props.camera.distance);
    const near = Math.max(props.camera.distance - frame / 2, 0.001);
    const far = props.camera.distance + frame / 2;
    const label = (units: number) => formatGain(units * MM_PER_UNIT, true);
    return { near: label(near), far: label(far) };
  });

  /** `SLICE Z · 120 mm` — which plane, and where it is standing. */
  const slice = createMemo(() => ({
    axis: ["X", "Y", "Z"][props.sdfAxis],
    at: formatDistance(slicePosition(props.sdfFraction) * MM_PER_UNIT),
  }));

  const mesh = createMemo(() => {
    const info = simView()?.info;
    if (!info) return null;
    return info.method ? `${info.name} · ${info.method}` : info.name;
  });

  return (
    <>
      <Show when={props.show}>
        <div
          class="graticule-gain"
          classList={{ plated: props.sdfView !== "solid" }}
          data-testid="graticule-gain"
          title="Construction grid spacing, on a 1-2-5 ladder, and the world plane it is ruled on: the floor XY wherever the floor is worth drawing, the wall the camera faces in the shallow views where it is not. Scroll to zoom freely; hold Alt to zoom in detents. A > prefix means an on-screen measurement is not to scale."
        >
          <span>GRID</span>
          <b>{gain().text}</b>
          {/* The plane qualifies the spacing the way the octant qualifies the
              view: a number of millimetres between lines says nothing until
              you know which plane the lines are on, and the grid no longer
              always answers "the floor". */}
          <i>{gain().unit} · {gridPlane(props.camera.yaw, props.camera.pitch)}</i>
          <span>VIEW</span>
          <b>{viewLabel(props.camera.yaw, props.camera.pitch)}</b>
          <i>· {octant(props.camera.yaw, props.camera.pitch)}</i>
        </div>
      </Show>
      {/* Row two: what the distance-field view is showing, in the same
          instrument face as the grid gain above it. It follows the view
          rather than the grid switch, because it describes the view. */}
      <Show when={isSliceView(props.sdfView)}>
        <div
          class="graticule-gain graticule-contour plated"
          data-testid="sdf-contour-readout"
          title="The cutting plane, and the contour intervals on it: major lines at the floor grid's spacing, minor lines at a fifth of it, tapering out two major intervals either side of the surface. A tier stops being drawn once one interval is thinner than a few pixels."
        >
          <span>SLICE</span>
          <b>{slice().axis}</b>
          <i>· {slice().at.value} {slice().at.unit}</i>
          <span>MAJOR</span>
          <b>{contours().major.text}</b>
          <i>{contours().major.unit}</i>
          <span>MINOR</span>
          <b>{contours().minor.text}</b>
          <i>{contours().minor.unit}</i>
        </div>
        <div class="sdf-key" data-testid="sdf-key">
          {/* The ramp comes from `sdfRamp.ts` rather than from a gradient
              typed into the stylesheet: the legend and the shader have to be
              the same ramp, and there is only one of it. */}
          <div class="sdf-key-ramp" style={{ background: sdfRampCss() }} />
          <div class="sdf-key-values">
            <span>
              {props.sdfView === "gradient"
                ? "0.5"
                : `−${contours().full.text} ${contours().full.unit}`}
            </span>
            <span>{props.sdfView === "gradient" ? "|∇f| 1.0" : "f = 0"}</span>
            <span>
              {props.sdfView === "gradient"
                ? "1.5"
                : `+${contours().full.text} ${contours().full.unit}`}
            </span>
          </div>
        </div>
      </Show>
      <Show when={props.sdfView === "depth"}>
        <div
          class="graticule-gain graticule-contour plated"
          data-testid="sdf-depth-readout"
          title="Linear depth along the primary ray, across the framed volume — half a frame either side of the orbit target, so the part fills the ramp at any zoom."
        >
          <span>DEPTH</span>
          <b>{depthRange().near.text}</b>
          <i>{depthRange().near.unit} near</i>
          <b>{depthRange().far.text}</b>
          <i>{depthRange().far.unit} far</i>
        </div>
      </Show>
      <Show when={props.sdfView === "normal"}>
        <div
          class="graticule-gain graticule-contour plated"
          data-testid="sdf-normal-readout"
          title="World-space surface normals, encoded n × 0.5 + 0.5."
        >
          <span>NORMAL</span>
          <b>n × 0.5 + 0.5</b>
          <i>· +X red · +Y green · +Z blue</i>
        </div>
      </Show>
      <Show when={props.show}>
        <dl class="title-block" data-testid="title-block">
          <Row label="SCENE" value={sceneName() ?? "untitled"} />
          <Row label="STUDY" value={study()?.name ?? null} />
          <Row label="MESH" value={mesh()} />
          {/* The program names the solver: a ThermalStudy is solved by the
              thermal path, an ElasticStudy by the elastic one. The frontend is
              not told which backend ran it, so it does not claim one. */}
          <Row label="SOLVER" value={study()?.kind ?? null} />
          <Row
            label="REV"
            value={sceneName() ? (dirty() ? "modified" : "saved") : "unsaved"}
          />
        </dl>
      </Show>
    </>
  );
}
