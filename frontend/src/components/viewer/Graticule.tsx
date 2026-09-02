/**
 * The graticule's text: the gain readout and the title block.
 *
 * The ruled faceplate itself is drawn on the GPU (`viewer/graticule.wgsl`), so
 * that it can sit *under* the geometry. Its two labels cannot — they are type,
 * and type belongs to the browser — so they live here, over the canvas, and
 * read the same camera the shader does.
 *
 * Top-left, never moving: what one division is worth and where the camera is
 * standing. Bottom-right, ASME Y14.100's corner: the title block, five rows of
 * what is on screen, with an em dash wherever the app cannot fill a field. Both
 * appear and disappear with the grid — one instrument, one switch.
 */

import { Show, createMemo } from "solid-js";
import { dirty, sceneName, simView, studies } from "../../state";
import { formatGain, gainOf, octant, viewLabel } from "../../viewer/graticule";
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

  const mesh = createMemo(() => {
    const info = simView()?.info;
    if (!info) return null;
    return info.method ? `${info.name} · ${info.method}` : info.name;
  });

  return (
    <Show when={props.show}>
      <div
        class="graticule-gain"
        data-testid="graticule-gain"
        title="Floor grid spacing, on a 1-2-5 ladder. Scroll to zoom freely; hold Alt to zoom in detents. A > prefix means an on-screen measurement is not to scale."
      >
        <span>GRID</span>
        <b>{gain().text}</b>
        <i>{gain().unit}</i>
        <span>VIEW</span>
        <b>{viewLabel(props.camera.yaw, props.camera.pitch)}</b>
        <i>· {octant(props.camera.yaw, props.camera.pitch)}</i>
      </div>
      <dl class="title-block" data-testid="title-block">
        <Row label="SCENE" value={sceneName() ?? "untitled"} />
        <Row label="STUDY" value={study()?.name ?? null} />
        <Row label="MESH" value={mesh()} />
        {/* The program names the solver: a ThermalStudy is solved by the
            thermal path, an ElasticStudy by the elastic one. The frontend is
            not told which backend ran it, so it does not claim one. */}
        <Row label="SOLVER" value={study()?.kind ?? null} />
        <Row label="REV" value={sceneName() ? (dirty() ? "modified" : "saved") : "unsaved"} />
      </dl>
    </Show>
  );
}
