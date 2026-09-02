/**
 * The navigation cube's geometry: a chamfered cube, and how to project it.
 *
 * ── Why a chamfered cube ─────────────────────────────────────────────────
 * A plain cube has six faces and therefore six click targets, which is six of
 * the twenty-six standard views. The other twenty are edges and corners, and
 * on a plain cube they are infinitely thin — there is nothing to press.
 * Chamfering the cube turns each of them into a facet with real area: twelve
 * bevels for the edge views (45° about one axis) and eight corner triangles
 * for the isometric octants. The widget then *is* the rose. That is what
 * FreeCAD's navigation cube does, and it is the reason it does it.
 *
 * ── Why it is not CSS 3D ─────────────────────────────────────────────────
 * The previous version built the cube from six `transform: rotateX(…)` planes.
 * That works for six squares and stops working here: twenty-six facets in
 * three orientations, each needing its own rotation *and* its own polygon
 * outline, with the browser's own perspective quietly shearing all of it. The
 * geometry is projected here instead, orthographically — the projection the
 * viewport itself defaults to, so a corner standpoint draws the regular
 * hexagon that says "isometric" rather than a foreshortened approximation of
 * one — and drawn as SVG polygons. Hit testing comes free and is exact: a
 * corner facet is clickable exactly where the corner is drawn.
 *
 * Everything here is arithmetic over a camera, with no DOM in it, which is
 * what lets `test/viewRose.test.ts` state that every facet faces the camera it
 * claims to for all twenty-six standard views and a sweep of free ones.
 */

import { VIEW_PRESETS, anglesForDirection, viewPresetName } from "./display";

/**
 * How much of each half-edge the chamfer cuts away.
 *
 * 0 is a plain cube with no edge or corner facets at all; 0.5 is a
 * cuboctahedron with no face left. 0.22 leaves the six faces at 56% of the
 * cube's width — enough for BOTTOM to be legible across one — with bevels and
 * corners big enough to be pressed without aiming.
 */
export const CHAMFER = 0.22;

/** Distance from the centre to a face plane. The cube's own half-edge. */
const OUTER = 1;
/** How far a facet's own corners sit from the axis it is cut back along. */
const INNER = 1 - 2 * CHAMFER;

export type Vec3 = readonly [number, number, number];

/** Face, bevel, or corner — how many axes the direction commits to. */
export type FacetRank = "face" | "edge" | "corner";

export interface Facet {
  /** VIEW_PRESETS key this facet snaps the camera to. */
  key: string;
  /**
   * The view directly opposite — where Shift-clicking this facet goes.
   *
   * Twenty of the twenty-six facets are turned away from the camera at any
   * moment, and a facet you cannot see is a facet you cannot press: from above
   * the floor there is no BOTTOM to click, ever. The view keys already answer
   * this — Ctrl+7 is Blender's Bottom, the far side of 7's Top — so the cube
   * answers it the same way rather than inventing a second idiom, under Shift
   * rather than Ctrl because macOS takes Control-click for itself (see
   * `choose` in `components/ViewCube.tsx`).
   */
  opposite: string;
  rank: FacetRank;
  /** Outward normal, normalized, in world axes. */
  normal: Vec3;
  /** The facet's outline, in cube units, wound around its normal. */
  polygon: Vec3[];
  /** Centre of the outline — where a label goes. */
  centre: Vec3;
  /** In-plane axes for a face's label: `right` and `up` as the label reads. */
  labelRight: Vec3;
  labelUp: Vec3;
  /** FRONT, TOP, … — faces only; the other twenty carry no type. */
  label: string;
}

const FACE_LABELS: Record<string, string> = {
  front: "FRONT",
  back: "BACK",
  left: "LEFT",
  right: "RIGHT",
  top: "TOP",
  bottom: "BOTTOM",
};

const scale = (v: Vec3, k: number): Vec3 => [v[0] * k, v[1] * k, v[2] * k];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const normalize = (v: Vec3): Vec3 => {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return scale(v, 1 / length);
};

const centroid = (points: Vec3[]): Vec3 =>
  scale(points.reduce(add, [0, 0, 0] as Vec3), 1 / points.length);

/**
 * Reverse a polygon if it is wound clockwise about its own normal.
 *
 * Every facet is generated from a sign pattern, and whether that pattern comes
 * out anticlockwise depends on the handedness of the two axes it was built
 * over — which flips for the Y faces, because X × Z is −Y. Rather than
 * special-casing an axis (and getting it wrong), each outline is measured
 * against its own normal and turned round when it disagrees. Consistent
 * winding is what lets a renderer cull by winding and by normal and get the
 * same answer, and it is asserted in `test/viewRose.test.ts`.
 */
function wind(polygon: Vec3[], normal: Vec3): Vec3[] {
  const edgeA = polygon[1].map((value, axis) => value - polygon[0][axis]) as unknown as Vec3;
  const edgeB = polygon[2].map((value, axis) => value - polygon[1][axis]) as unknown as Vec3;
  return dot(cross(edgeA, edgeB), normal) > 0 ? polygon : [...polygon].reverse();
}

/** A point with `outer` on one axis, `inner` on another, `free` on the third. */
function corner(axes: [number, number, number], values: [number, number, number]): Vec3 {
  const point: [number, number, number] = [0, 0, 0];
  point[axes[0]] = values[0];
  point[axes[1]] = values[1];
  point[axes[2]] = values[2];
  return point;
}

/**
 * All twenty-six facets of the chamfered cube.
 *
 * The solid's vertices are every signed permutation of (1, 1 − 2k, 1 − 2k),
 * and each facet is read straight off them:
 *
 *   face   — the four points with one axis at ±1 and the others at ±INNER
 *   bevel  — the four points spanning two axes, (OUTER, INNER) and (INNER,
 *            OUTER) on that pair, at ±INNER along the third
 *   corner — the three points that put OUTER on two axes and INNER on the
 *            remaining one
 *
 * Generated rather than typed out, because the ViewCube derives a camera
 * direction from each facet's normal and `VIEW_PRESETS` derives one from the
 * same sign triple; a hand-written table is where those two drift apart.
 */
export const FACETS: Facet[] = (() => {
  const facets: Facet[] = [];
  const axes = [0, 1, 2] as const;

  // ── the six faces ──
  for (const axis of axes) {
    for (const sign of [-1, 1] as const) {
      const [u, v] = axes.filter((other) => other !== axis);
      const normal: Vec3 = corner([axis, u, v], [sign, 0, 0]);
      const quadrants: [number, number][] = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ];
      const polygon = wind(
        quadrants.map(([su, sv]) =>
          corner([axis, u, v], [sign * OUTER, su * INNER, sv * INNER]),
        ),
        normal,
      );
      const key = viewPresetName(normal.map(Math.sign) as unknown as Vec3);
      // The label reads across the face and up the world — except on the two
      // polar faces, where "up the world" is the direction being looked along
      // and would foreshorten the type to a line. Both of them borrow +Y
      // instead, and both borrow the *same* +Y: that is the axis the camera's
      // own screen-up collapses onto at either pole (`cameraBasisFor` falls
      // back to +Y there, for the same degeneracy), so a label pinned to it
      // stands upright whichever pole you are under. Giving BOTTOM −Y — one
      // sign, mirroring TOP as if the cube were being unfolded — is what drew
      // it as "WOTTO8", upside down in its own view; `test/viewRose.test.ts`
      // now measures the projected basis rather than only its squareness.
      const labelUp: Vec3 = axis === 2 ? [0, 1, 0] : [0, 0, 1];
      facets.push({
        key,
        opposite: viewPresetName(scale(normal, -1).map(Math.sign) as unknown as Vec3),
        rank: "face",
        normal,
        polygon,
        centre: centroid(polygon),
        labelUp,
        labelRight: cross(labelUp, normal),
        label: FACE_LABELS[key],
      });
    }
  }

  // ── the twelve bevels ──
  for (const first of axes) {
    for (const second of axes) {
      if (second <= first) continue;
      const free = axes.find((axis) => axis !== first && axis !== second)!;
      for (const signA of [-1, 1] as const) {
        for (const signB of [-1, 1] as const) {
          const normal = normalize(corner([first, second, free], [signA, signB, 0]));
          const polygon = wind(
            [
              corner([first, second, free], [signA * OUTER, signB * INNER, -INNER]),
              corner([first, second, free], [signA * OUTER, signB * INNER, INNER]),
              corner([first, second, free], [signA * INNER, signB * OUTER, INNER]),
              corner([first, second, free], [signA * INNER, signB * OUTER, -INNER]),
            ],
            normal,
          );
          facets.push({
            key: viewPresetName(
              corner([first, second, free], [signA, signB, 0]) as Vec3,
            ),
            opposite: viewPresetName(
              corner([first, second, free], [-signA, -signB, 0]) as Vec3,
            ),
            rank: "edge",
            normal,
            polygon,
            centre: centroid(polygon),
            labelRight: [1, 0, 0],
            labelUp: [0, 0, 1],
            label: "",
          });
        }
      }
    }
  }

  // ── the eight corners ──
  for (const sx of [-1, 1] as const) {
    for (const sy of [-1, 1] as const) {
      for (const sz of [-1, 1] as const) {
        const signs: Vec3 = [sx, sy, sz];
        const normal = normalize(signs);
        // A corner facet's three vertices are the nearest corner of each of
        // the three faces meeting there: OUTER on one axis, INNER on the other
        // two. Written the other way round — OUTER on two — it is a triangle
        // on the wrong plane, floating outside the silhouette with a wedge of
        // nothing between it and the cube.
        const polygon = wind(
          [0, 1, 2].map(
            (outer) =>
              [0, 1, 2].map(
                (axis) => signs[axis] * (axis === outer ? OUTER : INNER),
              ) as unknown as Vec3,
          ),
          normal,
        );
        facets.push({
          key: viewPresetName(signs),
          opposite: viewPresetName([-sx, -sy, -sz]),
          rank: "corner",
          normal,
          polygon,
          centre: centroid(polygon),
          labelRight: [1, 0, 0],
          labelUp: [0, 0, 1],
          label: "",
        });
      }
    }
  }

  return facets;
})();

export interface Basis {
  /** Unit vector from the origin toward the camera. */
  direction: Vec3;
  right: Vec3;
  up: Vec3;
}

/**
 * The camera's screen basis, built exactly as the viewport builds its own.
 *
 * `viewer/math.ts` picks +Y as the up reference at the poles, where +Z is
 * degenerate; the same choice is made here, so the cube spins the way the
 * scene behind it spins rather than half a turn away from it.
 */
export function cameraBasisFor(yaw: number, pitch: number): Basis {
  const cosPitch = Math.cos(pitch);
  const direction: Vec3 = [
    cosPitch * Math.sin(yaw),
    -cosPitch * Math.cos(yaw),
    Math.sin(pitch),
  ];
  const forward = scale(direction, -1);
  const reference: Vec3 = Math.abs(forward[2]) > 0.999 ? [0, 1, 0] : [0, 0, 1];
  const right = normalize(cross(forward, reference));
  return { direction, right, up: cross(right, forward) };
}

/** Orthographic projection to SVG coordinates, y down, in cube units. */
export function project(point: Vec3, basis: Basis): [number, number] {
  return [dot(point, basis.right), -dot(point, basis.up)];
}

/** True when the camera is on the outside of this facet. */
export const facetVisible = (facet: Facet, basis: Basis): boolean =>
  dot(facet.normal, basis.direction) > 1e-6;

/**
 * The facet squarely in front, or null if the camera is somehow nowhere.
 *
 * The largest dot with the camera direction. This is the readout half of the
 * widget's job: what you see facing you has to be what you are looking at.
 */
export function frontFacet(basis: Basis): Facet | null {
  let best: Facet | null = null;
  let bestDot = -Infinity;
  for (const facet of FACETS) {
    const value = dot(facet.normal, basis.direction);
    if (value > bestDot) {
      bestDot = value;
      best = facet;
    }
  }
  return best;
}

/** Which of the camera's two freedoms a turn control steps. */
export type TurnAxis = "azimuth" | "elevation";

/** Zero out floating-point dust so a polar direction is exactly polar. */
const tidy = (v: Vec3): Vec3 => v.map((value) => (Math.abs(value) < 1e-9 ? 0 : value)) as unknown as Vec3;

/**
 * True where the camera draws itself with the polar convention.
 *
 * The same threshold as `cameraBasisFor` (and the renderer's `cameraBasis`):
 * inside it the screen basis is built on +Y rather than on world up, so a
 * yaw no longer turns anything the reader can see, and the turn controls have
 * to reason about the screen the reader is looking at rather than the angle.
 */
const atPole = (pitch: number): boolean => Math.abs(Math.sin(pitch)) > 0.999;

/**
 * Turn the camera a quarter turn about one of its own screen axes.
 *
 * The rule, in one sentence: **the control moves the camera a quarter turn
 * toward the side of the screen it sits on, and from every standpoint it
 * changes the view.** The two pairs are stepped for what they are.
 *
 * *Left and right* are a turntable — a quarter turn about world up, which is
 * the axis that keeps the horizon level, so from an isometric corner the
 * right control reaches the next corner round rather than dropping to the
 * horizon. Azimuth has no ends, so four presses walk Front → Right → Back →
 * Left → Front. At a pole the turntable is degenerate: world up *is* the view
 * axis, a yaw is a spin the camera cannot show (its up vector is pinned, see
 * `cameraBasisFor`), and a control that does nothing reads as broken. There
 * the turn is taken about the screen's own vertical instead — the camera
 * moves onto its screen-right axis — which lands on the face the reader can
 * see on that side of the cube: from TOP, right reaches RIGHT; from BOTTOM,
 * where world −X is on the screen's right, it reaches LEFT.
 *
 * *Up and down* are a quarter turn about the screen horizontal, the camera
 * moving toward its own screen-up: the new direction *is* the old up vector.
 * Off the poles that is pitch ± 90°, and when it would carry past a pole it
 * keeps going onto the far side, upright — this camera has no roll, so it
 * cannot come out inverted the way FreeCAD's does, and the walk is Front →
 * Top → Back → Top → Front instead of the rolling camera's four-cycle. That
 * is the honest walk for a camera whose up is pinned: "up" always means
 * toward what is currently up on screen. At the poles screen-up is +Y by the
 * same convention the cube and the scene are drawn with, so from TOP up
 * reaches BACK and down reaches FRONT, exactly the faces the reader sees
 * above and below the TOP label. Earlier versions clamped at the pole and
 * greyed the control out as "spent"; a greyed control was read as one that
 * did not work, and it is gone.
 */
export function stepView(
  yaw: number,
  pitch: number,
  axis: TurnAxis,
  turns: 1 | -1,
): { yaw: number; pitch: number } {
  if (axis === "azimuth" && !atPole(pitch)) {
    return { yaw: yaw + (turns * Math.PI) / 2, pitch };
  }
  const basis = cameraBasisFor(yaw, pitch);
  const target = tidy(scale(axis === "azimuth" ? basis.right : basis.up, turns));
  const angles = anglesForDirection(target[0], target[1], target[2]);
  if (axis === "azimuth") return angles;
  // Straight up or down has no azimuth, and `anglesForDirection` answers 0;
  // keep the meridian the camera arrived on instead, as pitch ± 90° would.
  if (target[0] === 0 && target[1] === 0) return { yaw, pitch: angles.pitch };
  // Keep the yaw continuous on the way over: an elevation step stays on the
  // camera's own meridian or lands on the antipodal one, and the principal
  // value `anglesForDirection` returns may be a whole turn away from the yaw
  // the camera is carrying. Add the wrapped difference rather than snapping.
  const wrapped = angles.yaw - yaw;
  const turn = Math.atan2(Math.sin(wrapped), Math.cos(wrapped));
  return { yaw: yaw + turn, pitch: angles.pitch };
}

/** The world axis triad, as the orientation legend draws it. */
export const TRIAD: { axis: Vec3; label: string; token: string }[] = [
  { axis: [1, 0, 0], label: "X", token: "axis-x" },
  { axis: [0, 1, 0], label: "Y", token: "axis-y" },
  { axis: [0, 0, 1], label: "Z", token: "axis-z" },
];

/** Every facet's preset must exist; asserted once, at module load. */
export const FACET_KEYS = FACETS.map((facet) => facet.key);

if (
  FACETS.some(
    (facet) =>
      VIEW_PRESETS[facet.key] === undefined || VIEW_PRESETS[facet.opposite] === undefined,
  )
) {
  throw new Error("navCube: a facet names a view the preset table does not have");
}
