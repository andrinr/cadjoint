/**
 * The view rose: twenty-six standard views, and the one place they are named.
 *
 * Two things are held here. The *table* — `VIEW_PRESETS` — has to contain
 * every direction exactly once, at the exact angles those directions imply,
 * and it has to call all eight corners ISO because that is what an isometric
 * direction is called in any octant. And the *cube* has to agree with it: the
 * facet turned toward you must be the facet the camera is standing on, for
 * every direction and not merely for the one that was looked at while
 * building it. That second claim is the one the widget shipped wrong — its
 * CSS pitch rotation doubled and mirrored the elevation, so a camera above the
 * floor was shown the BOTTOM face — and it is invisible at pitch 0, which is
 * how it survived.
 */

import { describe, expect, it } from "vitest";
import {
  VIEW_PRESETS,
  anglesForDirection,
  matchViewPreset,
  sameView,
  viewPresetName,
} from "../src/viewer/display";
import {
  CHAMFER,
  FACETS,
  cameraBasisFor,
  cross,
  dot,
  facetVisible,
  frontFacet,
  project,
  stepView,
  type Vec3,
} from "../src/viewer/navCube";
import { octant } from "../src/viewer/graticule";

/** The camera offset the app builds from a yaw/pitch, in a Z-up world. */
const offsetOf = (yaw: number, pitch: number): [number, number, number] => [
  Math.cos(pitch) * Math.sin(yaw),
  -Math.cos(pitch) * Math.cos(yaw),
  Math.sin(pitch),
];

const sign = (value: number) => (value > 1e-9 ? 1 : value < -1e-9 ? -1 : 0);

describe("the preset table", () => {
  it("holds all twenty-six directions, plus the iso alias", () => {
    // 6 faces + 12 edges + 8 corners = 26, and `iso` names one of them twice.
    expect(Object.keys(VIEW_PRESETS)).toHaveLength(27);
    const directions = new Set<string>();
    for (const preset of Object.values(VIEW_PRESETS)) {
      directions.add(offsetOf(preset.yaw, preset.pitch).map(sign).join(","));
    }
    expect(directions.size).toBe(26);
  });

  it("puts every preset on the direction its name spells", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      if (name === "iso") continue;
      const direction = offsetOf(preset.yaw, preset.pitch).map(sign) as [
        number,
        number,
        number,
      ];
      expect(viewPresetName(direction), name).toBe(name);
    }
  });

  it("keeps iso on the +X−Y+Z corner the session opens on", () => {
    expect(octant(VIEW_PRESETS.iso.yaw, VIEW_PRESETS.iso.pitch)).toBe("+X−Y+Z");
    expect(VIEW_PRESETS.iso).toEqual({ ...anglesForDirection(1, -1, 1), label: "ISO" });
    // A true 1:1:1 direction, not angles that merely look isometric: the
    // elevation of an isometric standpoint is atan(1/√2).
    expect(VIEW_PRESETS.iso.pitch).toBeCloseTo(Math.atan(1 / Math.SQRT2), 12);
  });

  it("gives the two polar views no azimuth to get wrong", () => {
    // `atan2(0, -0)` is π, not 0, so generating these naively yaws them half a
    // turn — which showed up as an upside-down TOP label on the cube.
    expect(VIEW_PRESETS.top.yaw).toBe(0);
    expect(VIEW_PRESETS.bottom.yaw).toBe(0);
  });

  it("matches a camera to a view, and refuses one that has orbited off", () => {
    expect(matchViewPreset(VIEW_PRESETS.front.yaw, VIEW_PRESETS.front.pitch)).toBe("front");
    expect(matchViewPreset(VIEW_PRESETS.front.yaw + 0.4, 0)).toBeNull();
    // Yaw wraps: a camera one full turn round is on the same view.
    expect(sameView(VIEW_PRESETS.right.yaw + 2 * Math.PI, 0, VIEW_PRESETS.right)).toBe(true);
    // Looking straight down, yaw is a spin about the view axis.
    expect(sameView(1.234, Math.PI / 2, VIEW_PRESETS.top)).toBe(true);
  });
});

describe("the chamfered cube's facets", () => {
  it("has one facet per standard view, in three ranks", () => {
    expect(FACETS).toHaveLength(26);
    const ranks = FACETS.map((facet) => facet.rank);
    expect(ranks.filter((rank) => rank === "face")).toHaveLength(6);
    expect(ranks.filter((rank) => rank === "edge")).toHaveLength(12);
    expect(ranks.filter((rank) => rank === "corner")).toHaveLength(8);
    expect(new Set(FACETS.map((facet) => facet.key)).size).toBe(26);
    for (const facet of FACETS) expect(VIEW_PRESETS[facet.key], facet.key).toBeDefined();
  });

  it("names each facet after the direction its own normal points", () => {
    for (const facet of FACETS) {
      const direction = facet.normal.map(sign) as [number, number, number];
      expect(viewPresetName(direction), facet.key).toBe(facet.key);
      // And the preset it snaps to is on that same direction.
      const preset = VIEW_PRESETS[facet.key];
      expect(offsetOf(preset.yaw, preset.pitch).map(sign), facet.key).toEqual([
        ...direction,
      ]);
    }
  });

  it("draws a closed, planar polygon on each facet's own plane", () => {
    for (const facet of FACETS) {
      expect(facet.polygon.length, facet.key).toBe(
        facet.rank === "corner" ? 3 : 4,
      );
      // Every vertex is the same distance along the normal — that is what
      // makes it a facet of a convex solid rather than a bent quad.
      const plane = dot(facet.polygon[0], facet.normal);
      for (const vertex of facet.polygon) {
        expect(dot(vertex, facet.normal), facet.key).toBeCloseTo(plane, 9);
      }
      expect(dot(facet.centre, facet.normal), facet.key).toBeCloseTo(plane, 9);
      // Wound about the outward normal, so a renderer that culls by winding
      // and one that culls by normal agree.
      const a = facet.polygon[1].map(
        (value, axis) => value - facet.polygon[0][axis],
      ) as unknown as Vec3;
      const b = facet.polygon[2].map(
        (value, axis) => value - facet.polygon[1][axis],
      ) as unknown as Vec3;
      expect(dot(cross(a, b), facet.normal), facet.key).toBeGreaterThan(0);
    }
  });

  it("puts the three ranks at the distances a chamfer implies", () => {
    // A cube of half-edge 1 cut back by `CHAMFER` on each half-edge: faces at
    // 1, bevels at (2 − 2k)/√2, corners at (3 − 4k)/√3. Getting the corner
    // plane wrong is not subtle — it floats the triangle off the solid.
    const at = (rank: string) =>
      FACETS.filter((facet) => facet.rank === rank).map((facet) =>
        dot(facet.centre, facet.normal),
      );
    for (const distance of at("face")) expect(distance).toBeCloseTo(1, 9);
    for (const distance of at("edge")) {
      expect(distance).toBeCloseTo((2 - 2 * CHAMFER) / Math.SQRT2, 9);
    }
    for (const distance of at("corner")) {
      expect(distance).toBeCloseTo((3 - 4 * CHAMFER) / Math.sqrt(3), 9);
    }
    // The chamfer has to actually cut something, or there are no edge and
    // corner targets to click.
    expect(CHAMFER).toBeGreaterThan(0);
    expect(CHAMFER).toBeLessThan(0.5);
  });

  it("shares every corner vertex with the three facets around it", () => {
    // The chamfer only reads as a chamfer if the solid is closed: every
    // vertex of a corner triangle has to also be a vertex of the bevel and
    // the face beside it, or there is a wedge of nothing between them.
    const key = (point: Vec3) => point.map((value) => value.toFixed(6)).join(",");
    const vertices = new Map<string, number>();
    for (const facet of FACETS) {
      for (const point of facet.polygon) {
        vertices.set(key(point), (vertices.get(key(point)) ?? 0) + 1);
      }
    }
    // 24 vertices — each has exactly one coordinate at ±1, so it belongs to
    // exactly one face — and four facets meet at every one of them: that
    // face, the two bevels either side, and the corner triangle.
    expect(vertices.size).toBe(24);
    for (const [point, count] of vertices) expect(count, point).toBe(4);
  });

  it("points each facet's Shift-click at the view straight through the cube", () => {
    // Twenty of the twenty-six facets are turned away at any moment, so a
    // facet you cannot see is a view you could not otherwise ask for. The
    // modifier takes the far side, the way Ctrl+7 is Bottom to 7's Top —
    // which is only true if `opposite` really is the antipode.
    const byKey = new Map(FACETS.map((facet) => [facet.key, facet]));
    for (const facet of FACETS) {
      const other = byKey.get(facet.opposite);
      expect(other, facet.key).toBeDefined();
      expect(other!.rank, facet.key).toBe(facet.rank);
      // Antipodal normals, and the relation is its own inverse.
      for (const axis of [0, 1, 2]) {
        expect(other!.normal[axis], `${facet.key}.${axis}`).toBeCloseTo(
          -facet.normal[axis],
          12,
        );
      }
      expect(other!.opposite, facet.key).toBe(facet.key);
    }
    expect(byKey.get("top")!.opposite).toBe("bottom");
    expect(byKey.get("front")!.opposite).toBe("back");
    expect(byKey.get("front-right-top")!.opposite).toBe("back-left-bottom");
  });

  it("exactly doubles what the cube can reach from any standpoint", () => {
    // The modifier earns its place by never being redundant: a facet and its
    // antipode are never both on the near side of a convex solid, so Shift
    // reaches twenty views a plain click cannot — including, from anywhere above
    // the floor, BOTTOM, which is otherwise unclickable at every standpoint
    // the camera is allowed to occupy.
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      const shown = FACETS.filter((facet) => facetVisible(facet, basis));
      const plain = new Set(shown.map((facet) => facet.key));
      const shifted = new Set(shown.map((facet) => facet.opposite));
      for (const key of shifted) expect(plain.has(key), `${name} ${key}`).toBe(false);
      expect(plain.size + shifted.size, name).toBe(2 * shown.length);
    }
    // From anywhere actually *above* the floor — which is where the camera
    // spends its life — BOTTOM is one Shift-click away. (Dead-on FRONT is the
    // boundary case: the top face is edge-on there, visible from neither
    // side, so it is not clickable and neither is its antipode.)
    for (let yaw = -Math.PI; yaw < Math.PI; yaw += 0.23) {
      for (const pitch of [0.05, 0.4, 0.95, 1.45]) {
        const label = `yaw ${yaw.toFixed(2)} pitch ${pitch}`;
        const basis = cameraBasisFor(yaw, pitch);
        const reachable = FACETS.filter((facet) => facetVisible(facet, basis)).map(
          (facet) => facet.opposite,
        );
        expect(reachable, label).toContain("bottom");
      }
    }
  });

  it("keeps every face's label basis square to its own face", () => {
    for (const facet of FACETS.filter((entry) => entry.rank === "face")) {
      expect(dot(facet.labelRight, facet.normal), facet.key).toBeCloseTo(0, 9);
      expect(dot(facet.labelUp, facet.normal), facet.key).toBeCloseTo(0, 9);
      expect(dot(facet.labelRight, facet.labelUp), facet.key).toBeCloseTo(0, 9);
      expect(facet.label, facet.key).toBe(facet.key.toUpperCase());
    }
  });

  it("stands every face's label up in the view that face names", () => {
    // Squareness is not orientation: a basis can be perfectly orthonormal and
    // still be turned half a circle, which is how BOTTOM shipped reading
    // "WOTTO8" from directly below. Project each label's own basis from the
    // standpoint the face is named after and require the plain identity —
    // `labelRight` along screen +x, `labelUp` along screen −y, since SVG's y
    // runs down the page. Positive determinant alone would let a 180° through.
    for (const facet of FACETS.filter((entry) => entry.rank === "face")) {
      const preset = VIEW_PRESETS[facet.key];
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      const [rx, ry] = project(facet.labelRight, basis);
      const [ux, uy] = project(facet.labelUp, basis);
      expect(rx, `${facet.key} right.x`).toBeCloseTo(1, 9);
      expect(ry, `${facet.key} right.y`).toBeCloseTo(0, 9);
      expect(ux, `${facet.key} up.x`).toBeCloseTo(0, 9);
      expect(uy, `${facet.key} up.y`).toBeCloseTo(-1, 9);
    }
  });
});

describe("the cube agrees with the camera", () => {
  const expectedFacet = (yaw: number, pitch: number) => {
    const direction = offsetOf(yaw, pitch);
    return FACETS.reduce((best, facet) =>
      dot(facet.normal, direction) > dot(best.normal, direction) ? facet : best,
    );
  };

  it("turns the camera's own facet to the front at all twenty-six views", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      expect(frontFacet(basis)!.key, name).toBe(
        expectedFacet(preset.yaw, preset.pitch).key,
      );
      // …and it is the facet the preset is named after.
      if (name !== "iso") expect(frontFacet(basis)!.key, name).toBe(name);
    }
  });

  it("holds over a sweep of free standpoints", () => {
    for (let yaw = -Math.PI; yaw < Math.PI; yaw += 0.17) {
      for (let pitch = -1.45; pitch <= 1.45; pitch += 0.11) {
        const label = `yaw ${yaw.toFixed(2)} pitch ${pitch.toFixed(2)}`;
        expect(frontFacet(cameraBasisFor(yaw, pitch))!.key, label).toBe(
          expectedFacet(yaw, pitch).key,
        );
      }
    }
  });

  it("shows exactly the facets the camera is outside of", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      const basis = cameraBasisFor(preset.yaw, preset.pitch);
      const direction = offsetOf(preset.yaw, preset.pitch);
      const shown = FACETS.filter((facet) => facetVisible(facet, basis))
        .map((facet) => facet.key)
        .sort();
      const facing = FACETS.filter((facet) => dot(facet.normal, direction) > 1e-6)
        .map((facet) => facet.key)
        .sort();
      expect(shown, name).toEqual(facing);
      // A convex solid seen from outside never shows more than half of it.
      expect(shown.length, name).toBeLessThan(FACETS.length);
      expect(shown.length, name).toBeGreaterThan(0);
    }
  });

  it("shows TOP from above and BOTTOM from below", () => {
    const facing = (name: string) => {
      const preset = VIEW_PRESETS[name];
      return frontFacet(cameraBasisFor(preset.yaw, preset.pitch))!.label;
    };
    expect(facing("top")).toBe("TOP");
    expect(facing("bottom")).toBe("BOTTOM");
    // The standpoint from the screenshot that exposed the bug: above the
    // floor, so never BOTTOM whatever the azimuth.
    const corner = VIEW_PRESETS["front-left-top"];
    const basis = cameraBasisFor(corner.yaw, corner.pitch);
    const shown = FACETS.filter(
      (facet) => facetVisible(facet, basis) && facet.rank === "face",
    )
      .map((facet) => facet.key)
      .sort();
    expect(shown).toEqual(["front", "left", "top"]);
  });

  it("projects a corner standpoint to a regular hexagon", () => {
    // The signature of a true isometric direction under a parallel
    // projection, and the reason the widget carries no CSS perspective: the
    // cube's eight vertices project onto six points at one radius.
    const preset = VIEW_PRESETS.iso;
    const basis = cameraBasisFor(preset.yaw, preset.pitch);
    const radii = new Set<string>();
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        for (const sz of [-1, 1]) {
          const [x, y] = project([sx, sy, sz], basis);
          radii.add(Math.hypot(x, y).toFixed(6));
        }
      }
    }
    // Six on the silhouette at one radius, two at the centre (the near and
    // far corners, which project on top of each other).
    expect(radii.size).toBe(2);
    expect([...radii].map(Number).sort((a, b) => a - b)[0]).toBeCloseTo(0, 9);
  });
});

describe("the quarter-turn controls", () => {
  const at = (name: string) => ({
    yaw: VIEW_PRESETS[name].yaw,
    pitch: VIEW_PRESETS[name].pitch,
  });
  const directionOf = (camera: { yaw: number; pitch: number }): Vec3 =>
    cameraBasisFor(camera.yaw, camera.pitch).direction;

  it("walk Front → Right → Back → Left → Front, and keep going", () => {
    let camera = at("front");
    for (const expected of ["right", "back", "left", "front"]) {
      camera = stepView(camera.yaw, camera.pitch, "azimuth", 1);
      expect(matchViewPreset(camera.yaw, camera.pitch), expected).toBe(expected);
    }
  });

  it("carry the camera over the pole rather than stopping at it", () => {
    // A rolling camera would walk Front → Top → Back → Bottom → Front. This
    // one has no roll, so past the pole it comes out upright on the far side,
    // and "up" always means toward whatever is up on screen — which at TOP
    // is BACK, whichever way the camera arrived: Front → Top → Back → Top →
    // Back. What matters is that every press changes the view; the old
    // version clamped at Top and greyed the control out, which read as a
    // control that did not work.
    let camera = at("front");
    for (const expected of ["top", "back", "top", "back"]) {
      camera = stepView(camera.yaw, camera.pitch, "elevation", 1);
      expect(matchViewPreset(camera.yaw, camera.pitch), expected).toBe(expected);
    }
    // BOTTOM is drawn with the same +Y up, so below the floor "down" is
    // toward FRONT: Front → Bottom → Front → Bottom.
    camera = at("front");
    for (const expected of ["bottom", "front", "bottom", "front"]) {
      camera = stepView(camera.yaw, camera.pitch, "elevation", -1);
      expect(matchViewPreset(camera.yaw, camera.pitch), expected).toBe(expected);
    }
    // And a free standpoint near the pole goes over it and lands just past
    // it on the far side, still upright.
    const over = stepView(0.3, 1.45, "elevation", 1);
    expect(over.pitch).toBeCloseTo(Math.PI - 1.45 - Math.PI / 2, 9);
    // Half a turn round, either way about: the meridian flipped.
    expect(Math.abs(Math.cos(over.yaw - 0.3) + 1)).toBeLessThan(1e-9);
  });

  it("turn toward what is on screen at the poles", () => {
    // Straight down, world up is the view axis and a yaw is an invisible
    // spin. The cube draws TOP with +Y up and +X right, so the four controls
    // reach the four faces the reader sees around the TOP label.
    const top = at("top");
    expect(matchViewPreset(...values(stepView(top.yaw, top.pitch, "elevation", 1)))).toBe("back");
    expect(matchViewPreset(...values(stepView(top.yaw, top.pitch, "elevation", -1)))).toBe("front");
    expect(matchViewPreset(...values(stepView(top.yaw, top.pitch, "azimuth", 1)))).toBe("right");
    expect(matchViewPreset(...values(stepView(top.yaw, top.pitch, "azimuth", -1)))).toBe("left");
    // From below, the screen's right is world −X: the control that sits on
    // the right reaches the face on the right, which is LEFT.
    const bottom = at("bottom");
    expect(matchViewPreset(...values(stepView(bottom.yaw, bottom.pitch, "elevation", 1)))).toBe("back");
    expect(matchViewPreset(...values(stepView(bottom.yaw, bottom.pitch, "elevation", -1)))).toBe("front");
    expect(matchViewPreset(...values(stepView(bottom.yaw, bottom.pitch, "azimuth", 1)))).toBe("left");
    expect(matchViewPreset(...values(stepView(bottom.yaw, bottom.pitch, "azimuth", -1)))).toBe("right");
    // Whatever meridian the camera reached the pole on: TOP is TOP.
    for (const yaw of [0, 0.7, Math.PI / 2, -2.4]) {
      const turned = stepView(yaw, Math.PI / 2, "azimuth", 1);
      expect(matchViewPreset(turned.yaw, turned.pitch), `yaw ${yaw}`).toBe("right");
    }
  });

  it("always change the view, from every standpoint", () => {
    // The whole point: no greyed-out control anywhere. Up and down move the
    // camera onto its own screen-up axis, a full quarter turn. Left and right
    // turn about world up, which near a pole barely moves the *direction* —
    // but spins the screen a quarter turn, and the screen is what the reader
    // sees. So what is measured is the drawn basis: some axis of it has to
    // swing well clear of where it was.
    const swung = (yaw: number, pitch: number, next: { yaw: number; pitch: number }) => {
      const before = cameraBasisFor(yaw, pitch);
      const after = cameraBasisFor(next.yaw, next.pitch);
      return Math.min(
        dot(before.direction, after.direction),
        dot(before.right, after.right),
        dot(before.up, after.up),
      );
    };
    for (let yaw = -Math.PI; yaw < Math.PI; yaw += 0.29) {
      for (const pitch of [-Math.PI / 2, -1.5, -1.45, -0.9, -0.3, 0, 0.3, 0.9, 1.45, 1.5, Math.PI / 2]) {
        const label = `yaw ${yaw.toFixed(2)} pitch ${pitch.toFixed(2)}`;
        const basis = cameraBasisFor(yaw, pitch);
        for (const turns of [1, -1] as const) {
          const up = directionOf(stepView(yaw, pitch, "elevation", turns));
          for (const axis of [0, 1, 2]) {
            expect(up[axis], `${label} up`).toBeCloseTo(turns * basis.up[axis], 9);
          }
          expect(swung(yaw, pitch, stepView(yaw, pitch, "azimuth", turns)), `${label} round`)
            .toBeLessThan(0.75);
        }
      }
    }
  });

  it("land on a standard view from an isometric standpoint", () => {
    // A corner is 35.26° up and 45° round, so a quarter turn round from it
    // lands on the mirror corner rather than nowhere the readout can name.
    const turned = stepView(VIEW_PRESETS.iso.yaw, VIEW_PRESETS.iso.pitch, "azimuth", 1);
    expect(matchViewPreset(turned.yaw, turned.pitch)).toBe("back-right-top");
  });

  it("keep the yaw continuous across an elevation step", () => {
    // Over the pole the meridian flips by half a turn, and the principal
    // value of the new azimuth may be a full turn from the one the camera is
    // carrying. The cube is drawn from the direction so a 2π jump is
    // invisible there, but nothing downstream should have to know that.
    const there = stepView(3.0, 0.9, "elevation", 1);
    expect(Math.abs(there.yaw - 3.0)).toBeLessThan(Math.PI + 1e-9);
    const same = stepView(-3.0, 0.2, "elevation", -1);
    expect(same.yaw).toBeCloseTo(-3.0, 12);
  });

  it("are exact inverses on the azimuth", () => {
    const there = stepView(0.4, 0.3, "azimuth", 1);
    const back = stepView(there.yaw, there.pitch, "azimuth", -1);
    expect(back.yaw).toBeCloseTo(0.4, 12);
    expect(back.pitch).toBeCloseTo(0.3, 12);
  });
});

const values = (camera: { yaw: number; pitch: number }): [number, number] => [
  camera.yaw,
  camera.pitch,
];
