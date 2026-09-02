/**
 * The ground grid's two contracts.
 *
 * 1. The spacing is a *measurement*: the lines really are a definite number of
 *    millimetres apart in the world, derived from the camera and the scene's
 *    declared unit, always on the 1-2-5 ladder so the readout never has to
 *    hedge about the number itself. What it does still hedge about is whether
 *    you can measure with it *on screen*: under perspective a cell shrinks
 *    with depth, and the reading is marked uncalibrated. A grid whose readout
 *    can be wrong is decoration.
 * 2. The tones are *furniture*: measured against paper they must land inside
 *    a band, not merely above a floor. Too weak and the grid is invisible;
 *    too strong and it competes with the field it exists to sit behind.
 */

import { describe, expect, it } from "vitest";
import {
  DIVISIONS,
  MM_PER_UNIT,
  distanceForGain,
  formatGain,
  gainFor,
  gainOf,
  GRID_ALPHA,
  GRID_FADE,
  GRID_MAJOR_EVERY,
  gridSpacing,
  isDetented,
  nearestDetent,
  octant,
  stepDetent,
  viewLabel,
  worldHeightFor,
} from "../src/viewer/graticule";
import { FOV_SCALE, orthoHeightFor } from "../src/viewer/math";
import { VIEW_PRESETS } from "../src/viewer/display";
import { detentZoomCamera, MAX_DISTANCE, MIN_DISTANCE } from "../src/components/viewer/camera";
import { contrastRatio } from "../src/simColors";
import { CHROME, GRATICULE_TONES, hexToRgb } from "../src/tokens";

const PAPER = hexToRgb(CHROME["surface-viewport"]);

describe("the 1-2-5 ladder", () => {
  it("snaps to the nearest rung on a log scale", () => {
    expect(nearestDetent(10)).toBe(10);
    expect(nearestDetent(11)).toBe(10);
    expect(nearestDetent(9)).toBe(10);
    expect(nearestDetent(862.5)).toBe(1000);
    expect(nearestDetent(3)).toBe(2);
    expect(nearestDetent(3.5)).toBe(5);
    expect(nearestDetent(0.07)).toBe(0.05);
    expect(nearestDetent(7000)).toBe(5000);
  });

  it("only ever returns a 1, 2 or 5 followed by zeros", () => {
    for (const sample of [0.013, 0.4, 7.7, 63, 480, 3300, 91_000]) {
      const rung = nearestDetent(sample);
      const mantissa = rung / 10 ** Math.round(Math.log10(rung / 5) + Math.log10(5));
      expect([1, 2, 5], `${sample} -> ${rung}`).toContain(Math.round(mantissa));
    }
  });

  it("refuses a scale that is not a positive number", () => {
    expect(nearestDetent(0)).toBeNaN();
    expect(nearestDetent(-4)).toBeNaN();
    expect(nearestDetent(Number.NaN)).toBeNaN();
  });

  it("steps a full rung even from a rung it is already sitting on", () => {
    expect(stepDetent(10, 1)).toBe(20);
    expect(stepDetent(10, -1)).toBe(5);
    expect(stepDetent(5, -1)).toBe(2);
    expect(stepDetent(2, -1)).toBe(1);
    expect(stepDetent(1, -1)).toBeCloseTo(0.5, 12);
    expect(stepDetent(1, 1)).toBe(2);
  });

  it("steps to the next rung in the direction of travel when off-ladder", () => {
    expect(stepDetent(862.5, 1)).toBe(1000);
    expect(stepDetent(862.5, -1)).toBe(500);
    expect(stepDetent(1.4, 1)).toBe(2);
    expect(stepDetent(1.4, -1)).toBe(1);
  });

  it("walks a full decade in three steps, both ways", () => {
    let up = 10;
    for (let step = 0; step < 3; step++) up = stepDetent(up, 1);
    expect(up).toBe(100);
    let down = 100;
    for (let step = 0; step < 3; step++) down = stepDetent(down, -1);
    expect(down).toBeCloseTo(10, 9);
  });

  it("reports detent only on the ladder itself", () => {
    expect(isDetented(10)).toBe(true);
    expect(isDetented(1000)).toBe(true);
    expect(isDetented(0.2)).toBe(true);
    expect(isDetented(862.5)).toBe(false);
    expect(isDetented(11)).toBe(false);
  });
});

describe("the scale derivation", () => {
  it("divides the framed world height into eight square divisions", () => {
    // 1 world unit = 1 m, the only length the repository declares
    // (cadjoint/meshing/export.py stamps SI_UNIT($,.METRE.)).
    expect(MM_PER_UNIT).toBe(1000);
    expect(gainFor(8)).toBe(1000);
    expect(gainFor(DIVISIONS * 0.01)).toBeCloseTo(10, 9);
  });

  it("round-trips gain, world height and orbit distance", () => {
    for (const mm of [75, 100, 862.5, 1000, 11_250]) {
      expect(worldHeightFor(mm)).toBeCloseTo((mm / MM_PER_UNIT) * DIVISIONS, 12);
      expect(gainFor(worldHeightFor(mm))).toBeCloseTo(mm, 9);
      expect(orthoHeightFor(distanceForGain(mm))).toBeCloseTo(worldHeightFor(mm), 9);
      expect(distanceForGain(mm)).toBeCloseTo(worldHeightFor(mm) / FOV_SCALE, 12);
    }
  });

  it("snaps the default framing to a whole rung", () => {
    // The renderer's starting camera: distance 4.6, FOV_SCALE 1.5, which
    // frames 862.5 mm to a division — so the grid is ruled at 1 m, and the
    // readout states 1 m rather than 863.
    expect(gainFor(orthoHeightFor(4.6))).toBeCloseTo(862.5, 9);
    expect(gainOf(4.6, "orthographic").mm).toBe(1000);
  });

  it("only ever states a spacing that is on the ladder", () => {
    for (let distance = MIN_DISTANCE; distance < MAX_DISTANCE; distance *= 1.07) {
      const { mm } = gainOf(distance, "orthographic");
      expect(isDetented(mm), `${distance}`).toBe(true);
    }
  });

  it("rules the grid in world units at the spacing it states", () => {
    for (const distance of [MIN_DISTANCE, 1, 4.6, 20, MAX_DISTANCE]) {
      expect(gridSpacing(distance)).toBeCloseTo(
        gainOf(distance, "orthographic").mm / MM_PER_UNIT,
        12,
      );
    }
    expect(gridSpacing(4.6)).toBeCloseTo(1, 12);
  });

  it("is calibrated under an orthographic camera, where the image is to scale", () => {
    const gain = gainOf(distanceForGain(1000), "orthographic");
    expect(gain.mm).toBeCloseTo(1000, 9);
    expect(gain.calibrated).toBe(true);
  });

  it("is never calibrated under perspective, where a cell shrinks with depth", () => {
    const gain = gainOf(distanceForGain(1000), "perspective");
    expect(gain.mm).toBeCloseTo(1000, 9);
    expect(gain.calibrated).toBe(false);
  });

  it("covers the whole zoom range with rungs", () => {
    const closest = gainOf(MIN_DISTANCE, "orthographic").mm;
    const furthest = gainOf(MAX_DISTANCE, "orthographic").mm;
    expect(closest).toBe(100);
    expect(furthest).toBe(10_000);
    const reachable = [100, 200, 500, 1000, 2000, 5000, 10_000];
    for (const rung of reachable) {
      const distance = distanceForGain(rung);
      expect(distance, `${rung} mm/div`).toBeGreaterThanOrEqual(MIN_DISTANCE);
      expect(distance, `${rung} mm/div`).toBeLessThanOrEqual(MAX_DISTANCE);
    }
  });
});

describe("the gain readout", () => {
  it("prints three significant figures and switches unit at a metre", () => {
    expect(formatGain(100, true)).toMatchObject({ value: "100", unit: "mm" });
    expect(formatGain(10, true)).toMatchObject({ value: "10.0", unit: "mm" });
    expect(formatGain(1000, true)).toMatchObject({ value: "1.00", unit: "m" });
    expect(formatGain(11_250, true)).toMatchObject({ value: "11.3", unit: "m" });
  });

  it("prefixes an uncalibrated scale with >, as a 2465 does", () => {
    expect(formatGain(862.5, false).text).toBe(">863");
    expect(formatGain(1000, false).text).toBe(">1.00");
    expect(formatGain(1000, true).text).toBe("1.00");
  });

  it("states an absent scale rather than printing a wrong one", () => {
    expect(formatGain(0, true).text).toBe("—");
    expect(formatGain(Number.NaN, true).text).toBe("—");
  });
});

describe("the view readout", () => {
  it("names a standard view and reports FREE once orbited off it", () => {
    for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
      expect(viewLabel(preset.yaw, preset.pitch)).toBe(name.toUpperCase());
    }
    expect(viewLabel(VIEW_PRESETS.front.yaw + 0.4, 0)).toBe("FREE");
  });

  it("spells the octant the camera stands in, in a Z-up world", () => {
    // Front looks along +Y from −Y; Top stands on +Z; Left stands on −X.
    expect(octant(VIEW_PRESETS.iso.yaw, VIEW_PRESETS.iso.pitch)).toBe("+X−Y+Z");
    expect(octant(VIEW_PRESETS.front.yaw, VIEW_PRESETS.front.pitch)).toBe("−Y");
    expect(octant(VIEW_PRESETS.back.yaw, VIEW_PRESETS.back.pitch)).toBe("+Y");
    expect(octant(VIEW_PRESETS.left.yaw, VIEW_PRESETS.left.pitch)).toBe("−X");
    expect(octant(VIEW_PRESETS.right.yaw, VIEW_PRESETS.right.pitch)).toBe("+X");
    expect(octant(VIEW_PRESETS.top.yaw, VIEW_PRESETS.top.pitch)).toBe("+Z");
    expect(octant(VIEW_PRESETS.bottom.yaw, VIEW_PRESETS.bottom.pitch)).toBe("−Z");
  });
});

describe("detented zoom", () => {
  const camera = { yaw: 0.75, pitch: 0.32, distance: 4.6, target: [0, 0, 0] as const };

  it("lands on the ladder from an off-ladder start", () => {
    const out = detentZoomCamera({ ...camera }, -100);
    expect(gainOf(out.distance, "orthographic").calibrated).toBe(true);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(500, 6);
  });

  it("walks one rung per notch, in the wheel's direction", () => {
    let out = detentZoomCamera({ ...camera }, 100);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(1000, 6);
    out = detentZoomCamera(out, 100);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(2000, 6);
    out = detentZoomCamera(out, -100);
    expect(gainOf(out.distance, "orthographic").mm).toBeCloseTo(1000, 6);
  });

  it("refuses a step that would be clamped off the ladder", () => {
    const far = detentZoomCamera({ ...camera, distance: MAX_DISTANCE }, 100);
    expect(far.distance).toBe(MAX_DISTANCE);
    const near = detentZoomCamera({ ...camera, distance: MIN_DISTANCE }, -100);
    expect(near.distance).toBe(MIN_DISTANCE);
  });

  it("leaves the rest of the camera alone", () => {
    const out = detentZoomCamera({ ...camera }, 100);
    expect(out.yaw).toBe(camera.yaw);
    expect(out.pitch).toBe(camera.pitch);
    expect(out.target).toBe(camera.target);
  });
});

describe("the floor is quieter than structure", () => {
  /** A token laid on paper at `alpha`, which is what the shader composites. */
  const printed = (tone: "graticule-line" | "graticule-axis", alpha: number) => {
    const ink = hexToRgb(CHROME[tone]);
    const mixed = ink.map((channel, index) => channel * alpha + PAPER[index] * (1 - alpha));
    return contrastRatio(mixed as [number, number, number], PAPER);
  };

  it("prints a minor line below the band structure is held to", () => {
    // A spatial cue, not structure: 1.3–1.4 is where it stops surviving being
    // attended away from, which is the whole requirement.
    const minor = printed("graticule-line", GRID_ALPHA.minor);
    expect(minor).toBeGreaterThan(1.3);
    expect(minor).toBeLessThan(1.4);
  });

  it("keeps the three weights separable and in order", () => {
    const minor = printed("graticule-line", GRID_ALPHA.minor);
    const major = printed("graticule-line", GRID_ALPHA.major);
    const axis = printed("graticule-axis", GRID_ALPHA.axis);
    expect(major).toBeGreaterThan(minor + 0.15);
    expect(axis).toBeGreaterThan(major + 0.05);
    // And the strongest of them is still under the structural floor.
    expect(axis).toBeLessThan(1.75);
  });

  it("steps the whole plane back together when a sketch owns the reference", () => {
    const full = printed("graticule-line", GRID_ALPHA.minor);
    const dimmed = printed("graticule-line", GRID_ALPHA.minor * GRID_ALPHA.offPlane);
    expect(dimmed).toBeLessThan(full);
    // Still visible: the floor keeps saying which way up the world is.
    expect(dimmed).toBeGreaterThan(1.1);
  });

  it("fades outward from the target before the far field can alias", () => {
    expect(GRID_FADE.start).toBeLessThan(GRID_FADE.end);
    // The fade has to begin outside the framed height, or the grid under the
    // part you are looking at is already dissolving.
    expect(GRID_FADE.start).toBeGreaterThan(FOV_SCALE / 2);
    expect(GRID_MAJOR_EVERY).toBe(5);
  });
});

describe("the graticule tones stay furniture", () => {
  it("holds every tone inside the furniture band on paper", () => {
    for (const tone of GRATICULE_TONES) {
      const ratio = contrastRatio(hexToRgb(CHROME[tone]), PAPER);
      // The floor is what survives the render: the framebuffer is smaller
      // than the CSS box under the quality budget, so a hairline is resampled
      // across ~1.25 pixels and loses roughly 0.2 of its ratio in the process.
      expect(ratio, `${tone} on paper`).toBeGreaterThan(1.6);
      // The ceiling is the 3:1 a meaningful mark owes, held well clear: this
      // one is deliberately not meaningful, and past ~2.8 the grid starts to
      // survive being attended away from, which is the one thing it must not do.
      expect(ratio, `${tone} on paper`).toBeLessThan(2.8);
    }
  });

  it("keeps the token tones separable and in order", () => {
    const ratios = GRATICULE_TONES.map((tone) =>
      contrastRatio(hexToRgb(CHROME[tone]), PAPER),
    );
    for (let index = 1; index < ratios.length; index++) {
      expect(ratios[index], GRATICULE_TONES[index]).toBeGreaterThan(ratios[index - 1] + 0.15);
    }
  });

  it("stays quieter than the ink drawn on top of it", () => {
    const ink = contrastRatio(hexToRgb(CHROME["viewport-ink-2"]), PAPER);
    for (const tone of GRATICULE_TONES) {
      expect(contrastRatio(hexToRgb(CHROME[tone]), PAPER), tone).toBeLessThan(ink);
    }
  });
});
