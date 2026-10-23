/**
 * File → Export…, the decisions under the dialog.
 *
 * What a format takes, which names the program offers, how a request body
 * is shaped for each format, what the download is called, and how a
 * worker's traceback is reduced to the one line a dialog can show.
 */

import { describe, expect, it } from "vitest";
import {
  EXPORT_FORMATS,
  EXPORT_RESOLUTION,
  attachmentFilename,
  candidateObjects,
  clampResolution,
  defaultExportName,
  downloadName,
  errorSummary,
  exportRequest,
  exportTargets,
  formatInfo,
  stepContentsLine,
} from "../src/export";

const PROGRAM = `"""A part."""

import jax.numpy as jnp

from cadjoint.construction import PolygonProfile, Solid, extrude
from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
from cadjoint.geometry import Scalar, Vector
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union

fin_depth = Scalar(1.2, free=True, name="fin_depth")
size = Vector([1.0, 0.5, 0.5])
aluminum = Material(name="aluminum", color=[0.8, 0.82, 0.85])
count = 3
label = "deck"
comb = PolygonProfile([[0, 0], [1, 0], [1, 1]], name="comb")
fins = extrude(comb, depth=fin_depth)
deck = Solid.box(size=[1, 1, 0.2], position=[0, 0, 0], name="deck")
    inner = Union(fins, deck)
scene = Union(fins, deck).with_material(aluminum)
heat = ThermalStudy(name="sink", resolution=12, conductivity=1.0, bcs=[])
if count == 3:
    pass
scene == deck
`;

describe("the format table", () => {
  it("lists every server format once, geometry first and the result last", () => {
    expect(EXPORT_FORMATS.map((entry) => entry.value)).toEqual(["stl", "obj", "step", "vtk"]);
    expect(EXPORT_FORMATS.filter((entry) => entry.takes === "study").map((e) => e.value)).toEqual([
      "vtk",
    ]);
  });

  it("names the one option each geometry format has", () => {
    expect(formatInfo("stl").option?.key).toBe("binary");
    expect(formatInfo("obj").option?.key).toBe("merge_planar");
    expect(formatInfo("step").option?.key).toBe("analytic");
    expect(formatInfo("vtk").option).toBeNull();
  });

  it("refuses a format the server does not write", () => {
    expect(() => formatInfo("gltf" as never)).toThrow(/gltf/);
  });
});

describe("candidateObjects", () => {
  it("offers scene first, then the module-level names that could be geometry", () => {
    expect(candidateObjects(PROGRAM)).toEqual(["scene", "fins", "deck"]);
  });

  it("skips parameters, materials, studies, profiles and literals", () => {
    const names = candidateObjects(PROGRAM);
    for (const skipped of ["fin_depth", "size", "aluminum", "count", "label", "comb", "heat"]) {
      expect(names).not.toContain(skipped);
    }
  });

  it("skips indented bindings and comparisons, and never repeats a name", () => {
    expect(candidateObjects(PROGRAM)).not.toContain("inner");
    expect(candidateObjects("scene = a()\nscene = b()\nother = c()\n")).toEqual([
      "scene",
      "other",
    ]);
  });

  it("always offers scene, even for an empty program", () => {
    expect(candidateObjects("")).toEqual(["scene"]);
  });
});

describe("exportTargets and the default name", () => {
  it("offers objects for geometry formats and studies for vtk", () => {
    expect(exportTargets("stl", PROGRAM, ["sink"])).toEqual(["scene", "fins", "deck"]);
    expect(exportTargets("vtk", PROGRAM, ["sink", "load"])).toEqual(["sink", "load"]);
  });

  it("starts on scene, or on the first study, or on nothing", () => {
    expect(defaultExportName("step", ["sink"])).toBe("scene");
    expect(defaultExportName("vtk", ["sink", "load"])).toBe("sink");
    expect(defaultExportName("vtk", [])).toBe("");
  });
});

describe("exportRequest", () => {
  const options = {
    format: "stl" as const,
    name: " scene ",
    resolution: 64,
    binary: false,
    analytic: false,
    mergePlanar: false,
  };

  it("sends only the option the format reads", () => {
    expect(exportRequest("src", options)).toEqual({
      source: "src",
      format: "stl",
      name: "scene",
      resolution: 64,
      binary: false,
    });
    expect(exportRequest("src", { ...options, format: "step" })).toEqual({
      source: "src",
      format: "step",
      name: "scene",
      resolution: 64,
      analytic: false,
    });
    expect(exportRequest("src", { ...options, format: "obj" })).toEqual({
      source: "src",
      format: "obj",
      name: "scene",
      resolution: 64,
      merge_planar: false,
    });
  });

  it("sends no lattice at all for a result export", () => {
    expect(exportRequest("src", { ...options, format: "vtk", name: "sink" })).toEqual({
      source: "src",
      format: "vtk",
      name: "sink",
    });
  });

  it("keeps the resolution inside the server's bracket, as an integer", () => {
    expect(clampResolution(3)).toBe(EXPORT_RESOLUTION.min);
    expect(clampResolution(10_000)).toBe(EXPORT_RESOLUTION.max);
    expect(clampResolution(47.6)).toBe(48);
    expect(clampResolution(Number.NaN)).toBe(EXPORT_RESOLUTION.default);
    expect(exportRequest("src", { ...options, resolution: 1e9 }).resolution).toBe(
      EXPORT_RESOLUTION.max,
    );
  });
});

describe("file names", () => {
  it("reads the quoted, bare and RFC 5987 forms of Content-Disposition", () => {
    expect(attachmentFilename('attachment; filename="scene.stl"', "x")).toBe("scene.stl");
    expect(attachmentFilename("attachment; filename=scene.obj", "x")).toBe("scene.obj");
    expect(attachmentFilename("attachment; filename*=UTF-8''sc%C3%A8ne.step", "x")).toBe(
      "scène.step",
    );
    expect(attachmentFilename("inline", "fallback.stl")).toBe("fallback.stl");
    expect(attachmentFilename(null, "fallback.stl")).toBe("fallback.stl");
  });

  it("prefixes the download with the scene's stem when there is one", () => {
    expect(downloadName("scene.stl", "heatsink.py")).toBe("heatsink-scene.stl");
    expect(downloadName("scene.stl", "bracket")).toBe("bracket-scene.stl");
    expect(downloadName("scene.stl", null)).toBe("scene.stl");
    expect(downloadName("scene.stl", ".py")).toBe("scene.stl");
  });
});

describe("errorSummary", () => {
  it("reduces a worker traceback to its exception line", () => {
    const traceback = [
      "Traceback (most recent call last):",
      '  File "<playground>", line 3, in <module>',
      "    raise ValueError('no')",
      "ValueError: The program binds no SDF object named 'nope' (exportable: 'scene').",
      "",
    ].join("\n");
    expect(errorSummary(traceback)).toBe(
      "ValueError: The program binds no SDF object named 'nope' (exportable: 'scene').",
    );
  });

  it("passes a plain message through, joined onto one line", () => {
    expect(errorSummary("Export `resolution`: Input should be less than or equal to 256.")).toBe(
      "Export `resolution`: Input should be less than or equal to 256.",
    );
    expect(errorSummary("one\n\ntwo\n")).toBe("one two");
    expect(errorSummary("  \n")).toBe("Export failed.");
  });
});

describe("stepContentsLine", () => {
  it("names analytic, spline and faceted faces, with the bar the splines meet", () => {
    expect(
      stepContentsLine({
        faces: 30,
        surfaces: { plane: 20, cylinder: 2, bspline: 5, faceted: 3 },
        fit_tolerance: 1.04e-4,
      }),
    ).toBe("30 faces: 22 analytic · 5 spline (≤ 1.0e−4) · 3 faceted");
  });

  it("says zero rather than leaving a kind out, and drops the bar without a spline", () => {
    expect(stepContentsLine({ faces: 7, surfaces: { plane: 6, cylinder: 1 }, fit_tolerance: 1e-4 })).toBe(
      "7 faces: 7 analytic · 0 spline · 0 faceted",
    );
  });

  it("counts a face the lattice path dropped", () => {
    expect(stepContentsLine({ faces: 3, surfaces: { plane: 2, dropped: 1 } })).toBe(
      "3 faces: 2 analytic · 0 spline · 0 faceted · 1 dropped",
    );
  });
});
