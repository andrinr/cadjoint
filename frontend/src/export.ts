/**
 * File → Export…, the part with no DOM in it.
 *
 * The dialog is a thin form over `POST /api/export`; everything it has to
 * *decide* — which formats there are and what each one takes, which names in
 * the program are worth offering, how a request body is shaped, what a
 * download is called — is here, as plain functions a unit test can call.
 * `api.ts` does the fetch and `components/ExportDialog.tsx` draws the form;
 * neither computes anything this module does not.
 */

import type { ExportFormat, ExportRequest } from "./types";

/**
 * The lattice budget, mirrored from `cadjoint/viewer/_limits.py`.
 *
 * The server refuses a request outside the bracket with a message the dialog
 * shows, so these only have to keep the control honest — a number box whose
 * range says the same thing the validator does.
 */
export const EXPORT_RESOLUTION = { min: 8, max: 256, default: 64 } as const;

/** What a format takes: an SDF variable of the program, or a declared study. */
export type ExportTarget = "object" | "study";

export interface ExportFormatInfo {
  value: ExportFormat;
  /** The label on the picker — the format's own name, as CAD tools write it. */
  label: string;
  extension: string;
  takes: ExportTarget;
  /** One sentence under the picker saying what the file will contain. */
  note: string;
  /** The one option the format has, if it has one. */
  option: { key: "binary" | "analytic" | "merge_planar"; label: string } | null;
}

/**
 * Every format the server writes, in the order the picker lists them.
 *
 * STEP's note says what makes it the interesting one: it is not a faceted
 * mesh with a `.step` extension but the derived B-rep — exact planes and
 * cylinders with their real boundary curves — faceted only where a face is
 * a blend the graph cannot certify (`cadjoint/viewer/_export.py`).
 */
export const EXPORT_FORMATS: readonly ExportFormatInfo[] = [
  {
    value: "stl",
    label: "STL",
    extension: "stl",
    takes: "object",
    note: "Triangles of the dual-contour surface. Binary by default; ASCII on request.",
    option: { key: "binary", label: "Binary" },
  },
  {
    value: "obj",
    label: "OBJ",
    extension: "obj",
    takes: "object",
    note: "Wavefront OBJ. Flat regions become single n-gon faces; curved regions stay triangles.",
    option: { key: "merge_planar", label: "Merge planar faces" },
  },
  {
    value: "step",
    label: "STEP",
    extension: "step",
    takes: "object",
    note: "AP214 from the derived B-rep: exact planes and cylinders with their real edge curves; blends faceted.",
    option: { key: "analytic", label: "Analytic surfaces" },
  },
  {
    value: "vtk",
    label: "VTK",
    extension: "vtk",
    takes: "study",
    note: "A declared study's solved mesh and fields, for ParaView. Solves the study if the program has not.",
    option: null,
  },
];

/** The picker entry for a format. */
export function formatInfo(format: ExportFormat): ExportFormatInfo {
  const info = EXPORT_FORMATS.find((entry) => entry.value === format);
  if (!info) throw new Error(`Unknown export format ${format}.`);
  return info;
}

/**
 * Constructors whose result is never an SDF, so a variable bound to one is
 * not worth offering as an export target. Everything else — a primitive, a
 * boolean, a transform, an extrusion, a `Solid.box(...)` — might be.
 */
const NOT_GEOMETRY =
  /^(?:Scalar|Vector|Vector2|Material|SimMesh|ThermalStudy|ElasticStudy|Optimization|PolygonProfile|SketchPlane|jnp|np|jax)\b/;

/** A right-hand side that is plainly a literal, not a call. */
const LITERAL = /^(?:[-+]?\d|["'[({]|True\b|False\b|None\b)/;

/**
 * The names the program binds at module level that could be geometry.
 *
 * Read off the text, not the payload: the compile payload names features the
 * viewer can outline, and the thing a user most often wants to export — the
 * `Union(...)` a whole assembly is bound to — is not one of those. A
 * left-hand identifier on an unindented line is cheap to find and honest
 * about what it is: a suggestion. The server checks the real object and, on
 * a miss, answers with the names that would have worked.
 *
 * `scene` leads (it is the default and the one that always exists); the rest
 * keep source order. Private names, loop variables and literals are skipped.
 */
export function candidateObjects(source: string): string[] {
  const names: string[] = [];
  for (const line of source.split("\n")) {
    const match = /^([A-Za-z][A-Za-z0-9_]*)\s*(?::\s*[^=]+)?=(?!=)\s*(.*)$/.exec(line);
    if (!match) continue;
    const [, name, rhs] = match;
    if (NOT_GEOMETRY.test(rhs) || LITERAL.test(rhs)) continue;
    if (!names.includes(name)) names.push(name);
  }
  return ["scene", ...names.filter((name) => name !== "scene")];
}

/** What the name picker offers for a format: objects, or declared studies. */
export function exportTargets(
  format: ExportFormat,
  source: string,
  studyNames: readonly string[],
): string[] {
  return formatInfo(format).takes === "study" ? [...studyNames] : candidateObjects(source);
}

/** The name a fresh dialog starts on for a format. */
export function defaultExportName(
  format: ExportFormat,
  studyNames: readonly string[],
): string {
  return formatInfo(format).takes === "study" ? (studyNames[0] ?? "") : "scene";
}

export interface ExportOptions {
  format: ExportFormat;
  name: string;
  resolution: number;
  binary: boolean;
  analytic: boolean;
  mergePlanar: boolean;
}

/** A resolution the server will accept: an integer inside the bracket. */
export function clampResolution(value: number): number {
  if (!Number.isFinite(value)) return EXPORT_RESOLUTION.default;
  return Math.min(EXPORT_RESOLUTION.max, Math.max(EXPORT_RESOLUTION.min, Math.round(value)));
}

/**
 * The request body for one export.
 *
 * Only the option the format actually reads is sent — the server accepts all
 * three on any format, but a `binary: false` riding along on a STEP request
 * is the kind of thing that reads as a bug in a job's field list.
 */
export function exportRequest(source: string, options: ExportOptions): ExportRequest {
  const info = formatInfo(options.format);
  const body: ExportRequest = {
    source,
    format: options.format,
    name: options.name.trim(),
  };
  if (info.takes === "object") body.resolution = clampResolution(options.resolution);
  if (info.option?.key === "binary") body.binary = options.binary;
  if (info.option?.key === "analytic") body.analytic = options.analytic;
  if (info.option?.key === "merge_planar") body.merge_planar = options.mergePlanar;
  return body;
}

/**
 * The file name a `Content-Disposition` header carries, or the fallback.
 *
 * Handles the quoted plain form the server sends and the RFC 5987 `filename*`
 * form a proxy might rewrite it into.
 */
export function attachmentFilename(header: string | null | undefined, fallback: string): string {
  if (!header) return fallback;
  const extended = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/.exec(header);
  if (extended) {
    try {
      return decodeURIComponent(extended[1].trim()) || fallback;
    } catch {
      return fallback;
    }
  }
  const quoted = /filename\s*=\s*"([^"]*)"/.exec(header);
  if (quoted) return quoted[1] || fallback;
  const bare = /filename\s*=\s*([^;\s]+)/.exec(header);
  return bare ? bare[1] : fallback;
}

/**
 * The download's name: the scene's stem, then the object's, then the extension.
 *
 * `heatsink-scene.stl` says which program the part came from; `scene.stl`
 * on its own does not, and an unsaved buffer has nothing to say, so it
 * keeps the server's name.
 */
export function downloadName(
  serverFilename: string,
  sceneName: string | null | undefined,
): string {
  if (!sceneName) return serverFilename;
  const stem = sceneName.replace(/\.py$/i, "").trim();
  return stem ? `${stem}-${serverFilename}` : serverFilename;
}

/**
 * A worker failure, in one line.
 *
 * The worker answers an exception with its whole traceback — right for the
 * console, wrong for a dialog. The last non-empty line of a traceback is the
 * exception itself, which is the sentence a user can act on.
 */
export function errorSummary(text: string): string {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length === 0) return "Export failed.";
  if (!/^Traceback \(most recent call last\)/.test(lines[0])) return lines.join(" ");
  return lines[lines.length - 1];
}
