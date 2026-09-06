/**
 * Pure helpers for the study panel: readable selection summaries and the
 * request bodies for the study patch operations.
 *
 * Studies are declared in the scene program; the panel only ever edits them
 * through /patch source edits, so everything here is a straight mapping from
 * payload shapes to patch-request shapes with no state of its own.
 */

import type {
  StudyBc,
  StudyBcType,
  StudyKind,
  StudyPayload,
  StudyPayloadKind,
  StudySelection,
} from "./types";
import { byId } from "./identity";

/** Compact numeric formatting for selection summaries: drop trailing zeros. */
const num = (value: number): string => {
  const fixed = Math.abs(value) >= 1000 ? value.toPrecision(4) : value.toFixed(2);
  return String(Number(fixed));
};

const vec = (values: number[]): string => `[${values.map(num).join(", ")}]`;

/** One-line human summary of a selection description, composites included. */
export function describeSelection(selection: StudySelection): string {
  switch (selection.kind) {
    case "box":
      return `box ${vec(selection.min_corner)} → ${vec(selection.max_corner)}`;
    case "sphere":
      return `sphere ${vec(selection.center)} r ${num(selection.radius)}`;
    case "halfspace":
      return `halfspace at ${vec(selection.point)} · n ${vec(selection.normal)}`;
    case "side":
      return selection.tol === null
        ? `side ${selection.side}`
        : `side ${selection.side} ± ${num(selection.tol)}`;
    case "predicate":
      return `predicate ${selection.name}()`;
    case "and":
      return selection.operands.map(describeSelection).join(" ∧ ");
    case "or":
      return selection.operands.map(describeSelection).join(" ∨ ");
    case "not":
      return `¬(${describeSelection(selection.operand)})`;
  }
}

export const BC_LABELS: Record<StudyBcType, string> = {
  dirichlet: "Fixed value",
  heat_flux: "Heat flux",
  fixed: "Fixed support",
  traction: "Traction",
  inlet: "Inlet",
  outlet: "Outlet",
  walls: "Duct walls",
  heat_source: "Heat source",
  held_temperature: "Held temperature",
};

/**
 * BC types that make sense for a study kind.
 *
 * A flow study's five differ from the mesh kinds' in a way the add form has
 * to respect: an inlet, an outlet and the duct walls are faces of the
 * lattice, so they place no region and the form must not ask for one. See
 * `bcPlacesRegion`.
 */
export function bcTypesFor(kind: StudyPayloadKind): StudyBcType[] {
  if (kind === "thermal") return ["dirichlet", "heat_flux"];
  if (kind === "elastic") return ["fixed", "traction"];
  if (kind === "flow") return ["inlet", "outlet", "walls", "heat_source", "held_temperature"];
  return [];
}

/** Whether the GUI can add and edit this kind of study's conditions. */
export function isEditableStudyKind(kind: StudyPayloadKind): kind is StudyKind {
  return kind === "thermal" || kind === "elastic" || kind === "flow";
}

/** Whether this condition picks a region, or is a face of the lattice. */
export function bcPlacesRegion(type: StudyBcType): boolean {
  return BC_PLACEMENT[type] === undefined;
}

/**
 * The scalar/vector a BC row shows, or null when it carries no number.
 *
 * `fixed` and `outlet` are the two that state a condition and nothing else.
 * `walls` carries a temperature only when the duct is heated, so an unheated
 * wall is null rather than a misleading zero.
 */
export function bcValue(bc: StudyBc): number | [number, number, number] | null {
  if (bc.type === "dirichlet") return bc.value ?? 0;
  if (bc.type === "heat_flux") return bc.flux ?? 0;
  if (bc.type === "traction") return bc.vector ?? [0, 0, 0];
  if (bc.type === "inlet") return bc.velocity ?? [0, 0, 0];
  if (bc.type === "heat_source") return bc.power ?? 0;
  if (bc.type === "held_temperature") return bc.value ?? 0;
  if (bc.type === "walls") return bc.temperature ?? null;
  return null;
}

/** How a condition that places nothing describes where it acts. */
export const BC_PLACEMENT: Partial<Record<StudyBcType, string>> = {
  inlet: "lattice inlet face",
  outlet: "lattice outlet face",
  walls: "duct walls",
};

/** The numeric constructor arguments a study kind exposes for editing. */
export function studyArguments(study: StudyPayload): { key: string; value: number }[] {
  const rows: { key: string; value: number }[] = [];
  if (typeof study.resolution === "number") {
    rows.push({ key: "resolution", value: study.resolution });
  }
  // A study's material map carries numbers for the properties stated in the
  // declaration and the sentinel string "material" for the ones it defers to
  // the assigned Material. Only the numbers are literals in the source, so
  // only the numbers get an editable row; a deferred property is shown by the
  // material chip on the card instead of by a field that would rewrite it.
  for (const [key, value] of Object.entries(study.material ?? {})) {
    if (typeof value === "number") rows.push({ key, value });
  }
  if (study.kind === "thermal") {
    rows.push({ key: "source", value: typeof study.source === "number" ? study.source : 0 });
  }
  return rows;
}

/** Selection kinds the builder form offers (predicates are code-only). */
export type BuilderSelectionKind = "side" | "box" | "sphere" | "halfspace";

/** Editable state of the add-BC builder form, converted on submit. */
export interface BcDraft {
  bcType: StudyBcType;
  selectionKind: BuilderSelectionKind;
  side: string;
  minCorner: [number, number, number];
  maxCorner: [number, number, number];
  center: [number, number, number];
  radius: number;
  point: [number, number, number];
  normal: [number, number, number];
  value: number;
  vector: [number, number, number];
}

export function defaultDraft(kind: StudyPayloadKind): BcDraft {
  return {
    bcType: bcTypesFor(kind)[0] ?? "fixed",
    selectionKind: "side",
    side: "+x",
    minCorner: [0, 0, 0],
    maxCorner: [1, 1, 1],
    center: [0, 0, 0],
    radius: 0.5,
    point: [0, 0, 0],
    normal: [0, 0, 1],
    value: kind === "thermal" ? 100 : kind === "flow" ? 0.02 : 0,
    vector: [0, 0, -1],
  };
}

export function draftSelection(draft: BcDraft): StudySelection {
  switch (draft.selectionKind) {
    case "side":
      return { kind: "side", side: draft.side, tol: null };
    case "box":
      return { kind: "box", min_corner: [...draft.minCorner], max_corner: [...draft.maxCorner] };
    case "sphere":
      return { kind: "sphere", center: [...draft.center], radius: draft.radius };
    case "halfspace":
      return { kind: "halfspace", point: [...draft.point], normal: [...draft.normal] };
  }
}

/** Body for POST /patch adding the drafted BC (App prepends `source`). */
export function addBcRequest(study: StudyPayload, draft: BcDraft): Record<string, unknown> {
  const body: Record<string, unknown> = {
    op: "add_study_bc",
    ...byId(study),
    study: study.index,
    bc_type: draft.bcType,
  };
  // A condition that places nothing must not send a selection: the server
  // refuses the pair rather than quietly ignoring half of it.
  if (bcPlacesRegion(draft.bcType)) body.selection = draftSelection(draft);
  // `value` is required for valued conditions and forbidden for the two that
  // state themselves — a fixed support and an outlet.
  if (draft.bcType === "traction") body.value = [...draft.vector];
  else if (draft.bcType !== "fixed" && draft.bcType !== "outlet") body.value = draft.value;
  return body;
}

export function addStudyRequest(kind: StudyKind): Record<string, unknown> {
  return { op: "add_study", kind };
}

export function deleteStudyRequest(study: StudyPayload): Record<string, unknown> {
  return { op: "delete_study", ...byId(study), study: study.index };
}

export function deleteBcRequest(study: StudyPayload, bc: number): Record<string, unknown> {
  return { op: "delete_study_bc", ...byId(study), study: study.index, bc };
}

export function setBcValueRequest(
  study: StudyPayload,
  bc: number,
  value: number | number[],
): Record<string, unknown> {
  return { op: "set_study_value", ...byId(study), study: study.index, bc, value };
}

export function setArgumentRequest(
  study: StudyPayload,
  argument: string,
  value: number,
): Record<string, unknown> {
  return { op: "set_study_value", ...byId(study), study: study.index, argument, value };
}
