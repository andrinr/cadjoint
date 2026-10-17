/**
 * What the last removal took out, for the status line's "removed … · Undo"
 * (guide F7). Named in the program's words from the state before the patch.
 */

export interface Named {
  name?: string | null;
  index?: number;
  id?: string | null;
  line?: number | null;
  stableId?: string | null;
  kind?: string;
}

export interface RemovalLookups {
  nodes: readonly Named[];
  studies: readonly Named[];
  meshes: readonly Named[];
  optimizations: readonly Named[];
}

const byAddress = (list: readonly Named[], body: Record<string, unknown>, indexKey: string) =>
  list.find(
    (item) =>
      (body.id !== undefined && (item.id === body.id || item.stableId === body.id)) ||
      (body[indexKey] !== undefined && item.index === body[indexKey]) ||
      (body.line !== undefined && item.line === body.line),
  );

/** The sentence for a removal patch, or null when the patch removes nothing. */
export function removalLabel(body: Record<string, unknown>, lookups: RemovalLookups): string | null {
  switch (body.op) {
    case "delete_object": {
      const node = byAddress(lookups.nodes, body, "index");
      return `removed ${node?.name ?? node?.kind ?? "the object"}`;
    }
    case "delete_constraint":
      return "removed a constraint";
    case "delete_study": {
      const study = byAddress(lookups.studies, body, "study");
      return `removed ${study?.name ?? "the study"}`;
    }
    case "delete_study_bc":
      return "removed a boundary condition";
    case "delete_mesh": {
      const mesh = byAddress(lookups.meshes, body, "mesh");
      return `removed ${mesh?.name ?? "the mesh"}`;
    }
    case "delete_optimization": {
      const run = byAddress(lookups.optimizations, body, "optimization");
      return `removed ${run?.name ?? "the optimization"}`;
    }
    default:
      return null;
  }
}
