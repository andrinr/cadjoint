/**
 * Layout persistence.
 *
 * A stored arrangement is a convenience, never a correctness requirement, so
 * what is asserted here is mostly what happens when the record is wrong:
 * truncated, from another version, or naming a window this build dropped. In
 * every one of those cases the answer is the same — fall back to the defaults
 * and keep the dock working.
 */

import { describe, expect, it } from "vitest";
import {
  clearWorkspace,
  emptyWorkspace,
  isRestorableLayout,
  layoutPanelIds,
  normaliseWorkspace,
  parseWorkspace,
  readWorkspace,
  serializeWorkspace,
  storedWindowsForMode,
  WORKSPACE_STORAGE_KEY,
  writeWorkspace,
  type Workspace,
  type WorkspaceStorage,
} from "../src/windows/workspace";
import { defaultModeWindows } from "../src/windows/windowState";

function memoryStorage(): WorkspaceStorage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
    removeItem: (key) => void data.delete(key),
  };
}

const layoutOf = (...ids: string[]) => ({
  grid: { root: {}, width: 100, height: 100, orientation: "HORIZONTAL" },
  panels: Object.fromEntries(ids.map((id) => [id, { id, contentComponent: id }])),
});

describe("layout inspection", () => {
  it("reads the window ids out of a stored grid", () => {
    expect(layoutPanelIds(layoutOf("viewport", "editor")).sort()).toEqual(["editor", "viewport"]);
    expect(layoutPanelIds(undefined)).toEqual([]);
    expect(layoutPanelIds({} as never)).toEqual([]);
  });

  it("refuses a grid naming a window this build does not have", () => {
    expect(isRestorableLayout(layoutOf("viewport", "editor"))).toBe(true);
    expect(isRestorableLayout(layoutOf("viewport", "console"))).toBe(false);
    expect(isRestorableLayout(layoutOf())).toBe(false);
  });
});

describe("parsing", () => {
  it("survives every shape of junk", () => {
    for (const raw of [null, "", "not json", "[]", "3", '{"version":"x"}']) {
      expect(parseWorkspace(raw).windows.model).toEqual(defaultModeWindows("model"));
      expect(parseWorkspace(raw).layouts.model).toBeUndefined();
    }
  });

  it("discards a record written by another version", () => {
    const future = JSON.stringify({
      version: 99,
      layouts: { model: layoutOf("viewport") },
      windows: { model: { ...defaultModeWindows("model"), objects: "minimised" } },
    });
    expect(parseWorkspace(future).windows.model.objects).toBe("open");
    expect(parseWorkspace(future).layouts.model).toBeUndefined();
  });

  it("drops one mode's unreadable grid without touching the others", () => {
    const workspace = emptyWorkspace();
    workspace.layouts.model = layoutOf("viewport", "editor");
    workspace.layouts.sketch = layoutOf("viewport", "gone") as never;
    const round = normaliseWorkspace(workspace);
    expect(round.layouts.model).toBeDefined();
    expect(round.layouts.sketch).toBeUndefined();
  });

  it("keeps only statuses it recognises", () => {
    const raw = JSON.stringify({
      version: 1,
      layouts: {},
      windows: { model: { objects: "minimised", materials: "banana", console: "open" } },
    });
    const parsed = parseWorkspace(raw);
    expect(parsed.windows.model.objects).toBe("minimised");
    expect(parsed.windows.model.materials).toBe("open");
    expect("console" in parsed.windows.model).toBe(false);
  });
});

describe("round trips", () => {
  it("preserves a good record exactly", () => {
    const workspace: Workspace = emptyWorkspace();
    workspace.layouts.model = layoutOf("viewport", "editor", "objects");
    workspace.windows.model = { ...defaultModeWindows("model"), materials: "minimised" };
    const round = parseWorkspace(serializeWorkspace(workspace));
    expect(round).toEqual(workspace);
    expect(storedWindowsForMode(round, "model").sort()).toEqual([
      "editor",
      "objects",
      "viewport",
    ]);
  });

  it("reads back what it wrote through storage", () => {
    const storage = memoryStorage();
    const workspace = emptyWorkspace();
    workspace.layouts.simulate = layoutOf("viewport", "studies");
    writeWorkspace(storage, workspace);
    expect(storage.data.has(WORKSPACE_STORAGE_KEY)).toBe(true);
    expect(readWorkspace(storage).layouts.simulate).toBeDefined();
    clearWorkspace(storage);
    expect(readWorkspace(storage).layouts.simulate).toBeUndefined();
  });

  it("treats a storage that throws as no storage at all", () => {
    const hostile: WorkspaceStorage = {
      getItem: () => {
        throw new Error("private mode");
      },
      setItem: () => {
        throw new Error("quota");
      },
      removeItem: () => {
        throw new Error("quota");
      },
    };
    expect(readWorkspace(hostile)).toEqual(emptyWorkspace());
    expect(() => writeWorkspace(hostile, emptyWorkspace())).not.toThrow();
    expect(() => clearWorkspace(hostile)).not.toThrow();
    expect(readWorkspace(undefined)).toEqual(emptyWorkspace());
  });
});
