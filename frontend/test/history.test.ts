import { describe, expect, it } from "vitest";
import { HISTORY_LIMIT, SourceHistory } from "../src/history";

describe("SourceHistory", () => {
  it("starts empty with nothing to undo or redo", () => {
    const history = new SourceHistory();
    expect(history.canUndo()).toBe(false);
    expect(history.canRedo()).toBe(false);
    expect(history.undo()).toBeNull();
    expect(history.redo()).toBeNull();
  });

  it("undoes and redoes committed snapshots in order", () => {
    const history = new SourceHistory();
    history.commit("a");
    history.commit("b");
    history.commit("c");

    expect(history.undo()).toBe("b");
    expect(history.undo()).toBe("a");
    expect(history.undo()).toBeNull();
    expect(history.redo()).toBe("b");
    expect(history.redo()).toBe("c");
    expect(history.redo()).toBeNull();
  });

  it("ignores commits that do not change the text", () => {
    const history = new SourceHistory();
    history.commit("a");
    history.commit("a");
    history.commit("a");
    expect(history.canUndo()).toBe(false);

    history.commit("b");
    history.commit("b");
    expect(history.depth).toBe(1);
    expect(history.undo()).toBe("a");
  });

  it("clears the redo branch when a new state is committed after undo", () => {
    const history = new SourceHistory();
    history.commit("a");
    history.commit("b");
    expect(history.undo()).toBe("a");
    expect(history.canRedo()).toBe(true);

    history.commit("c");
    expect(history.canRedo()).toBe(false);
    expect(history.undo()).toBe("a");
  });

  it("drops the oldest snapshots beyond the bound", () => {
    const history = new SourceHistory(3);
    for (const text of ["a", "b", "c", "d", "e"]) history.commit(text);

    expect(history.depth).toBe(3);
    expect(history.undo()).toBe("d");
    expect(history.undo()).toBe("c");
    expect(history.undo()).toBe("b");
    // "a" fell off the bounded end.
    expect(history.undo()).toBeNull();
  });

  it("defaults to a 100-entry bound", () => {
    const history = new SourceHistory();
    for (let index = 0; index < 250; index++) history.commit(`state ${index}`);
    expect(history.depth).toBe(HISTORY_LIMIT);
  });
});
