import { describe, expect, it } from "vitest";
import { FEATURES, MENU_ORDER } from "../src/design/features";
import { MENU_COMMANDS, menuModel, modeReason, unboundMenuPaths } from "../src/shell/menuModel";
import { SHORTCUTS, matchShortcut, shortcutLabel, type KeyChord } from "../src/shortcuts";
import { WINDOW_DEFS } from "../src/windows/panels";

const chord = (key: string, mods: Partial<KeyChord> = {}): KeyChord => ({
  key,
  shiftKey: false,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  ...mods,
});

describe("the menu bar is the agreed one", () => {
  it("has the seven menus in order", () => {
    expect(menuModel().map((menu) => menu.name)).toEqual(MENU_ORDER);
  });

  it("binds every leaf of the feature index to a command", () => {
    expect(unboundMenuPaths()).toEqual([]);
  });

  it("binds no command twice", () => {
    const commands = Object.values(MENU_COMMANDS);
    expect(new Set(commands).size).toBe(commands.length);
  });

  it("lists the windows by their tab names", () => {
    const window = menuModel().find((menu) => menu.name === "Window")!;
    const windows = window.groups.find((group) => group.title === "Windows")!;
    expect(windows.entries.map((entry) => entry.label).sort()).toEqual(
      WINDOW_DEFS.filter((def) => !def.permanent)
        .map((def) => def.title)
        .sort(),
    );
    expect(windows.entries.map((entry) => entry.testid)).toContain("menu-window-scenes");
  });

  it("quotes the key for every feature the index gives one", () => {
    const entries = menuModel().flatMap((menu) => menu.groups.flatMap((group) => group.entries));
    const missing = FEATURES.filter((feature) => feature.rule.menu && feature.rule.key)
      .filter((feature) => !/cycles|chip/.test(feature.rule.key!))
      .filter((feature) => {
        const leaf = feature.rule.menu!.split(" › ").at(-1)!.split(/,\s*/)[0];
        return !entries.find((entry) => entry.label === leaf)?.key;
      })
      .map((feature) => feature.name);
    expect(missing).toEqual([]);
  });

  it("explains a mode-scoped item", () => {
    expect(modeReason(["sketch"], "model")).toBe("Only in sketch mode");
    expect(modeReason(["model", "sketch"], "simulate")).toBe("Only in model or sketch mode");
    expect(modeReason(["sketch"], "sketch")).toBeNull();
    expect(modeReason(undefined, "simulate")).toBeNull();
  });
});

describe("the shortcut table", () => {
  it("binds each chord once", () => {
    const seen = new Set<string>();
    for (const shortcut of SHORTCUTS.filter((item) => item.dispatch !== false)) {
      const id = `${shortcut.mod ? "mod+" : ""}${shortcut.shift === true ? "shift+" : ""}${shortcut.key}`;
      expect(seen.has(id), id).toBe(false);
      seen.add(id);
    }
  });

  it("tells C from ⇧C, and ⌘S from ⇧⌘S", () => {
    expect(matchShortcut(chord("c"), "viewport")?.command).toBe("tool-cylinder");
    expect(matchShortcut(chord("C", { shiftKey: true }), "viewport")?.command).toBe(
      "show-construction",
    );
    expect(matchShortcut(chord("s", { metaKey: true }), "app")?.command).toBe("file-save");
    expect(matchShortcut(chord("s", { ctrlKey: true, shiftKey: true }), "app")?.command).toBe(
      "file-save-as",
    );
    // A chord is not a viewport key, and a bare key is not a chord.
    expect(matchShortcut(chord("s", { metaKey: true }), "viewport")).toBeUndefined();
    expect(matchShortcut(chord("s"), "app")).toBeUndefined();
  });

  it("binds the settled keys", () => {
    expect(matchShortcut(chord("?", { shiftKey: true }), "viewport")?.command).toBe(
      "help-shortcuts",
    );
    expect(matchShortcut(chord("["), "viewport")?.command).toBe("grid-finer");
    expect(matchShortcut(chord("]"), "viewport")?.command).toBe("grid-coarser");
    expect(matchShortcut(chord("t"), "viewport")?.command).toBe("gizmo-scale");
    expect(matchShortcut(chord("k"), "viewport")?.command).toBe("tool-sketch");
    expect(matchShortcut(chord("e"), "viewport")?.command).toBe("shape-extrude");
    expect(matchShortcut(chord(".", { metaKey: true }), "app")?.command).toBe("program-stop");
    expect(matchShortcut(chord("e", { metaKey: true }), "app")?.command).toBe("file-export");
    expect(matchShortcut(chord("o", { metaKey: true }), "app")?.command).toBe("file-open");
    expect(
      matchShortcut(chord("Enter", { metaKey: true, shiftKey: true }), "app")?.command,
    ).toBe("sim-solve");
    expect(shortcutLabel("sim-solve")).toBe("⇧⌘↵");
  });

  it("leaves Alt combinations alone", () => {
    expect(matchShortcut(chord("b", { altKey: true }), "viewport")).toBeUndefined();
  });
});
