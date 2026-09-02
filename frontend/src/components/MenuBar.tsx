/**
 * Slim application menu bar: File / Edit / Window / Help.
 *
 * Menus follow the native pattern — click (or ArrowDown) opens, arrows move,
 * Escape or a click elsewhere closes. Everything acts on the shared state
 * signals; file operations go through the server's `scenes` workspace, and
 * Export… through `/api/export`.
 */

import { For, Show, createSignal, onCleanup } from "solid-js";
import * as api from "../api";
import { sanitizeSceneName } from "../scenes";
import { SHORTCUT_GROUPS } from "../shortcuts";
import {
  dirty,
  editingMode,
  panels,
  sceneName,
  setEditingMode,
  setPanelVisible,
  setSceneName,
  setStatus,
  source,
  type PanelVisibility,
} from "../state";
import { EDITING_MODES } from "../editingMode";
import { windowManager } from "../windows/manager";
import type { WindowId } from "../windows/panels";
import { ExportDialog } from "./ExportDialog";

export interface MenuBarProps {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  /** Reset to the starter example (App confirms when the buffer is dirty). */
  onNew: () => void;
  /** Adopt a loaded scene file as the current program and recompile. */
  onAdoptScene: (name: string, source: string) => void;
}

type MenuId = "file" | "edit" | "window" | "help";
type DialogId = "saveAs" | "export" | "help";

const PANEL_ITEMS: { key: keyof PanelVisibility; label: string }[] = [
  { key: "editor", label: "Editor" },
  { key: "objectTree", label: "Object tree" },
  { key: "materials", label: "Materials" },
  { key: "sketch", label: "Sketch panel" },
];

/**
 * The windows the coarse `panels()` record does not cover.
 *
 * `PANEL_ITEMS` above is the older, four-key vocabulary the shell still
 * mirrors; everything else is addressed by window id straight through the
 * dock's manager. That is the whole list of what the Window menu can open:
 * the four simulation windows (which used to be tabs inside one panel and so
 * had no way in at all), the scene browser and the process monitor — the two
 * that are parked in every desk rather than docked in any.
 */
const WINDOW_ITEMS: { id: WindowId; label: string }[] = [
  { id: "meshes", label: "Meshes" },
  { id: "studies", label: "Studies" },
  { id: "optimize", label: "Optimize" },
  { id: "results", label: "Results" },
  { id: "scenes", label: "Scenes" },
  { id: "processes", label: "Processes" },
];

const REPO_README_URL = "https://github.com/andrinr/cadjoint#readme";

export function MenuBar(props: MenuBarProps) {
  const [openMenu, setOpenMenu] = createSignal<MenuId | null>(null);
  const [dialog, setDialog] = createSignal<DialogId | null>(null);
  const [saveAsName, setSaveAsName] = createSignal("");
  const [saveAsError, setSaveAsError] = createSignal("");

  const closeOnOutside = (event: MouseEvent) => {
    if (!(event.target as HTMLElement).closest(".menu-bar")) setOpenMenu(null);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

  // Escape closes an open dialog (or a menu opened without keyboard focus).
  // Capture phase, so the viewer's global Escape handling (clear selection,
  // reset editing mode) does not also fire behind the dialog.
  const closeOnEscape = (event: KeyboardEvent) => {
    if (event.key !== "Escape") return;
    if (dialog() !== null) {
      event.stopPropagation();
      setDialog(null);
    } else if (openMenu() !== null) {
      event.stopPropagation();
      // Keep the keyboard flow: focus returns to the menu's title button.
      document.querySelector<HTMLElement>(".menu-title.active")?.focus();
      setOpenMenu(null);
    }
  };
  document.addEventListener("keydown", closeOnEscape, true);
  onCleanup(() => document.removeEventListener("keydown", closeOnEscape, true));

  const toggleMenu = (id: MenuId) => setOpenMenu(openMenu() === id ? null : id);
  /** An open menubar behaves like hover navigation between its menus. */
  const glideTo = (id: MenuId) => {
    if (openMenu() !== null && openMenu() !== id) setOpenMenu(id);
  };

  /**
   * Open… is the scene browser now, not a list of file names.
   *
   * A modal list of names could only answer "which file", and the question a
   * user actually has in front of a directory of parts is "which *part*".
   * The Scenes window answers that — a picture, the docstring, what each file
   * declares — so Open… raises it rather than opening a smaller, worse
   * version of it in a dialog. It is parked in every desk, so this is
   * usually one click and no layout change.
   */
  const showScenes = () => {
    setOpenMenu(null);
    windowManager()?.open("scenes");
  };

  const saveAs = async (rawName: string): Promise<void> => {
    const name = sanitizeSceneName(rawName);
    if (name === null) {
      setSaveAsError("Use a plain file name such as bracket.py — no folders.");
      return;
    }
    try {
      const result = await api.saveScene(name, source());
      if (!result.ok) {
        const message = result.error ?? `Could not save ${name}.`;
        setSaveAsError(message);
        setStatus({ kind: "error", text: message });
        return;
      }
      setSceneName(name);
      setDialog(null);
      setStatus({ kind: "ready", text: `Saved ${name}` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSaveAsError(message);
      setStatus({ kind: "error", text: message });
    }
  };

  const save = () => {
    setOpenMenu(null);
    const name = sceneName();
    if (name === null) {
      showSaveAsDialog();
      return;
    }
    void saveAs(name);
  };

  const showSaveAsDialog = () => {
    setOpenMenu(null);
    setSaveAsName(sceneName() ?? "scene.py");
    setSaveAsError("");
    setDialog("saveAs");
  };

  /**
   * Export… is a dialog, not a window: it asks four things and produces one
   * file, and a form that is answered and dismissed has no business in the
   * dock. The work it starts is a job, so it is watched — and cancelled —
   * from the chip and the Processes window like a solve.
   */
  const showExportDialog = () => {
    setOpenMenu(null);
    setDialog("export");
  };

  const download = () => {
    setOpenMenu(null);
    const blob = new Blob([source()], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = sceneName() ?? "scene.py";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  /** Arrow-key navigation between the items of an open dropdown. */
  const onMenuKeyDown = (event: KeyboardEvent) => {
    const menu = (event.currentTarget as HTMLElement).closest(".menu");
    if (!menu) return;
    const items = [
      ...menu.querySelectorAll<HTMLElement>(
        ".menu-dropdown button:not(:disabled), .menu-dropdown a",
      ),
    ];
    const index = items.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      (items[(index + 1) % items.length] ?? items[0])?.focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      (items[(index - 1 + items.length) % items.length] ?? items.at(-1))?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setOpenMenu(null);
      menu.querySelector<HTMLElement>(".menu-title")?.focus();
    } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      const order: MenuId[] = ["file", "edit", "window", "help"];
      const active = openMenu();
      if (!active) return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = order[(order.indexOf(active) + step + order.length) % order.length];
      setOpenMenu(next);
      document
        .querySelector<HTMLElement>(`[data-testid=menu-${next}]`)
        ?.focus();
    }
  };

  const menuTitle = (id: MenuId, label: string) => (
    <button
      type="button"
      class="menu-title"
      classList={{ active: openMenu() === id }}
      aria-haspopup="menu"
      aria-expanded={openMenu() === id}
      onClick={() => toggleMenu(id)}
      onPointerEnter={() => glideTo(id)}
      onKeyDown={(event) => {
        if (event.key === "ArrowDown" && openMenu() !== id) {
          event.preventDefault();
          setOpenMenu(id);
        } else {
          onMenuKeyDown(event);
        }
      }}
      data-testid={`menu-${id}`}
    >
      {label}
    </button>
  );

  const item = (
    label: string,
    action: () => void,
    options: { shortcut?: string; disabled?: boolean; testid?: string } = {},
  ) => (
    <button
      type="button"
      role="menuitem"
      disabled={options.disabled}
      onClick={action}
      onKeyDown={onMenuKeyDown}
      data-testid={options.testid}
    >
      <span>{label}</span>
      <Show when={options.shortcut}>
        <small>{options.shortcut}</small>
      </Show>
    </button>
  );

  return (
    <>
      <nav class="menu-bar" aria-label="Application menu" data-testid="menu-bar">
        <div class="menu">
          {menuTitle("file", "File")}
          <Show when={openMenu() === "file"}>
            <div class="menu-dropdown" role="menu">
              {item("New", () => (setOpenMenu(null), props.onNew()), {
                testid: "menu-file-new",
              })}
              {item("Open…", showScenes, { testid: "menu-file-open" })}
              <hr />
              {item("Save", save, { testid: "menu-file-save" })}
              {item("Save As…", showSaveAsDialog, { testid: "menu-file-save-as" })}
              <hr />
              {item("Export…", showExportDialog, { testid: "menu-file-export" })}
              {item("Download scene.py", download, { testid: "menu-file-download" })}
            </div>
          </Show>
        </div>

        <div class="menu">
          {menuTitle("edit", "Edit")}
          <Show when={openMenu() === "edit"}>
            <div class="menu-dropdown" role="menu">
              {item("Undo", () => (setOpenMenu(null), props.onUndo()), {
                shortcut: "⌘Z",
                disabled: !props.canUndo,
                testid: "menu-edit-undo",
              })}
              {item("Redo", () => (setOpenMenu(null), props.onRedo()), {
                shortcut: "⇧⌘Z",
                disabled: !props.canRedo,
                testid: "menu-edit-redo",
              })}
            </div>
          </Show>
        </div>

        <div class="menu">
          {menuTitle("window", "Window")}
          <Show when={openMenu() === "window"}>
            <div class="menu-dropdown" role="menu">
              <For each={EDITING_MODES}>
                {(mode) => (
                  <button
                    type="button"
                    role="menuitemradio"
                    aria-checked={editingMode() === mode}
                    onClick={() => {
                      setEditingMode(mode);
                      setOpenMenu(null);
                    }}
                    onKeyDown={onMenuKeyDown}
                    data-testid={`menu-window-mode-${mode}`}
                  >
                    <i class="menu-check">{editingMode() === mode ? "•" : ""}</i>
                    <span>
                      {mode.charAt(0).toUpperCase() + mode.slice(1)} mode
                    </span>
                    <Show when={mode === "model"}>
                      <small>Esc</small>
                    </Show>
                  </button>
                )}
              </For>
              <hr />
              <For each={PANEL_ITEMS}>
                {(panel) => (
                  <button
                    type="button"
                    role="menuitemcheckbox"
                    aria-checked={panels()[panel.key]}
                    onClick={() => setPanelVisible(panel.key, !panels()[panel.key])}
                    onKeyDown={onMenuKeyDown}
                    data-testid={`menu-window-${panel.key}`}
                  >
                    <i class="menu-check">{panels()[panel.key] ? "✓" : ""}</i>
                    <span>{panel.label}</span>
                  </button>
                )}
              </For>
              {/* The simulation windows, the scene browser and the process
                  monitor. The last two are parked in every desk rather than
                  docked, so this is where they are opened from — and where a
                  user who has never seen the tray finds out they exist. */}
              <Show when={windowManager()}>
                {(manager) => (
                  <For each={WINDOW_ITEMS}>
                    {(entry) => (
                      <button
                        type="button"
                        role="menuitemcheckbox"
                        aria-checked={manager().status(entry.id) === "open"}
                        onClick={() => manager().toggle(entry.id)}
                        onKeyDown={onMenuKeyDown}
                        data-testid={`menu-window-${entry.id}`}
                      >
                        <i class="menu-check">
                          {manager().status(entry.id) === "open" ? "✓" : ""}
                        </i>
                        <span>{entry.label}</span>
                      </button>
                    )}
                  </For>
                )}
              </Show>
              {/* Float and dock, and the way back from both. The dock owns
                  the arrangement and publishes this manager on mount, so the
                  section simply is not there before it exists — which is also
                  the case in a unit test, where no dock is ever built. */}
              <Show when={windowManager()}>
                {(manager) => (
                  <>
                    <hr />
                    <For each={manager().windows}>
                      {(window) => (
                        <button
                          type="button"
                          role="menuitemcheckbox"
                          aria-checked={manager().isFloating(window.id)}
                          disabled={manager().status(window.id) === "closed"}
                          onClick={() => {
                            // Unlike the visibility checks above, this one
                            // closes the menu: the window it moves lands over
                            // the dropdown, and a menu underneath a floating
                            // panel is a menu you cannot click.
                            setOpenMenu(null);
                            if (manager().isFloating(window.id)) manager().dock(window.id);
                            else manager().float(window.id);
                          }}
                          onKeyDown={onMenuKeyDown}
                          data-testid={`menu-window-float-${window.id}`}
                        >
                          <i class="menu-check">
                            {manager().isFloating(window.id) ? "✓" : ""}
                          </i>
                          <span>Float {window.title}</span>
                        </button>
                      )}
                    </For>
                    <hr />
                    {item(
                      "Reset layout",
                      () => (setOpenMenu(null), manager().resetLayout()),
                      { testid: "menu-window-reset" },
                    )}
                  </>
                )}
              </Show>
            </div>
          </Show>
        </div>

        <div class="menu">
          {menuTitle("help", "Help")}
          <Show when={openMenu() === "help"}>
            <div class="menu-dropdown" role="menu">
              {item(
                "Keyboard shortcuts",
                () => (setOpenMenu(null), setDialog("help")),
                { testid: "menu-help-shortcuts" },
              )}
              <a
                href={REPO_README_URL}
                target="_blank"
                rel="noreferrer"
                role="menuitem"
                onKeyDown={onMenuKeyDown}
                data-testid="menu-help-readme"
              >
                <span>CADJOINT README</span>
                <small>↗</small>
              </a>
            </div>
          </Show>
        </div>

        <span class="menu-scene" data-testid="menu-scene-name">
          {sceneName() ?? "unsaved scene"}
          {dirty() ? " •" : ""}
        </span>
      </nav>

      <Show when={dialog() === "saveAs"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Save the scene as"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Save scene as</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <form
              class="dialog-body save-as-form"
              onSubmit={(event) => {
                event.preventDefault();
                void saveAs(saveAsName());
              }}
            >
              <label>
                File name
                <input
                  type="text"
                  value={saveAsName()}
                  onInput={(event) => {
                    setSaveAsName(event.currentTarget.value);
                    setSaveAsError("");
                  }}
                  data-testid="save-as-name"
                />
              </label>
              <Show when={saveAsError()}>
                <p class="dialog-error" data-testid="save-as-error">
                  {saveAsError()}
                </p>
              </Show>
              <p class="dialog-note">
                Written to <code>scenes/</code> in the playground server's
                working directory.
              </p>
              <button type="submit" class="primary" data-testid="save-as-confirm">
                Save
              </button>
            </form>
          </div>
        </div>
      </Show>

      <Show when={dialog() === "export"}>
        <ExportDialog onClose={() => setDialog(null)} />
      </Show>

      <Show when={dialog() === "help"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Keyboard shortcuts"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Keyboard shortcuts</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <div class="dialog-body shortcut-groups" data-testid="shortcut-dialog">
              <For each={SHORTCUT_GROUPS}>
                {(group) => (
                  <section>
                    <h4>{group.title}</h4>
                    <ul>
                      <For each={group.items}>
                        {(shortcut) => (
                          <li>
                            <span>{shortcut.action}</span>
                            <kbd>{shortcut.keys}</kbd>
                          </li>
                        )}
                      </For>
                    </ul>
                  </section>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>
    </>
  );
}
