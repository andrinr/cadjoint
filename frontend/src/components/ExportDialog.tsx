/**
 * File → Export…: one object of the program, as a file.
 *
 * A small form and nothing else: the format, what to export, how fine, the
 * one option the format has, and Export. The decisions are in `export.ts`
 * (what each format takes, which names to offer, how the body is built),
 * the request is `api.exportFile`, and the download is the browser's own —
 * a blob URL on an anchor, exactly as "Download scene.py" does it.
 *
 * The run is a job like any other, so while it works the chip beside the
 * mode switcher counts the seconds and can cancel it; the dialog only says
 * that it is working, and disables the button so the same file is not
 * requested twice.
 *
 * For an analytic STEP of the scene, the dialog says what the file will
 * hold before it is written — how many faces go out on an exact surface, on
 * a fitted B-spline and as facets — read off the derived B-rep: the
 * inspector's, when it describes this program, else one the dialog derives
 * itself (a job like the export). Once written, the status line repeats the
 * file's own count.
 */

import { For, Show, createEffect, createMemo, createSignal, onCleanup } from "solid-js";
import * as api from "../api";
import {
  EXPORT_FORMATS,
  EXPORT_RESOLUTION,
  clampResolution,
  defaultExportName,
  downloadName,
  errorSummary,
  exportRequest,
  exportTargets,
  formatInfo,
  stepContentsLine,
} from "../export";
import { cancelClientJob, formatBytes, nextRequestId } from "../jobs";
import {
  brepPayload,
  brepSource,
  sceneName,
  setBrepPayload,
  setBrepSource,
  setStatus,
  source,
  studies,
} from "../state";
import type { ExportFormat } from "../types";
import { tierMissing } from "../tier";
import { DialogHeader, Select, TierReason, TierTag, ToggleSwitch } from "./ui";
import "./export.css";

export interface ExportDialogProps {
  onClose: () => void;
}

/** Hand the browser a file to save, the way the scene download does. */
function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  // Revoked on the next tick: revoking synchronously races the click in
  // some browsers and the download arrives empty.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function ExportDialog(props: ExportDialogProps) {
  const studyNames = createMemo(() => studies().map((study) => study.name));

  const [format, setFormat] = createSignal<ExportFormat>("stl");
  const [name, setName] = createSignal(defaultExportName("stl", studyNames()));
  const [resolution, setResolution] = createSignal<number>(EXPORT_RESOLUTION.default);
  const [binary, setBinary] = createSignal(true);
  const [analytic, setAnalytic] = createSignal(true);
  const [mergePlanar, setMergePlanar] = createSignal(true);
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal("");

  const info = () => formatInfo(format());
  /** STEP without the private tier: the faceted writer, said at the choice. */
  const facetedStep = () => tierMissing("step_export");
  const targets = createMemo(() => exportTargets(format(), source(), studyNames()));
  /** A study format with nothing to write is the one case the form refuses. */
  const nothingToExport = () => info().takes === "study" && targets().length === 0;

  /** An analytic STEP of the scene: the one export whose contents can be said in advance. */
  const previewsContents = () =>
    format() === "step" && analytic() && !facetedStep() && name().trim() === "scene";
  const contents = createMemo(() => {
    const payload = brepPayload();
    return payload && brepSource() === source() ? (payload.diagnostics?.step ?? null) : null;
  });
  const [deriving, setDeriving] = createSignal(false);
  let derivation: string | null = null;
  /** The program last asked for: a refusal is not asked again until it changes. */
  let asked: string | null = null;
  const stopDeriving = () => {
    if (derivation) void cancelClientJob(derivation);
    derivation = null;
    setDeriving(false);
  };
  createEffect(() => {
    const program = source();
    if (!previewsContents() || contents() || derivation || asked === program) return;
    asked = program;
    const clientId = nextRequestId();
    derivation = clientId;
    setDeriving(true);
    void api
      .brep(program, { clientId })
      .then((result) => {
        if (derivation !== clientId || !result.ok || !result.brep?.available) return;
        setBrepPayload(result.brep);
        setBrepSource(program);
      })
      .finally(() => {
        if (derivation === clientId) {
          derivation = null;
          setDeriving(false);
        }
      });
  });
  onCleanup(stopDeriving);

  const pickFormat = (next: ExportFormat) => {
    const before = info().takes;
    setFormat(next);
    setError("");
    // Switching between an object format and a study format changes what
    // the name means; a name that was an object cannot be a study.
    if (formatInfo(next).takes !== before) setName(defaultExportName(next, studyNames()));
  };

  const run = async () => {
    if (busy() || nothingToExport()) return;
    const body = exportRequest(source(), {
      format: format(),
      name: name(),
      resolution: resolution(),
      binary: binary(),
      analytic: analytic() && !facetedStep(),
      mergePlanar: mergePlanar(),
    });
    if (!body.name) {
      setError("Name the object to export.");
      return;
    }
    setBusy(true);
    setError("");
    setStatus({ kind: "busy", text: `Exporting ${body.name} as ${info().label}…` });
    try {
      const outcome = await api.exportFile(body);
      if (!outcome.ok) {
        const message = errorSummary(outcome.error);
        setError(message);
        setStatus({ kind: "error", text: message });
        return;
      }
      const filename = downloadName(outcome.filename, sceneName());
      saveBlob(outcome.blob, filename);
      const held = outcome.report?.step ? ` · ${stepContentsLine(outcome.report.step)}` : "";
      setStatus({
        kind: "ready",
        text: `Exported ${filename} · ${formatBytes(outcome.blob.size)}${held}`,
      });
      props.onClose();
    } catch (failure) {
      const message = failure instanceof Error ? failure.message : String(failure);
      setError(message);
      setStatus({ kind: "error", text: message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="dialog-backdrop" onClick={() => !busy() && props.onClose()}>
      <div
        class="dialog dialog-compact"
        role="dialog"
        aria-label="Export"
        onClick={(event) => event.stopPropagation()}
        data-testid="export-dialog"
      >
        <DialogHeader title="Export" onClose={props.onClose} closeDisabled={busy()} />
        <form
          class="dialog-body export-form"
          onSubmit={(event) => {
            event.preventDefault();
            void run();
          }}
        >
          <div class="export-row">
            <label class="export-field">
              <span>
                Format
                <Show when={format() === "step"}>
                  <TierTag kind="step_export" testId="export-tier" />
                </Show>
              </span>
              <Select
                value={format()}
                options={EXPORT_FORMATS.map((entry) => ({
                  value: entry.value,
                  label: entry.value === "step" && facetedStep() ? "STEP · faceted" : entry.label,
                }))}
                onChange={(value) => pickFormat(value as ExportFormat)}
                disabled={busy()}
                testId="export-format"
              />
            </label>
            <div class="export-option">
              <Show when={info().option}>
                {(option) => (
                  <ToggleSwitch
                    checked={
                      option().key === "binary"
                        ? binary()
                        : option().key === "analytic"
                          ? analytic() && !facetedStep()
                          : mergePlanar()
                    }
                    onChange={(checked) => {
                      if (option().key === "binary") setBinary(checked);
                      else if (option().key === "analytic") setAnalytic(checked);
                      else setMergePlanar(checked);
                    }}
                    disabled={busy() || (option().key === "analytic" && facetedStep())}
                    testId={`export-option-${option().key}`}
                  >
                    {option().label}
                  </ToggleSwitch>
                )}
              </Show>
            </div>
          </div>

          <div class="export-row">
            <label class="export-field">
              {info().takes === "study" ? "Study" : "Object"}
              <Show
                when={info().takes === "object"}
                fallback={
                  <Select
                    value={name()}
                    options={targets().map((entry) => ({ value: entry }))}
                    onChange={setName}
                    disabled={busy() || nothingToExport()}
                    testId="export-name"
                  />
                }
              >
                <input
                  type="text"
                  list="export-objects"
                  value={name()}
                  onInput={(event) => {
                    setName(event.currentTarget.value);
                    setError("");
                  }}
                  disabled={busy()}
                  spellcheck={false}
                  autocomplete="off"
                  data-testid="export-name"
                />
                <datalist id="export-objects">
                  <For each={targets()}>{(entry) => <option value={entry} />}</For>
                </datalist>
              </Show>
            </label>
            <Show when={info().takes === "object"}>
              <label class="export-field">
                <span>
                  Resolution <span class="export-unit">cells</span>
                </span>
                <input
                  type="number"
                  min={EXPORT_RESOLUTION.min}
                  max={EXPORT_RESOLUTION.max}
                  step="8"
                  value={resolution()}
                  onChange={(event) => {
                    const value = Number(event.currentTarget.value);
                    if (Number.isFinite(value)) setResolution(clampResolution(value));
                  }}
                  disabled={busy()}
                  data-testid="export-resolution"
                />
              </label>
            </Show>
          </div>

          <p class="dialog-note" data-testid="export-note">
            <Show
              when={format() === "step" && facetedStep()}
              fallback={
                nothingToExport()
                  ? "The program declares no study to export a result from."
                  : info().note
              }
            >
              <TierReason kind="step_export" testId="export-tier-reason" />
            </Show>
          </p>

          <Show when={previewsContents()}>
            <p class="dialog-note export-contents" data-testid="export-step-contents">
              <Show
                when={contents()}
                fallback={deriving() ? "Deriving the B-rep to count its faces…" : null}
              >
                {(held) => <>Will hold {stepContentsLine(held())}.</>}
              </Show>
            </p>
          </Show>

          <Show when={error()}>
            <p class="dialog-error" data-testid="export-error">
              {error()}
            </p>
          </Show>

          <div class="export-actions">
            <button
              type="submit"
              class="primary"
              disabled={busy() || nothingToExport()}
              data-testid="export-confirm"
            >
              {busy() ? "Exporting…" : "Export"}
            </button>
            <Show when={busy()}>
              <p class="export-progress" data-testid="export-progress">
                Extracting {name()} at {resolution()} cells — the chip beside the mode switcher
                can cancel it.
              </p>
            </Show>
          </div>
        </form>
      </div>
    </div>
  );
}
