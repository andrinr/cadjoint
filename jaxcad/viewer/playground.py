"""Local split-pane Python and WebGPU playground for JAXCAD scenes."""

from __future__ import annotations

import argparse
import contextlib
import json
import secrets
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

DEFAULT_PORT = 8765
MAX_SOURCE_BYTES = 100_000
COMPILE_TIMEOUT_SECONDS = 20

EXAMPLE_SOURCE = """from jaxcad.render import Material
from jaxcad.sdf.boolean import Union
from jaxcad.sdf.primitives import Plane, Sphere, Torus
from jaxcad.sdf.transforms import Rotate, Translate

# Geometry and materials take the same StableHLO -> WGSL route. Switch on
# Path trace to progressively resolve indirect light, reflections, and glass.
red = Material(color=[0.8, 0.035, 0.015], roughness=0.42)
gold = Material(
    color=[1.0, 0.5, 0.06],
    roughness=0.22,
    metallic=0.9,
    reflectivity=0.2,
)
glass = Material(color=[0.94, 0.98, 1.0], roughness=0.05, opacity=0.0, ior=1.5)
ground = Material(color=[0.1, 0.12, 0.16], roughness=0.75)

scene = Union(
    Translate(Sphere(radius=0.72, material=red), [-1.0, -0.38, 0.0]),
    Translate(
        Rotate(Torus(major_radius=0.62, minor_radius=0.2, material=gold), "y", 0.45),
        [0.15, -0.28, 0.0],
    ),
    # Float the glass in front of the torus from the default camera so its
    # refraction is immediately visible in path-tracing mode.
    Translate(Sphere(radius=0.58, material=glass), [1.0, 0.25, 1.0]),
    Plane(-1.1, material=ground),
    smoothness=0.0,
)
"""


def compile_source(source: str, timeout: float = COMPILE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Compile playground source in a disposable child process."""
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be a string."}
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return {
            "ok": False,
            "error": f"Source is larger than the {MAX_SOURCE_BYTES:,}-byte limit.",
        }

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "jaxcad.viewer._compile_worker"],
            input=json.dumps({"source": source}),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Compilation exceeded the {timeout:g}-second timeout.",
        }

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "The compiler process exited unexpectedly."
        return {"ok": False, "error": detail[-8_000:]}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {"ok": False, "error": f"Invalid compiler response:\n{detail[-8_000:]}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "Invalid compiler response."}
    return result


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>JAXCAD WebGPU Playground</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #e9e8e2;
      --muted: #95958f;
      --panel: #111312;
      --line: #2a2d29;
      --lime: #d9ff57;
      --coral: #ff8167;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body {
      background: #090b0a;
      color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .shell { display: grid; grid-template-rows: 58px 1fr; height: 100%; }
    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 18px; border-bottom: 1px solid var(--line); background: #0d0f0e;
    }
    .brand { display: flex; align-items: center; gap: 11px; letter-spacing: -.02em; }
    .mark {
      display: grid; place-items: center; width: 30px; height: 30px; color: #0a0b09;
      background: var(--lime); border-radius: 9px; font-weight: 900; transform: rotate(-4deg);
    }
    .brand strong { font-size: 16px; }
    .brand span, .hint { color: var(--muted); font-size: 12px; }
    .actions { display: flex; gap: 8px; }
    button {
      border: 1px solid #363a35; border-radius: 8px; padding: 8px 12px;
      background: #171a17; color: var(--ink); font: inherit; font-weight: 650; cursor: pointer;
    }
    button:hover { border-color: #555b52; background: #1d211d; }
    button.primary { border-color: var(--lime); background: var(--lime); color: #10120d; }
    button.active { border-color: var(--lime); color: var(--lime); background: #202616; }
    button:disabled { opacity: .55; cursor: wait; }
    select {
      border: 1px solid #363a35; border-radius: 8px; padding: 8px 10px;
      color: var(--ink); background: #171a17; font: inherit; font-weight: 650;
    }
    main { min-height: 0; display: grid; grid-template-columns: minmax(360px, 43%) 1fr; }
    .editor-pane { min-width: 0; display: grid; grid-template-rows: 42px 1fr auto; border-right: 1px solid var(--line); background: var(--panel); }
    .pane-title { display: flex; align-items: center; justify-content: space-between; padding: 0 14px; border-bottom: 1px solid var(--line); }
    .pane-title strong { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
    .file { color: var(--muted); font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    textarea {
      width: 100%; height: 100%; resize: none; border: 0; outline: none; padding: 20px 21px;
      color: #e4e6df; background: #111312; caret-color: var(--lime); tab-size: 4;
      font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .console {
      display: none; max-height: 180px; overflow: auto; margin: 0; padding: 12px 16px;
      border-top: 1px solid #5c302a; color: #ffb4a6; background: #1c1210;
      white-space: pre-wrap; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .viewer { position: relative; min-width: 0; min-height: 0; overflow: hidden; background: #151814; }
    canvas { display: block; width: 100%; height: 100%; }
    .viewer::after {
      content: ""; position: absolute; inset: 0; pointer-events: none;
      background: radial-gradient(circle at 50% 45%, transparent 15%, rgba(2, 3, 2, .38) 100%);
    }
    .status {
      position: absolute; z-index: 2; top: 16px; left: 16px; display: flex; align-items: center; gap: 8px;
      padding: 7px 10px; border: 1px solid rgba(255,255,255,.1); border-radius: 999px;
      color: #c6c8c0; background: rgba(12,14,12,.72); backdrop-filter: blur(10px); font-size: 12px;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #7c8178; }
    .status.ready .dot { background: var(--lime); box-shadow: 0 0 12px var(--lime); }
    .status.error .dot { background: var(--coral); }
    .viewer-error {
      display: none; position: absolute; z-index: 3; inset: 50% auto auto 50%;
      width: min(520px, calc(100% - 48px)); transform: translate(-50%, -50%);
      padding: 18px 20px; border: 1px solid #713f36; border-radius: 12px;
      color: #ffd1c8; background: rgba(35, 18, 15, .94);
      box-shadow: 0 20px 70px rgba(0, 0, 0, .45);
      white-space: pre-wrap; font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .viewer-error.visible { display: block; }
    .viewer-help { position: absolute; z-index: 2; right: 16px; bottom: 14px; color: rgba(235,235,228,.55); font-size: 11px; }
    dialog { width: min(860px, calc(100% - 48px)); height: min(680px, calc(100% - 48px)); padding: 0; border: 1px solid #3a3e38; border-radius: 13px; color: var(--ink); background: #111312; }
    dialog::backdrop { background: rgba(0,0,0,.65); backdrop-filter: blur(4px); }
    .dialog-head { height: 48px; display: flex; align-items: center; justify-content: space-between; padding: 0 14px 0 18px; border-bottom: 1px solid var(--line); }
    .dialog-head button { padding: 5px 9px; }
    pre { height: calc(100% - 48px); margin: 0; overflow: auto; padding: 18px; color: #cbd8bd; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
    @media (max-width: 760px) {
      .hint { display: none; }
      main { grid-template-columns: 1fr; grid-template-rows: 49% 51%; }
      .editor-pane { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand"><div class="mark">J</div><div><strong>JAXCAD Playground</strong><br><span>Python → StableHLO → WGSL</span></div></div>
      <div class="actions">
        <span class="hint">Run with Ctrl / ⌘ + Enter</span>
        <button id="wgsl-button" type="button" disabled>WGSL</button>
        <button id="path-button" type="button" disabled>Path trace</button>
        <select id="quality-select" aria-label="Path-tracing quality">
          <option value="draft">Draft</option>
          <option value="high" selected>High</option>
          <option value="ultra">Ultra</option>
        </select>
        <button id="reset-button" type="button">Reset</button>
        <button id="run-button" class="primary" type="button">Run scene</button>
      </div>
    </header>
    <main>
      <section class="editor-pane">
        <div class="pane-title"><strong>Scene</strong><span class="file">scene.py</span></div>
        <textarea id="editor" aria-label="Python scene source" spellcheck="false" wrap="off"></textarea>
        <pre id="console" class="console"></pre>
      </section>
      <section class="viewer">
        <canvas id="canvas"></canvas>
        <div id="status" class="status"><span class="dot"></span><span id="status-text">Starting WebGPU…</span></div>
        <div id="viewer-error" class="viewer-error" role="alert"></div>
        <div class="viewer-help">Drag to orbit · Scroll to zoom</div>
      </section>
    </main>
  </div>
  <dialog id="wgsl-dialog">
    <div class="dialog-head"><strong>Generated WGSL</strong><button id="close-dialog" type="button">Close</button></div>
    <pre id="wgsl-source"></pre>
  </dialog>
  <script>
    const TOKEN = __TOKEN_JSON__;
    const EXAMPLE_SOURCE = __EXAMPLE_JSON__;
    const editor = document.querySelector("#editor");
    const canvas = document.querySelector("#canvas");
    const runButton = document.querySelector("#run-button");
    const resetButton = document.querySelector("#reset-button");
    const wgslButton = document.querySelector("#wgsl-button");
    const pathButton = document.querySelector("#path-button");
    const qualitySelect = document.querySelector("#quality-select");
    const consoleEl = document.querySelector("#console");
    const viewerError = document.querySelector("#viewer-error");
    const statusEl = document.querySelector("#status");
    const statusText = document.querySelector("#status-text");
    const dialog = document.querySelector("#wgsl-dialog");
    const wgslSource = document.querySelector("#wgsl-source");
    editor.value = EXAMPLE_SOURCE;

    let device, context, format, uniformBuffer;
    let previewPipeline, previewBindGroup, pathPipeline, presentPipeline;
    let accumulationTextures = [];
    let pathBindGroups = [], presentBindGroups = [];
    let accumulationWidth = 0, accumulationHeight = 0;
    let accumulationReadIndex = 0, sampleCount = 0;
    let pathTracing = false;
    let pathReady = false;
    let shaderRevision = 0;
    let latestShaders = null;
    let webgpuError = "";
    let yaw = 0.75, pitch = 0.32, distance = 4.2;
    let dragging = false, lastX = 0, lastY = 0;
    const QUALITY_PRESETS = {
      draft: {
        label: "Draft", pixelBudget: 320_000, maxRatio: 1, bounces: 3, samples: 128,
      },
      high: {
        label: "High", pixelBudget: 900_000, maxRatio: 1.25, bounces: 6, samples: 512,
      },
      ultra: {
        label: "Ultra", pixelBudget: 1_600_000, maxRatio: 2, bounces: 8, samples: 1024,
      },
    };

    function selectedQuality() {
      return QUALITY_PRESETS[qualitySelect.value] || QUALITY_PRESETS.high;
    }

    function setStatus(kind, message) {
      statusEl.className = `status ${kind || ""}`;
      statusText.textContent = message;
    }

    function showError(message) {
      consoleEl.textContent = message;
      consoleEl.style.display = "block";
      viewerError.textContent = message;
      viewerError.classList.add("visible");
      setStatus("error", "Scene failed");
    }

    function clearError() {
      viewerError.textContent = "";
      viewerError.classList.remove("visible");
    }

    function firefoxLinuxHint() {
      const isFirefox = /Firefox\//.test(navigator.userAgent);
      const isLinux = /Linux/.test(navigator.userAgent);
      return isFirefox && isLinux
        ? " Stable Firefox on Linux does not currently enable WebGPU by default; use Chrome/Chromium or Firefox Nightly with dom.webgpu.enabled."
        : "";
    }

    async function withTimeout(promise, milliseconds, label) {
      let timeout;
      try {
        return await Promise.race([
          promise,
          new Promise((_, reject) => {
            timeout = setTimeout(
              () => reject(new Error(`${label} timed out after ${milliseconds / 1000} seconds.`)),
              milliseconds,
            );
          }),
        ]);
      } finally {
        clearTimeout(timeout);
      }
    }

    async function initWebGPU() {
      if (!window.isSecureContext) {
        webgpuError = "WebGPU requires a secure context. Open this viewer on localhost or over HTTPS.";
        showError(webgpuError);
        return;
      }
      if (!navigator.gpu) {
        webgpuError =
          "WebGPU is not available in this browser." +
          firefoxLinuxHint() +
          " Use a current WebGPU-capable browser.";
        showError(webgpuError);
        return;
      }
      const adapter = await withTimeout(
        navigator.gpu.requestAdapter({ powerPreference: "high-performance" }),
        8_000,
        "WebGPU adapter request",
      );
      if (!adapter) {
        webgpuError = "No WebGPU adapter was found on this system." + firefoxLinuxHint();
        showError(webgpuError);
        return;
      }
      device = await withTimeout(adapter.requestDevice(), 8_000, "WebGPU device request");
      context = canvas.getContext("webgpu");
      if (!context) {
        webgpuError = "The browser exposed WebGPU but could not create a WebGPU canvas context.";
        showError(webgpuError);
        return;
      }
      format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });
      uniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.addEventListener("uncapturederror", (event) => {
        const message = event.error?.message || "Unknown WebGPU validation error.";
        console.error(message);
        showError(`WebGPU validation error: ${message}`);
      });
      device.lost.then((info) => {
        webgpuError = `WebGPU device lost: ${info.message}`;
        showError(webgpuError);
      });
      if (latestShaders) await installShaders(latestShaders);
      else setStatus("", "WebGPU ready");
    }

    async function checkedShaderModule(code, label) {
      const module = device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((item) => item.type === "error");
      if (errors.length) {
        throw new Error(
          errors.map((item) => `${label} ${item.lineNum}:${item.linePos} ${item.message}`).join("\n")
        );
      }
      return module;
    }

    function destroyAccumulationTextures() {
      for (const texture of accumulationTextures) texture.destroy();
      accumulationTextures = [];
      pathBindGroups = [];
      presentBindGroups = [];
      accumulationWidth = 0;
      accumulationHeight = 0;
    }

    function resetAccumulation() {
      sampleCount = 0;
    }

    function ensureAccumulationTextures() {
      if (
        accumulationTextures.length === 2 &&
        accumulationWidth === canvas.width &&
        accumulationHeight === canvas.height
      ) return;

      destroyAccumulationTextures();
      accumulationWidth = canvas.width;
      accumulationHeight = canvas.height;
      const descriptor = {
        size: [canvas.width, canvas.height],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      };
      accumulationTextures = [
        device.createTexture(descriptor),
        device.createTexture(descriptor),
      ];
      pathBindGroups = accumulationTextures.map((texture) => device.createBindGroup({
        layout: pathPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: texture.createView() },
        ],
      }));
      presentBindGroups = accumulationTextures.map((texture) => device.createBindGroup({
        layout: presentPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: texture.createView() }],
      }));
      accumulationReadIndex = 0;
      resetAccumulation();
    }

    async function installShaders(shaders) {
      if (!device) {
        showError(webgpuError || "WGSL compiled, but WebGPU is unavailable.");
        return;
      }
      const revision = ++shaderRevision;
      pathReady = false;
      pathButton.disabled = true;

      const previewModule = await checkedShaderModule(shaders.preview, "Preview WGSL");
      const nextPreviewPipeline = await device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: previewModule, entryPoint: "vs_main" },
        fragment: { module: previewModule, entryPoint: "fs_main", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
      if (revision !== shaderRevision) return;
      previewPipeline = nextPreviewPipeline;
      previewBindGroup = device.createBindGroup({
        layout: previewPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      });
      clearError();
      destroyAccumulationTextures();
      setStatus("ready", "Preview ready · preparing path tracer…");
      scheduleRender();

      const [pathModule, presentModule] = await Promise.all([
        checkedShaderModule(shaders.path, "Path-tracing WGSL"),
        checkedShaderModule(shaders.present, "Presentation WGSL"),
      ]);
      const [nextPathPipeline, nextPresentPipeline] = await Promise.all([
        device.createRenderPipelineAsync({
          layout: "auto",
          vertex: { module: pathModule, entryPoint: "vs_main" },
          fragment: {
            module: pathModule,
            entryPoint: "fs_path_trace",
            targets: [{ format: "rgba16float" }],
          },
          primitive: { topology: "triangle-list" },
        }),
        device.createRenderPipelineAsync({
          layout: "auto",
          vertex: { module: presentModule, entryPoint: "vs_present" },
          fragment: {
            module: presentModule,
            entryPoint: "fs_present",
            targets: [{ format }],
          },
          primitive: { topology: "triangle-list" },
        }),
      ]);
      if (revision !== shaderRevision) return;
      pathPipeline = nextPathPipeline;
      presentPipeline = nextPresentPipeline;
      pathReady = true;
      destroyAccumulationTextures();
      pathButton.disabled = false;
      setStatus("ready", "Live · WebGPU");
      invalidateRender();
    }

    async function compileScene() {
      if (runButton.disabled) return;
      runButton.disabled = true;
      runButton.textContent = "Compiling…";
      consoleEl.style.display = "none";
      setStatus("", "JAX compiling…");
      try {
        const response = await fetch("/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Jaxcad-Token": TOKEN },
          body: JSON.stringify({ source: editor.value }),
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || `Compile request failed (${response.status})`);
        latestShaders = {
          preview: result.preview_shader,
          path: result.path_shader,
          present: result.present_shader,
        };
        wgslSource.textContent = pathTracing ? latestShaders.path : latestShaders.preview;
        wgslButton.disabled = false;
        if (result.output) {
          consoleEl.textContent = result.output;
          consoleEl.style.display = "block";
        }
        await installShaders(latestShaders);
      } catch (error) {
        showError(error instanceof Error ? error.message : String(error));
      } finally {
        runButton.disabled = false;
        runButton.textContent = "Run scene";
      }
    }

    function resize() {
      const cssWidth = Math.max(1, canvas.clientWidth);
      const cssHeight = Math.max(1, canvas.clientHeight);
      const quality = selectedQuality();
      const maxRatio = pathTracing ? quality.maxRatio : 1.5;
      const requestedRatio = Math.min(devicePixelRatio || 1, maxRatio);
      const pixelBudget = pathTracing ? quality.pixelBudget : 1_200_000;
      const budgetRatio = Math.sqrt(pixelBudget / (cssWidth * cssHeight));
      const ratio = Math.min(requestedRatio, budgetRatio);
      const width = Math.max(1, Math.floor(cssWidth * ratio));
      const height = Math.max(1, Math.floor(cssHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        context.configure({ device, format, alphaMode: "opaque" });
        destroyAccumulationTextures();
        return true;
      }
      return false;
    }

    let framePending = false;
    function scheduleRender() {
      if (framePending) return;
      framePending = true;
      requestAnimationFrame(render);
    }

    function invalidateRender() {
      resetAccumulation();
      scheduleRender();
    }

    function render() {
      framePending = false;
      if (!device || !previewPipeline || !previewBindGroup) return;
      resize();
      const cp = Math.cos(pitch);
      const camera = [
        distance * cp * Math.sin(yaw),
        distance * Math.sin(pitch),
        distance * cp * Math.cos(yaw),
      ];
      const values = new Float32Array([
        canvas.width, canvas.height, 0, 0,
        ...camera, 0,
        0, 0, 0, 0,
        0.55, 0.8, 0.35, 3,
        0.035, 0.045, 0.035, 1,
        sampleCount, selectedQuality().bounces, 1, 0,
      ]);
      device.queue.writeBuffer(uniformBuffer, 0, values);
      const encoder = device.createCommandEncoder();
      const renderPath = pathTracing && pathReady && !dragging;

      if (renderPath) {
        ensureAccumulationTextures();
        const writeIndex = 1 - accumulationReadIndex;
        const tracePass = encoder.beginRenderPass({
          colorAttachments: [{
            view: accumulationTextures[writeIndex].createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear", storeOp: "store",
          }],
        });
        tracePass.setPipeline(pathPipeline);
        tracePass.setBindGroup(0, pathBindGroups[accumulationReadIndex]);
        tracePass.draw(3);
        tracePass.end();

        const presentPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.035, g: 0.045, b: 0.035, a: 1 },
            loadOp: "clear", storeOp: "store",
          }],
        });
        presentPass.setPipeline(presentPipeline);
        presentPass.setBindGroup(0, presentBindGroups[writeIndex]);
        presentPass.draw(3);
        presentPass.end();
        accumulationReadIndex = writeIndex;
      } else {
        const previewPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.035, g: 0.045, b: 0.035, a: 1 },
            loadOp: "clear", storeOp: "store",
          }],
        });
        previewPass.setPipeline(previewPipeline);
        previewPass.setBindGroup(0, previewBindGroup);
        previewPass.draw(3);
        previewPass.end();
      }
      device.queue.submit([encoder.finish()]);

      if (renderPath) {
        sampleCount += 1;
        const quality = selectedQuality();
        setStatus("ready", `${quality.label} path trace · ${sampleCount} spp`);
        if (sampleCount < quality.samples) scheduleRender();
      } else {
        setStatus(
          "ready",
          pathTracing && !pathReady
            ? "Preview · preparing path tracer…"
            : pathTracing
              ? "Moving · WebGPU preview"
              : "Live · WebGPU preview",
        );
      }
    }

    canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      yaw -= (event.clientX - lastX) * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + (event.clientY - lastY) * 0.008));
      lastX = event.clientX; lastY = event.clientY;
      invalidateRender();
    });
    function finishDrag(event) {
      if (!dragging) return;
      dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      invalidateRender();
    }
    canvas.addEventListener("pointerup", finishDrag);
    canvas.addEventListener("pointercancel", finishDrag);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault(); distance = Math.max(1.4, Math.min(12, distance * Math.exp(event.deltaY * 0.001)));
      invalidateRender();
    }, { passive: false });
    editor.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); compileScene(); }
      if (event.key === "Tab") {
        event.preventDefault();
        const start = editor.selectionStart;
        editor.setRangeText("    ", start, editor.selectionEnd, "end");
      }
    });
    runButton.addEventListener("click", compileScene);
    resetButton.addEventListener("click", () => { editor.value = EXAMPLE_SOURCE; compileScene(); });
    pathButton.addEventListener("click", () => {
      pathTracing = !pathTracing;
      pathButton.classList.toggle("active", pathTracing);
      pathButton.textContent = pathTracing ? "Preview" : "Path trace";
      if (latestShaders) {
        wgslSource.textContent = pathTracing ? latestShaders.path : latestShaders.preview;
      }
      invalidateRender();
    });
    qualitySelect.addEventListener("change", invalidateRender);
    wgslButton.addEventListener("click", () => {
      if (latestShaders) {
        wgslSource.textContent = pathTracing ? latestShaders.path : latestShaders.preview;
      }
      dialog.showModal();
    });
    document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
    new ResizeObserver(invalidateRender).observe(canvas);

    async function start() {
      try {
        await initWebGPU();
      } catch (error) {
        webgpuError = error instanceof Error ? error.message : String(error);
        showError(webgpuError);
      }
      await compileScene();
    }

    start();
  </script>
</body>
</html>
"""


def _page(token: str) -> bytes:
    html = _PAGE.replace("__TOKEN_JSON__", json.dumps(token)).replace(
        "__EXAMPLE_JSON__", json.dumps(EXAMPLE_SOURCE)
    )
    return html.encode("utf-8")


def _is_loopback_host(host_header: str | None) -> bool:
    if not host_header:
        return False
    hostname = urlsplit(f"//{host_header}").hostname
    return hostname in {"127.0.0.1", "localhost", "::1"}


def make_handler(token: str) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler bound to one unguessable browser session token."""

    class PlaygroundHandler(BaseHTTPRequestHandler):
        server_version = "JAXCADPlayground/0.1"

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def _host_allowed(self) -> bool:
            if _is_loopback_host(self.headers.get("Host")):
                return True
            self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Invalid Host header."})
            return False

        def do_GET(self) -> None:
            if not self._host_allowed():
                return
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = _page(token)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'",
            )
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_POST(self) -> None:
            if not self._host_allowed():
                return
            if self.path != "/compile":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
                return
            if not secrets.compare_digest(self.headers.get("X-Jaxcad-Token", ""), token):
                self._send_json(
                    HTTPStatus.FORBIDDEN, {"ok": False, "error": "Invalid session token."}
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_SOURCE_BYTES * 2:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"ok": False, "error": "Invalid request size."},
                )
                return
            try:
                request = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON request."}
                )
                return
            result = (
                compile_source(request.get("source"))
                if isinstance(request, dict)
                else {
                    "ok": False,
                    "error": "The request body must be an object.",
                }
            )
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY
            self._send_json(status, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

    return PlaygroundHandler


def create_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create a loopback-only playground server."""
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(token))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port (default: 8765)")
    parser.add_argument("--open", action="store_true", help="open the playground in your browser")
    args = parser.parse_args()

    server = create_server(args.port)
    address, port = server.server_address
    url = f"http://{address}:{port}/"
    print(f"JAXCAD playground: {url}")
    print("Local source is executed as Python in a timed child process. Only run code you trust.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping playground.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
