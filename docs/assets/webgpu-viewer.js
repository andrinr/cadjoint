const MAX_RENDER_DIMENSION = 900;

function showFallback(root, message) {
  root.classList.remove("is-live");
  root.classList.add("is-error", "is-fallback");
  const status = root.querySelector("[data-viewer-status]");
  if (status) status.textContent = message;
}

async function startViewer(root) {
  const canvas = root.querySelector("[data-viewer-canvas]");
  const status = root.querySelector("[data-viewer-status]");
  const reset = root.querySelector("[data-viewer-reset]");

  if (!navigator.gpu) {
    showFallback(root, "WebGPU unavailable · showing preview");
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      showFallback(root, "No WebGPU adapter · showing preview");
      return;
    }

    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Could not create a WebGPU canvas context");

    const format = navigator.gpu.getPreferredCanvasFormat();
    const response = await fetch(new URL("./orbital-scene.wgsl", import.meta.url));
    if (!response.ok) throw new Error(`Could not load the example shader (${response.status})`);
    const shader = await response.text();
    const module = device.createShaderModule({ code: shader });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(`WGSL ${errors[0].lineNum}:${errors[0].linePos} ${errors[0].message}`);

    const pipeline = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    const uniformBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    let yaw = 0.75;
    let pitch = 0.32;
    let distance = 4.2;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let framePending = false;

    function resize() {
      const bounds = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(
        window.devicePixelRatio || 1,
        MAX_RENDER_DIMENSION / Math.max(bounds.width, bounds.height),
      );
      const width = Math.max(1, Math.round(bounds.width * pixelRatio));
      const height = Math.max(1, Math.round(bounds.height * pixelRatio));
      if (canvas.width === width && canvas.height === height) return;
      canvas.width = width;
      canvas.height = height;
      context.configure({ device, format, alphaMode: "opaque" });
    }

    function render() {
      framePending = false;
      resize();
      const cosPitch = Math.cos(pitch);
      const camera = [
        distance * cosPitch * Math.sin(yaw),
        distance * Math.sin(pitch),
        distance * cosPitch * Math.cos(yaw),
      ];
      device.queue.writeBuffer(
        uniformBuffer,
        0,
        new Float32Array([
          canvas.width, canvas.height, 0, 0,
          ...camera, 0,
          0, 0, 0, 0,
          0.55, 0.8, 0.35, 0,
          0.035, 0.045, 0.035, 1,
        ]),
      );

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.035, g: 0.045, b: 0.035, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    function scheduleRender() {
      if (framePending) return;
      framePending = true;
      requestAnimationFrame(render);
    }

    function resetCamera() {
      yaw = 0.75;
      pitch = 0.32;
      distance = 4.2;
      scheduleRender();
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
      lastX = event.clientX;
      lastY = event.clientY;
      scheduleRender();
    });
    canvas.addEventListener("pointerup", () => {
      dragging = false;
    });
    canvas.addEventListener("pointercancel", () => {
      dragging = false;
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      distance = Math.max(1.4, Math.min(12, distance * Math.exp(event.deltaY * 0.001)));
      scheduleRender();
    }, { passive: false });
    reset.addEventListener("click", resetCamera);
    new ResizeObserver(scheduleRender).observe(canvas);

    device.lost.then((info) => {
      showFallback(root, `WebGPU device lost · ${info.message || "showing preview"}`);
    });

    root.classList.remove("is-error", "is-fallback");
    root.classList.add("is-live");
    status.textContent = "Live · WebGPU";
    scheduleRender();
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    console.error("JAXCAD WebGPU documentation viewer failed:", error);
    showFallback(root, `${detail} · showing preview`);
  }
}

for (const root of document.querySelectorAll("[data-jaxcad-webgpu-viewer]")) {
  startViewer(root);
}
