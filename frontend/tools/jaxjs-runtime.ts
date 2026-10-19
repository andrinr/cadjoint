// The jax-js runtime the static viewer loads, re-exported from the jax-js
// backend's page (web-spike/src) under one module name. Vite resolves
// `cadjoint-jaxjs-runtime` here (vite.static.config.ts); TypeScript reads
// src/static/jaxjsRuntime.d.ts instead, so these sources are typechecked by
// their own package and not under the frontend's settings.
export { sample } from "../../web-spike/src/chunk";
export { dualContour } from "../../web-spike/src/dc";
export { badEdges, stl, volume } from "../../web-spike/src/mesh";
export { field, sceneRoot } from "../../web-spike/src/runtime";
