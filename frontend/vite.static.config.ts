import { resolve } from "node:path";
import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

// The public demo (static.html): no API proxy, relative asset paths, and
// its own output directory, so it can be served from any static host —
// GitHub Pages serves it under /cadjoint/, which relative paths survive.
// The jax-js runtime lives outside this package (web-spike/src), so both it
// and `@jax-js/jax`, which it imports, are resolved here explicitly.
export default defineConfig({
  base: "./",
  plugins: [
    solid(),
    // The page is static.html in the source tree, beside the playground's
    // index.html; a static host serves a directory's index.html.
    {
      name: "demo-index",
      enforce: "post",
      generateBundle(_, bundle) {
        if (bundle["static.html"]) bundle["static.html"].fileName = "index.html";
      },
    },
  ],
  resolve: {
    alias: {
      "cadjoint-jaxjs-runtime": resolve(import.meta.dirname, "tools/jaxjs-runtime.ts"),
      "@jax-js/jax": resolve(import.meta.dirname, "node_modules/@jax-js/jax"),
    },
  },
  server: { port: 5174, fs: { allow: [resolve(import.meta.dirname, "..")] } },
  preview: { port: 4318 },
  build: {
    outDir: "dist-static",
    emptyOutDir: true,
    target: "es2022",
    rollupOptions: { input: { index: resolve(import.meta.dirname, "static.html") } },
  },
});
