import { resolve } from "node:path";
import { defineConfig } from "vite";

// The static viewer (static.html): no API proxy, relative asset paths, and
// its own output directory, so it can be served from any static host.
// The jax-js runtime lives outside this package (web-spike/src), so both it
// and `@jax-js/jax`, which it imports, are resolved here explicitly.
export default defineConfig({
  base: "./",
  resolve: {
    alias: {
      "cadjoint-jaxjs-runtime": resolve(import.meta.dirname, "tools/jaxjs-runtime.ts"),
      "@jax-js/jax": resolve(import.meta.dirname, "node_modules/@jax-js/jax"),
    },
  },
  server: { port: 5174, fs: { allow: [resolve(import.meta.dirname, "..")] } },
  build: {
    outDir: "dist-static",
    emptyOutDir: true,
    target: "es2022",
    rollupOptions: { input: { static: resolve(import.meta.dirname, "static.html") } },
  },
});
