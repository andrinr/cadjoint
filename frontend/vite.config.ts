import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

// The Python playground server owns the API; in dev, Vite proxies to it so the
// browser sees one origin and the session token flow is identical to production.
const API_TARGET = process.env.CADJOINT_SERVER ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [solid()],
  build: {
    // Built assets are committed so `pip install cadjoint` needs no Node toolchain.
    outDir: "../cadjoint/viewer/static",
    emptyOutDir: true,
    target: "es2022",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": API_TARGET,
      "/compile": API_TARGET,
      "/patch": API_TARGET,
    },
  },
});
