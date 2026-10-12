import { defineConfig } from "vite";

// local development: the pursuit server runs on :8600 (demo/server); production sets VITE_API_BASE
export default defineConfig({
  server: { proxy: { "/api": { target: "http://127.0.0.1:8600", changeOrigin: true } } },
  preview: { proxy: { "/api": { target: "http://127.0.0.1:8600", changeOrigin: true } } },
});
