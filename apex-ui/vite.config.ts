import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/app/",
  build: {
    outDir: "../apex/static/app",
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/auth": "http://localhost:8765",
      "/config": "http://localhost:8765",
      "/subscribe": "http://localhost:8765",
      "/context": "http://localhost:8765",
      "/events": "http://localhost:8765",
      "/state": "http://localhost:8765",
      "/stream": {
        target: "ws://localhost:8765",
        ws: true,
      },
    },
  },
});
