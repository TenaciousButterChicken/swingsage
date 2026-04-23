import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API calls to the FastAPI server during dev so we don't fight CORS
// or hard-code ports into the frontend.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true,
      },
      "/captures": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
