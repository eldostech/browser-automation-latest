import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev the frontend and backend are same-origin from the browser's point of
// view: Vite proxies /api and /healthz (including the WebSocket upgrade) to
// the backend. That keeps API keys off the client, avoids CORS entirely, and
// means VITE_API_BASE only needs setting for split deployments.
const BACKEND = process.env.BACKEND_ORIGIN ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true, ws: true },
      '/healthz': { target: BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
