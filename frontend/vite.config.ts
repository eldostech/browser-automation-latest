import path from 'node:path';

import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * The repository root, where `.env` lives.
 *
 * Vite's default env directory is its own root — `frontend/` — so a
 * `VITE_API_BASE` set in the repository's `.env` was silently ignored: the file
 * documents the variable, the backend reads that file, and Vite was looking
 * somewhere else entirely. Pointing `envDir` here makes the documented place
 * the real one, so there is one `.env` rather than two that must agree.
 */
const ROOT = path.resolve(__dirname, '..');

export default defineConfig(({ mode }) => {
  // Third argument '' loads every variable, not just the VITE_ prefixed ones,
  // so the proxy below can follow the backend's own PORT. Only VITE_* are
  // exposed to client code; the rest stay in this config.
  const env = loadEnv(mode, ROOT, '');

  // In dev the frontend and backend are same-origin from the browser's point
  // of view: Vite proxies /api and /healthz (including the WebSocket upgrade)
  // to the backend. That avoids CORS entirely and keeps the port in one place.
  //
  // Derived from the same PORT the backend reads, so moving the backend to
  // 8002 moves the proxy with it rather than leaving a second number to
  // remember.
  const backend =
    env.BACKEND_ORIGIN || `http://localhost:${env.PORT || '8000'}`;

  return {
    plugins: [react()],
    envDir: ROOT,
    server: {
      port: Number(env.FRONTEND_PORT || 5173),
      proxy: {
        '/api': { target: backend, changeOrigin: true, ws: true },
        '/healthz': { target: backend, changeOrigin: true },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
