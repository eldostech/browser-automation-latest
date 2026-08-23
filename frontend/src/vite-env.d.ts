/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin. Empty means same-origin (dev proxy / nginx). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
