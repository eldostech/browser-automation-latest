/**
 * Backend client.
 *
 * The frontend holds no credentials of any kind. It knows one thing about the
 * outside world: where the backend is. The LLM key lives in the backend's
 * environment and is never sent here, and nothing sensitive is ever put in a
 * query string.
 */

import type {
  AgentEvent,
  BatchDetail,
  BatchSummary,
  CredentialSummary,
  DistillResult,
  ExecutionRecord,
  RunDetail,
  RunSummary,
  ServerConfig,
  UseCase,
  UseCaseSummary,
} from './events';

/** Empty by default: Vite (dev) and nginx (prod) proxy /api to the backend. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export interface CreateRunPayload {
  task: string;
  start_url?: string | null;
  max_steps?: number;
  timeout_seconds?: number;
  allowed_domains?: string[];
  require_approval?: boolean;
  screenshot_every_step?: boolean;
  headless?: boolean;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  getConfig: () => request<ServerConfig>('/api/config'),

  createRun: (payload: CreateRunPayload) =>
    request<{ run_id: string; status: string }>('/api/runs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listRuns: (status?: string, limit = 50) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (status) params.set('status', status);
    return request<{ runs: RunSummary[]; total: number }>(`/api/runs?${params}`);
  },

  getRun: (runId: string) => request<RunDetail>(`/api/runs/${runId}`),

  getEvents: (runId: string, afterSeq = 0) =>
    request<{ run_id: string; events: AgentEvent[] }>(
      `/api/runs/${runId}/events?after_seq=${afterSeq}`,
    ),

  cancelRun: (runId: string) =>
    request<{ run_id: string; cancelled: boolean }>(`/api/runs/${runId}/cancel`, {
      method: 'POST',
    }),

  resolveApproval: (runId: string, approvalId: string, decision: 'approve' | 'reject', note?: string) =>
    request<{ run_id: string; decision: string }>(`/api/runs/${runId}/approve`, {
      method: 'POST',
      body: JSON.stringify({ approval_id: approvalId, decision, note }),
    }),

  health: () => request<Record<string, unknown>>('/healthz'),

  // --- use cases ------------------------------------------------------------

  /** Promote a succeeded run into a reusable use case. The one LLM call. */
  distillRun: (runId: string) =>
    request<DistillResult>(`/api/runs/${runId}/distill`, { method: 'POST' }),

  listUseCases: (status?: string) => {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    return request<{ usecases: UseCaseSummary[] }>(`/api/usecases?${params}`);
  },

  getUseCase: (id: string, version?: number) => {
    const params = new URLSearchParams();
    if (version) params.set('version', String(version));
    return request<{ definition: UseCase; versions: { version: number; created_at: string }[] }>(
      `/api/usecases/${id}?${params}`,
    );
  },

  /** Saves reviewer edits as a NEW version; existing versions are immutable. */
  updateUseCase: (id: string, definition: UseCase) =>
    request<{ usecase_id: string; version: number; status: string }>(`/api/usecases/${id}`, {
      method: 'PUT',
      body: JSON.stringify(definition),
    }),

  publishUseCase: (id: string) =>
    request<{ usecase_id: string; status: string }>(`/api/usecases/${id}/publish`, {
      method: 'POST',
    }),

  archiveUseCase: (id: string) =>
    request<{ usecase_id: string; status: string }>(`/api/usecases/${id}`, { method: 'DELETE' }),

  // --- executing (zero LLM calls) -------------------------------------------

  executeUseCase: (
    id: string,
    payload: { inputs: Record<string, unknown>; credential_id?: string | null },
  ) =>
    request<{
      execution_id: string;
      run_id: string;
      status: string;
      outputs: Record<string, unknown>;
      error: string | null;
      llm_calls: number;
      llm_tokens: number;
    }>(`/api/usecases/${id}/execute`, { method: 'POST', body: JSON.stringify(payload) }),

  listExecutions: (usecaseId: string) =>
    request<{ executions: ExecutionRecord[] }>(`/api/usecases/${usecaseId}/executions`),

  activeExecution: () =>
    request<{ active: Record<string, unknown> | null }>('/api/executions/active'),

  // --- batches ---------------------------------------------------------------

  startBatch: (id: string, payload: { csv: string; credential_id?: string | null }) =>
    request<{ batch_id: string; total: number; columns: string[]; warnings: string[] }>(
      `/api/usecases/${id}/batch`,
      { method: 'POST', body: JSON.stringify(payload) },
    ),

  getBatch: (batchId: string) => request<BatchDetail>(`/api/batches/${batchId}`),

  listBatches: (usecaseId: string) =>
    request<{ batches: BatchSummary[] }>(`/api/usecases/${usecaseId}/batches`),

  resumeBatch: (batchId: string, credentialId?: string | null) =>
    request<{ batch_id: string; rows: number }>(`/api/batches/${batchId}/resume`, {
      method: 'POST',
      body: JSON.stringify({ credential_id: credentialId ?? null }),
    }),

  cancelBatch: (batchId: string) =>
    request<{ batch_id: string; cancelled: boolean }>(`/api/batches/${batchId}/cancel`, {
      method: 'POST',
    }),

  // --- credentials (write-only) ---------------------------------------------

  listCredentials: () =>
    request<{ credentials: CredentialSummary[]; vault_available: boolean }>('/api/credentials'),

  createCredential: (name: string, values: Record<string, string>) =>
    request<{ id: string; name: string; slots: string[] }>('/api/credentials', {
      method: 'POST',
      body: JSON.stringify({ name, values }),
    }),

  deleteCredential: (id: string) =>
    request<{ id: string; deleted: boolean }>(`/api/credentials/${id}`, { method: 'DELETE' }),
};

/** Download URL for a finished batch's results. */
export function batchResultsUrl(batchId: string): string {
  return `${API_BASE}/api/batches/${batchId}/results.csv`;
}

/** Absolute URL for an artifact returned by the backend as a relative path. */
export function artifactUrl(url: string): string {
  return url.startsWith('http') ? url : `${API_BASE}${url}`;
}

/** ws:// or wss:// URL for a run's live event stream. */
export function streamUrl(runId: string, afterSeq: number): string {
  const base = API_BASE || window.location.origin;
  const url = new URL(`${base}/api/runs/${runId}/stream`);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('after_seq', String(afterSeq));
  return url.toString();
}
