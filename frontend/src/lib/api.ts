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
import { session, type CurrentUser } from './session';

/** Empty by default: Vite (dev) and nginx (prod) proxy /api to the backend. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

/**
 * One named value a recording will use.
 *
 * `secret: true` changes three things at once on the server: the model is
 * shown a placeholder instead of the value, the value is registered for
 * redaction, and the resulting use case gets a credential slot rather than an
 * input column. The value is write-only in both directions — no endpoint
 * returns it, and nothing here should ever put it in application state longer
 * than the form needs it.
 */
export interface DeclaredField {
  name: string;
  value: string;
  secret: boolean;
  description?: string;
}

export interface CreateRunPayload {
  task: string;
  start_url?: string | null;
  /** Values the instruction refers to, named. See DeclaredField. */
  fields?: DeclaredField[];
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
    ...init,
    // Spread first, then set headers, so a caller passing its own headers
    // cannot accidentally drop the Authorization one.
    headers: {
      'Content-Type': 'application/json',
      ...session.headers(),
      ...(init?.headers ?? {}),
    },
  });

  if (response.status === 401) {
    // The token expired, was revoked, or the password changed. Clearing it
    // here means every caller gets the sign-in screen without each one having
    // to handle 401 itself.
    session.clear();
  }

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

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: CurrentUser;
}

export const api = {
  // -- authentication -------------------------------------------------------
  login: async (email: string, password: string): Promise<CurrentUser> => {
    const body = await request<LoginResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    session.start(body.access_token, body.user);
    return body.user;
  },

  logout: async (): Promise<void> => {
    try {
      await request<{ logged_out: boolean }>('/api/auth/logout', { method: 'POST' });
    } finally {
      // Local sign-out happens even if the revoke call fails, so a network
      // problem cannot leave someone stuck signed in.
      session.clear();
    }
  },

  me: () => request<CurrentUser>('/api/auth/me'),

  changePassword: (current_password: string, new_password: string) =>
    request<{ changed: boolean; note: string }>('/api/auth/password', {
      method: 'POST',
      body: JSON.stringify({ current_password, new_password }),
    }),

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
  /**
   * Turn a finished recording into a use case.
   *
   * `saveCredentialAs` decides what happens to the credentials the recording
   * used: naming one keeps them, omitting it discards them. There is no third
   * option — they are held in memory on the backend only until this call
   * resolves, so not choosing *is* discarding.
   */
  distillRun: (runId: string, options?: { name?: string; saveCredentialAs?: string | null }) =>
    request<DistillResult>(`/api/runs/${runId}/distill`, {
      method: 'POST',
      body: JSON.stringify({
        name: options?.name ?? null,
        save_credential_as: options?.saveCredentialAs ?? null,
      }),
    }),

  /** Which credential slots this run still holds values for. Names only. */
  heldCredentialSlots: (runId: string) =>
    request<{ run_id: string; slots: string[] }>(`/api/runs/${runId}/credential-slots`),

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

  /** Change the label only. No new version — a name is not part of the recipe. */
  renameUseCase: (id: string, name: string, description?: string) =>
    request<{ usecase_id: string; name: string }>(`/api/usecases/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name, description }),
    }),

  publishUseCase: (id: string) =>
    request<{ usecase_id: string; status: string }>(`/api/usecases/${id}/publish`, {
      method: 'POST',
    }),

  /** Reversible: hides it from the active list, keeps every reference. */
  archiveUseCase: (id: string) =>
    request<{ usecase_id: string; status: string }>(`/api/usecases/${id}`, { method: 'DELETE' }),

  /** Permanent: removes the use case, its versions and its execution records. */
  deleteUseCase: (id: string) =>
    request<{ usecase_id: string; deleted: boolean; removed: Record<string, number> }>(
      `/api/usecases/${id}?purge=true`,
      { method: 'DELETE' },
    ),

  // --- executing (zero LLM calls) -------------------------------------------

  executeUseCase: (
    id: string,
    payload: {
      inputs: Record<string, unknown>;
      credential_id?: string | null;
      /** false opens a visible browser so you can watch the replay. */
      headless?: boolean;
    },
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

  /**
   * Mend a use case that failed, using the page as it was when it broke.
   * One LLM call; the result is a new draft version awaiting review.
   */
  repairUseCase: (id: string, payload: { execution_id?: string; run_id?: string }) =>
    request<{
      usecase_id: string;
      repaired: boolean;
      version?: number;
      diagnosis: string;
      confidence: string;
      applied?: string[];
      unfixable_reason?: string;
      llm_tokens: number;
    }>(`/api/usecases/${id}/repair`, { method: 'POST', body: JSON.stringify(payload) }),

  // --- batches ---------------------------------------------------------------

  startBatch: (
    id: string,
    payload: {
      csv?: string;
      /** A base64 .xlsx, for people who keep their records in a spreadsheet. */
      xlsx_base64?: string;
      sheet?: string;
      credential_id?: string | null;
      headless?: boolean;
    },
  ) =>
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
  // The token goes in the query string because a browser cannot set an
  // Authorization header on a WebSocket handshake. It is a short-lived,
  // revocable session token rather than a long-lived key, which is what makes
  // that acceptable -- see session.ts.
  url.searchParams.set('token', session.token ?? '');
  return url.toString();
}
