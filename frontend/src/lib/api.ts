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
  BatchEstimate,
  AgentSessionDetail,
  AuditEntry,
  BatchDetail,
  BatchSummary,
  CredentialSummary,
  DatasetSummary,
  Locator,
  ModelCatalogue,
  LocatorCheckReport,
  MappingResult,
  RememberedFix,
  RunStep,
  RecordingDetail,
  ExecutionRecord,
  RunDetail,
  RunSummary,
  ServerConfig,
  StartAgentSession,
  Target,
  WorkspaceSpend,
  UseCase,
  UseCaseSummary,
} from './events';
import { modelFields } from './model';
import { session, type CurrentUser } from './session';

/** Empty by default: Vite (dev) and nginx (prod) proxy /api to the backend. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // A multipart body must set its own Content-Type, because only the browser
  // knows the boundary it generated. Sending application/json over a FormData
  // body produces a request the server cannot parse and an error that says
  // nothing about the cause.
  const isForm = init?.body instanceof FormData;
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    // Spread first, then set headers, so a caller passing its own headers
    // cannot accidentally drop the Authorization one.
    headers: {
      ...(isForm ? {} : { 'Content-Type': 'application/json' }),
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

  health: () => request<Record<string, unknown>>('/healthz'),

  // --- use cases ------------------------------------------------------------

  /** Promote a succeeded run into a reusable use case. The one LLM call. */
  /** Which credential slots this run still holds values for. Names only. */
  heldCredentialSlots: (runId: string) =>
    request<{ run_id: string; slots: string[] }>(`/api/runs/${runId}/credential-slots`),

  /** Who did what to one use case. Readable by anyone who can see it. */
  useCaseActivity: (id: string, limit = 100) =>
    request<{ usecase_id: string; entries: AuditEntry[] }>(
      `/api/usecases/${id}/activity?limit=${limit}`,
    ),

  /** The workspace-wide audit log. Administrators only. */
  auditLog: (params: { resource_type?: string; limit?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.resource_type) query.set('resource_type', params.resource_type);
    query.set('limit', String(params.limit ?? 200));
    return request<{ entries: AuditEntry[] }>(`/api/admin/audit?${query}`);
  },

  listUseCases: (status?: string) => {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    return request<{ usecases: UseCaseSummary[] }>(`/api/usecases?${params}`);
  },

  getUseCase: (id: string, version?: number) => {
    const params = new URLSearchParams();
    if (version) params.set('version', String(version));
    return request<{
      definition: UseCase;
      versions: { version: number; created_at: string }[];
      /** Resource-level state, separate from the definition a reviewer edits. */
      meta: { scripts_enabled: boolean; scripts_enabled_by: string | null } | null;
    }>(`/api/usecases/${id}?${params}`);
  },

  /**
   * Permit script steps on one use case. Administrators only.
   *
   * Separate from `allow_scripts` in the definition on purpose: that field is
   * part of a document an author can edit, so on its own it would let an author
   * grant themselves code execution. Both must agree before a script runs.
   */
  setScriptsEnabled: (id: string, enabled: boolean, reason = '') =>
    request<{ usecase_id: string; scripts_enabled: boolean; by: string }>(
      `/api/usecases/${id}/scripts`,
      { method: 'POST', body: JSON.stringify({ enabled, reason }) },
    ),

  /** Saves reviewer edits as a NEW version; existing versions are immutable. */
  updateUseCase: (id: string, definition: UseCase) =>
    request<{ usecase_id: string; version: number; status: string }>(`/api/usecases/${id}`, {
      method: 'PUT',
      body: JSON.stringify(definition),
    }),

  // -- models ---------------------------------------------------------------
  /** Every model this deployment can reach, with prices the provider publishes.
   *
   * Behind `usecase:read`: choosing which model to try is ordinary work for
   * anyone allowed to run a use case, and the reply carries no credential.
   */
  models: () => request<ModelCatalogue>('/api/models'),

  /** Can this deployment actually call that model? One tiny request.
   *
   * A key without credit, a model needing its own provider agreement, an id
   * that has been retired -- none of those are visible in a catalogue, and all
   * of them look identical to a broken workflow three steps into a run.
   */
  checkModel: (provider: string, model: string) =>
    request<{ provider: string; model: string; ok: boolean; error?: string }>(
      '/api/models/check',
      { method: 'POST', body: JSON.stringify({ provider, model }) },
    ),

  /** What these locators match on a real page, right now.
   *
   * Nothing is saved. This exists because editing a locator without it is
   * editing a string: a rung reads perfectly well and still matches nothing,
   * or matches four things, and the only way to find out used to be running
   * the use case and waiting out a timeout on row one of a batch.
   */
  checkLocators: (id: string, url: string, locators: Locator[], timeoutMs = 10000) =>
    request<LocatorCheckReport>(`/api/usecases/${id}/locator-check`, {
      method: 'POST',
      body: JSON.stringify({ url, locators, timeout_ms: timeoutMs }),
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
    }>(`/api/usecases/${id}/execute`, {
      method: 'POST',
      body: JSON.stringify({ ...payload, ...modelFields() }),
    }),

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
    }>(`/api/usecases/${id}/repair`, {
      method: 'POST',
      body: JSON.stringify({ ...payload, ...modelFields() }),
    }),

  /**
   * This use case as a Playwright Python script, to read or run elsewhere.
   *
   * One way only. The document is what TRACE runs and what a repair edits, so
   * an exported file that the platform read back would be a second source of
   * truth that drifts from the first.
   */
  exportPython: (id: string, version?: number) =>
    request<{ usecase_id: string; filename: string; script: string }>(
      `/api/usecases/${id}/export/python` + (version ? `?version=${version}` : ''),
    ),

  /**
   * Write down what a use case does, in plain language. One LLM call.
   *
   * An agent recording gets this at the end of its own session. This is the
   * same pass on demand, for a codegen recording -- which has no model in it
   * and so no account of itself -- and for anything recorded before this
   * existed. Saved at the status it already had: the prose changes nothing
   * that runs.
   */
  describeUseCase: (id: string) =>
    request<{
      usecase_id: string;
      version: number;
      status: string;
      instructions: string;
      steps_described: number;
      warnings: string[];
      llm_tokens: number;
    }>(`/api/usecases/${id}/describe`, {
      method: 'POST',
      body: JSON.stringify({ ...modelFields() }),
    }),

  /**
   * The finished step trail, with each step's screenshot and the baseline's.
   *
   * `first_divergence` is the step to open on: the first that failed or moved
   * more than a couple of percent. A thousand-row batch produces tens of
   * thousands of steps, and "scroll until something looks wrong" is not a
   * workflow.
   */
  getRunSteps: (runId: string) =>
    request<{ run_id: string; steps: RunStep[]; first_divergence: number | null }>(
      `/api/runs/${runId}/steps`,
    ),

  // --- healing memory --------------------------------------------------------

  listFixes: (domain?: string) =>
    request<{ fixes: RememberedFix[] }>(
      `/api/memory${domain ? `?domain=${encodeURIComponent(domain)}` : ''}`,
    ),

  /**
   * Record what a person worked out when the model was not sure enough.
   *
   * Attributed to them rather than to the model, which is what makes it
   * outrank an automatic fix next time the same thing breaks.
   */
  rememberFix: (payload: {
    step_id: string;
    page_url: string;
    page?: string;
    step_summary?: string;
    wanted?: string;
    explanation: string;
    usecase_id?: string | null;
    old_locator?: Record<string, unknown> | null;
    new_locator?: Record<string, unknown> | null;
  }) =>
    request<{ remembered: boolean; domain: string }>('/api/memory', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  forgetFix: (fixId: string) =>
    request<{ forgotten: string }>(`/api/memory/${fixId}`, { method: 'DELETE' }),

  // --- recording -------------------------------------------------------------

  /**
   * Open a browser window and record what the user does in it.
   *
   * No model is involved. The old recorder was an agent driving the browser to
   * work out how to do the task, which cost ~247,000 input tokens per
   * recording; the user already knows how, so the software watches instead.
   */
  startRecording: (startUrl: string, name: string) =>
    request<RecordingDetail>('/api/recordings', {
      method: 'POST',
      body: JSON.stringify({ start_url: startUrl, name }),
    }),

  listRecordings: () =>
    request<{ available: boolean; reason: string; recordings: RecordingDetail[] }>(
      '/api/recordings',
    ),

  getRecording: (recordingId: string) =>
    request<RecordingDetail>(`/api/recordings/${recordingId}`),

  cancelRecording: (recordingId: string) =>
    request<{ cancelled: boolean }>(`/api/recordings/${recordingId}/cancel`, { method: 'POST' }),

  discardRecording: (recordingId: string) =>
    request<{ discarded: string }>(`/api/recordings/${recordingId}`, { method: 'DELETE' }),

  /**
   * Turn a finished recording into a draft workflow.
   *
   * `fields` names the values that were typed and says which are credentials.
   * That answer is what decides the setup/per-row split as well: a credential
   * is the thing supplied once per session, so the last step that types one is
   * where signing in ends and the row work begins.
   */
  /**
   * Land a definition exported from another environment.
   *
   * Always arrives as a draft with `allow_scripts` off, whatever the source
   * said: an approval given in dev is not an approval here. The id is
   * preserved, so re-importing a revision appends a version rather than
   * duplicating the use case.
   */
  importUseCase: (definition: Record<string, unknown>) =>
    request<{ usecase_id: string; version: number; status: string; imported_by: string }>(
      '/api/usecases/import',
      { method: 'POST', body: JSON.stringify(definition) },
    ),

  /**
   * Turn what a discovery run extracted into a dataset the next pass can run on.
   *
   * The join between the two passes of a migration: the first walks the
   * vendor's list pages, this makes those rows runnable, the second pulls the
   * detail one row at a time.
   */
  datasetFromRun: (body: {
    execution_id?: string;
    batch_id?: string;
    output: string;
    name?: string;
  }) =>
    request<{ dataset_id: string; row_count: number; columns: { name: string }[] }>(
      '/api/datasets/from-run',
      { method: 'POST', body: JSON.stringify(body) },
    ),

  // --- targets ---------------------------------------------------------------

  listTargets: () => request<{ targets: Target[] }>('/api/targets'),

  /** Create or move one target. The name is the handle a use case refers to. */
  saveTarget: (name: string, baseUrl: string, description = '') =>
    request<Target>(`/api/targets/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify({ base_url: baseUrl, description }),
    }),

  deleteTarget: (name: string) =>
    request<{ deleted: string }>(`/api/targets/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  saveRecording: (
    recordingId: string,
    payload: {
      name?: string;
      description?: string;
      fields: { name: string; value: string; index?: number; secret: boolean }[];
      /** Pointed-at elements the person wants read into the results file. */
      extractions?: { line: number; name: string }[];
    },
  ) =>
    request<{
      usecase_id: string;
      version: number;
      status: string;
      /** Whether a field marked secret was written straight to the vault --
       *  never the value itself, just whether it happened. */
      credential_saved: boolean;
      credential_name: string | null;
    }>(`/api/recordings/${recordingId}/save`, { method: 'POST', body: JSON.stringify(payload) }),

  // --- datasets --------------------------------------------------------------

  /**
   * Upload a file of input rows.
   *
   * Sent as multipart rather than base64 in JSON: the file goes over the wire
   * once, at its own size, and the browser never has to hold a second copy of
   * it as a string. The server reads and profiles it before storing, so a file
   * that cannot be parsed is refused here rather than on row 700.
   */
  uploadDataset: (file: File, name?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (name) form.append('name', name);
    return request<DatasetSummary & { dataset_id: string }>('/api/datasets', {
      method: 'POST',
      body: form,
    });
  },

  listDatasets: () => request<{ datasets: DatasetSummary[] }>('/api/datasets'),

  getDataset: (datasetId: string, sample = 20) =>
    request<DatasetSummary>(`/api/datasets/${datasetId}?sample=${sample}`),

  deleteDataset: (datasetId: string) =>
    request<{ deleted: string }>(`/api/datasets/${datasetId}`, { method: 'DELETE' }),

  /**
   * Propose a column for each declared input.
   *
   * Nothing is committed by asking: the response is a ranking with reasons,
   * and the user confirms it. An automatic mapping that is wrong and
   * unreviewed does not fail — it succeeds into the wrong fields, a thousand
   * times, and nobody finds out from this application.
   */
  suggestMapping: (usecaseId: string, datasetId: string, version?: number) =>
    request<MappingResult>(`/api/usecases/${usecaseId}/mapping`, {
      method: 'POST',
      body: JSON.stringify({ dataset_id: datasetId, version }),
    }),

  // --- batches ---------------------------------------------------------------

  startBatch: (
    id: string,
    payload: {
      /** An uploaded dataset, with the mapping the user confirmed. */
      dataset_id?: string;
      mapping?: Record<string, string>;
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
      { method: 'POST', body: JSON.stringify({ ...payload, ...modelFields() }) },
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

  estimateBatch: (usecaseId: string, rows: number) =>
    request<BatchEstimate>(`/api/usecases/${usecaseId}/estimate?rows=${rows}`),

  getSpend: () => request<WorkspaceSpend>('/api/admin/spend'),

  setSpendLimit: (limitUsd: number | null) =>
    request<WorkspaceSpend>('/api/admin/spend/limit', {
      method: 'PUT',
      body: JSON.stringify({ limit_usd: limitUsd }),
    }),

  // --- the agent -----------------------------------------------------------
  // Deliberately the same shape as the recording endpoints: start, watch,
  // save. Two ways of producing the same artifact should read the same way.
  startAgentSession: (body: StartAgentSession) =>
    request<AgentSessionDetail>('/api/agent-sessions', {
      method: 'POST',
      body: JSON.stringify({ ...body, ...modelFields() }),
    }),

  getAgentSession: (sessionId: string) =>
    request<AgentSessionDetail>(`/api/agent-sessions/${sessionId}`),

  decideAgentSession: (sessionId: string, decision: 'approved' | 'rejected') =>
    request<{ decision: string }>(`/api/agent-sessions/${sessionId}/decide`, {
      method: 'POST',
      body: JSON.stringify({ decision }),
    }),

  cancelAgentSession: (sessionId: string) =>
    request<{ status: string }>(`/api/agent-sessions/${sessionId}/cancel`, {
      method: 'POST',
    }),

  saveAgentSession: (sessionId: string, name: string) =>
    request<{ usecase_id: string; version: number; warnings: string[] }>(
      `/api/agent-sessions/${sessionId}/save`,
      { method: 'POST', body: JSON.stringify({ name }) },
    ),
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
