/**
 * TypeScript mirror of `backend/events.py`.
 *
 * Keep the two in sync: `backend/tests/test_events.py` fails if a `type`
 * literal exists on the Python side but not here.
 */

export type RunStatus =
  | 'pending'
  | 'running'
  | 'awaiting_approval'
  | 'succeeded'
  | 'failed'
  | 'cancelled';

export const TERMINAL_STATUSES: RunStatus[] = ['succeeded', 'failed', 'cancelled'];

export interface BaseEvent {
  run_id: string;
  /** Monotonic per run. Also the resume token for WebSocket reconnection. */
  seq: number;
  ts: string;
}

export interface RunStartedEvent extends BaseEvent {
  type: 'run_started';
  task: string;
  start_url: string | null;
  options: Record<string, unknown>;
  tools: string[];
}

export interface ThinkingEvent extends BaseEvent {
  type: 'thinking';
  step: number;
  text: string;
  /** False while the model is still streaming this block. */
  done: boolean;
}

export interface ToolCallEvent extends BaseEvent {
  type: 'tool_call';
  step: number;
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  sensitive: boolean;
  approved_by_human: boolean;
}

export interface ToolResultEvent extends BaseEvent {
  type: 'tool_result';
  step: number;
  call_id: string;
  name: string;
  ok: boolean;
  duration_ms: number;
  text: string;
  truncated: boolean;
  attempts: number;
}

export interface ScreenshotEvent extends BaseEvent {
  type: 'screenshot';
  step: number;
  artifact_id: string;
  url: string;
  caption: string | null;
  page_url: string | null;
}

/** Replay only. A recorded use-case step is about to run. */
export interface StepStartedEvent extends BaseEvent {
  type: 'step_started';
  step: number;
  step_id: string;
  action: string;
  description: string;
  phase: 'setup' | 'row' | 'reset' | 'teardown';
}

export interface StepFinishedEvent extends BaseEvent {
  type: 'step_finished';
  step: number;
  step_id: string;
  ok: boolean;
  duration_ms: number;
  /** Which rung of the locator ladder matched; later rungs mean the site drifted. */
  matched_locator: string | null;
  locator_rung: number | null;
  assertion: string | null;
  message: string;
  skipped: boolean;
}

export interface ApprovalRequiredEvent extends BaseEvent {
  type: 'approval_required';
  step: number;
  approval_id: string;
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  reason: string;
  categories: string[];
  expires_at: string;
}

export interface ApprovalResolvedEvent extends BaseEvent {
  type: 'approval_resolved';
  step: number;
  approval_id: string;
  decision: 'approved' | 'rejected' | 'timeout';
  note: string | null;
}

export interface ErrorEvent extends BaseEvent {
  type: 'error';
  step: number | null;
  kind: string;
  message: string;
  recoverable: boolean;
  artifact_id: string | null;
  detail: Record<string, unknown>;
}

export interface RunFinishedEvent extends BaseEvent {
  type: 'run_finished';
  status: RunStatus;
  steps: number;
  duration_ms: number;
  summary: string | null;
  result: { answer?: string; data?: unknown } | null;
  error: string | null;
}

export type AgentEvent =
  | RunStartedEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | ScreenshotEvent
  | StepStartedEvent
  | StepFinishedEvent
  | ApprovalRequiredEvent
  | ApprovalResolvedEvent
  | ErrorEvent
  | RunFinishedEvent;

export type AgentEventType = AgentEvent['type'];

export const EVENT_TYPES: AgentEventType[] = [
  'run_started',
  'thinking',
  'tool_call',
  'tool_result',
  'screenshot',
  'step_started',
  'step_finished',
  'approval_required',
  'approval_resolved',
  'error',
  'run_finished',
];

/**
 * The socket also carries transport frames (heartbeats) that are not agent
 * events. They are distinguished by a `__` prefix and dropped here.
 */
export function isAgentEvent(message: unknown): message is AgentEvent {
  if (typeof message !== 'object' || message === null) return false;
  const type = (message as { type?: unknown }).type;
  return typeof type === 'string' && !type.startsWith('__') &&
    EVENT_TYPES.includes(type as AgentEventType);
}

export interface RunSummary {
  id: string;
  task: string;
  start_url: string | null;
  status: RunStatus;
  options: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  steps: number;
  duration_ms: number | null;
  summary: string | null;
  result: { answer?: string; data?: unknown } | null;
  error: string | null;
}

export interface RunDetail extends RunSummary {
  active: boolean;
  pending_approval: ApprovalRequiredEvent | null;
  artifacts: { id: string; kind: string; mime: string; url: string }[];
}

export interface ServerConfig {
  defaults: {
    max_steps: number;
    timeout_seconds: number;
    allowed_domains: string[];
    require_approval: boolean;
    screenshot_every_step: boolean;
    headless: boolean;
    browser: string;
  };
  model: string;
  /** "bedrock" or "anthropic" — which backend provider is serving the model. */
  provider: string;
  transport: string;
}

// ---------------------------------------------------------------------------
// Use cases: a recorded run, replayable without an LLM
// ---------------------------------------------------------------------------

export type UseCaseStatus = 'draft' | 'ready' | 'archived';

/** One rung of the locator ladder, most durable first. */
export interface Locator {
  strategy: 'role' | 'css' | 'text' | 'nth';
  role?: string | null;
  name?: string | null;
  selector?: string | null;
  text?: string | null;
  nth?: number;
}

export interface Assertion {
  kind: 'url_contains' | 'text_present' | 'element_visible' | 'element_count' | 'title_contains';
  value?: string | null;
  count?: number | null;
  locator?: Locator | null;
  negate?: boolean;
  timeout_ms?: number;
}

export interface FormFieldSpec {
  name: string;
  value: string;
  type: string;
  locators: Locator[];
}

export interface UseCaseStep {
  id: string;
  action: string;
  description: string;
  locators: Locator[];
  url?: string | null;
  value?: string | null;
  fields: FormFieldSpec[];
  output?: string | null;
  code?: string | null;
  assert?: Assertion | null;
  wait_for?: Record<string, unknown> | null;
  optional: boolean;
  on_failure: 'abort' | 'continue' | 'heal';
  timeout_ms: number;
  rejected_locators: Locator[];
}

export interface InputSpec {
  name: string;
  type: 'string' | 'url' | 'number' | 'boolean' | 'file';
  required: boolean;
  default?: unknown;
  description: string;
  example: string;
}

export interface SecretSpec {
  name: string;
  required: boolean;
  description: string;
}

export interface UseCase {
  schema_version: number;
  id: string;
  name: string;
  description: string;
  status: UseCaseStatus;
  version: number;
  source_run_id: string | null;
  allowed_domains: string[];
  /** Raw-JavaScript steps refuse to run until a person turns this on. */
  allow_scripts: boolean;
  inputs: InputSpec[];
  secrets: SecretSpec[];
  /** Runs once per batch — the sign-in. */
  setup_steps: UseCaseStep[];
  session_check: Assertion | null;
  row_reset: UseCaseStep | null;
  /** Runs once per input row. */
  row_steps: UseCaseStep[];
  teardown_steps: UseCaseStep[];
  outputs: string[];
  warnings: string[];
  created_at: string;
  updated_at: string;
}

export interface UseCaseSummary {
  id: string;
  name: string;
  description: string;
  status: UseCaseStatus;
  current_version: number;
  source_run_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface DistillResult {
  usecase_id: string;
  version: number;
  name: string;
  /** What the model proposed. Editable before you move on. */
  suggested_name: string;
  status: UseCaseStatus;
  warnings: string[];
  setup_steps: number;
  row_steps: number;
  inputs: string[];
  secrets: string[];
  blocked_scripts: string[];
}

export interface ExecutionRecord {
  id: string;
  batch_id: string | null;
  usecase_id: string;
  version: number;
  run_id: string | null;
  row_index: number | null;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown> | null;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  failed_step_id: string | null;
  error: string | null;
  /** Always 0 for a replay. That is the point of the feature. */
  llm_calls: number;
  llm_tokens: number;
  duration_ms: number | null;
  created_at: string;
}

export interface BatchSummary {
  id: string;
  usecase_id: string;
  version: number;
  status: string;
  total: number;
  succeeded: number;
  failed: number;
  credential_id: string | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface BatchDetail {
  batch: BatchSummary;
  executions: ExecutionRecord[];
  running: boolean;
  /** Rows never attempted — a stopped batch leaves these, and resume runs them. */
  pending: number;
}

export interface CredentialSummary {
  id: string;
  name: string;
  slots: string[];
  created_at: string;
  last_used_at: string | null;
}
