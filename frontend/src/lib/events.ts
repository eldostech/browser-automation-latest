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
