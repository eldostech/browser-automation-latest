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

/** One entry in the audit trail. */
export interface AuditEntry {
  id: number;
  actor_id: string | null;
  actor_email: string;
  action: string;
  resource_type: string;
  resource_id: string;
  detail: Record<string, unknown>;
  created_at: string;
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
  /** Who started it. Denormalized, so it outlives the account. */
  owner_id: string | null;
  owner_email: string;
}

export interface RunDetail extends RunSummary {
  active: boolean;
  pending_approval: ApprovalRequiredEvent | null;
  artifacts: { id: string; kind: string; mime: string; url: string }[];
}

/** A name, and the address it means in this deployment. */
export interface Target {
  id: string;
  name: string;
  base_url: string;
  description: string;
  updated_at: string;
  updated_by: string;
}

export interface ServerConfig {
  defaults: {
    browser: string;
    headless: boolean;
    trace: boolean;
    screenshots: string;
    healing: boolean;
  };
  /** What to call this deployment: "Dev", "UAT", "Production". Blank on a
   *  single-environment install, and then the badge is not shown. */
  environment: string;
  recorder: { enabled: boolean };
  /** The third deployment role. `reason` says why not, when it is off. */
  agent: { enabled: boolean; reason: string | null };
  model: string;
  /** Always "bedrock" — the only provider. */
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
  rejected_locators: Locator[];  /** `extract_rows` — the fields read out of each matched row. */
  columns?: { name: string; selector: string; attribute?: string }[];
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

/**
 * How much a model may do while a use case runs.
 *
 * `null` means the document never chose and follows the deployment. It is not
 * the same as "strict": a use case published before the choice existed was
 * being healed if the deployment allowed it, and must keep being.
 */
export type UseCaseMode = 'strict' | 'guided';

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
  /** Seconds between rows; null uses the deployment default. */
  row_delay_seconds: number | null;
  /** Which target supplies the base URL; "" means the recorded one. */
  target: string;
  /** How it runs; null follows the deployment. */
  mode: UseCaseMode | null;
  /** Which authoring path produced this document. */
  authored_by: 'person' | 'agent';
  /** The origin this was recorded against, used when no target is named. */
  base_url: string;
  warnings: string[];
  /** Recorded calls that did NOT become steps, and why. */
  dropped: string[];
  created_at: string;
  updated_at: string;
}

export interface UseCaseSummary {
  id: string;
  name: string;
  description: string;
  status: UseCaseStatus;
  /** Mirrored from the definition so a list needs no extra query. */
  mode: UseCaseMode | null;
  authored_by: 'person' | 'agent';
  current_version: number;
  source_run_id: string | null;
  created_at: string;
  updated_at: string;
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
  /** Who ran this row. Denormalized, so it survives the account's deletion. */
  owner_id: string | null;
  owner_email: string;
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

/**
 * What one column of an uploaded file holds, as far as the file can say.
 *
 * `kind` is inferred and never applied — the values themselves are always the
 * text that will be typed into the page. It is here so the mapper can refuse
 * to offer a date column for a number field, and so a person can see at a
 * glance which column is which.
 */
/** One line of a recording that could not be represented as a step. */
export interface UnsupportedLine {
  line: number;
  source: string;
  reason: string;
}

export interface RecordedStep {
  id: string;
  action: string;
  url: string | null;
  value: string | null;
  /** The first rung of the ladder, rendered for a human. */
  locator: string;
}

/**
 * A recording in progress, or one that has finished.
 *
 * `recording` means a browser window is open and the user is working in it.
 * Everything below `status` is present only once it is `ready`.
 */
export interface RecordingDetail {
  recording_id: string;
  status: 'recording' | 'parsing' | 'ready' | 'failed' | 'cancelled';
  name: string;
  start_url: string;
  started_at: number;
  finished_at: number | null;
  error: string | null;
  owner_email: string;
  summary?: string;
  domains?: string[];
  steps?: RecordedStep[];
  /** Values typed during the recording, for the user to name and classify. */
  typed?: string[];
  /** Elements pointed at with the recorder's assert buttons. Each is offered
   *  as either a check or a value to read into the results file. */
  captured?: {
    describe: string;
    label: string;
    value: string;
    kind: 'text' | 'value';
    line: number;
  }[];
  /** The same values, each with the control it went into. `label` is what was
   *  on screen; it is empty when the control had no accessible name, which is
   *  common for a custom dropdown. */
  values?: { value: string; action: string; label: string }[];
  unsupported?: UnsupportedLine[];
}

/**
 * One step of a finished run, read back as a row rather than as an event.
 *
 * `pixel_diff` is the fraction of the page that changed since the last run
 * that worked. `null` means there was nothing to compare against — a first
 * run, or a step that has never succeeded — which is a different thing from
 * `0` meaning nothing moved.
 */
export interface RunStep {
  id: string;
  run_id: string;
  seq: number;
  step_id: string;
  phase: string;
  action: string;
  locator: string;
  /** Which rung of the ladder matched. Above zero means the recording is drifting. */
  locator_rung: number | null;
  /** Where the step happened; the domain scopes any fix recorded from it. */
  page_url: string;
  status: 'succeeded' | 'failed' | 'skipped' | 'healed';
  duration_ms: number;
  error: string | null;
  row_index: number | null;
  screenshot_id: string | null;
  baseline_id: string | null;
  pixel_diff: number | null;
  /** The ratio in words, so nobody has to read four decimal places. */
  diff: string;
  screenshot_url: string | null;
  baseline_url: string | null;
  created_at: string;
}

/**
 * A locator that broke, and what fixed it.
 *
 * `confirmed_by` is `"model"` when healing worked it out alone, or an email
 * when a person did. That distinction is not decorative: a fix somebody looked
 * at is ranked above a closer match nobody checked when these are put in front
 * of the model.
 */
export interface RememberedFix {
  id: string;
  usecase_id: string | null;
  domain: string;
  step_id: string;
  error_kind: string;
  old_locator: { name?: string; selector?: string } | null;
  new_locator: { name?: string; selector?: string } | null;
  explanation: string;
  confirmed_by: string;
  created_at: string;
}

export interface ColumnProfile {
  name: string;
  kind: 'text' | 'integer' | 'number' | 'date' | 'boolean' | 'empty';
  /** A recognised value shape, where every populated value has it. */
  shape: 'email' | 'url' | 'phone' | 'date' | null;
  non_null: number;
  nulls: number;
  distinct: number;
  examples: string[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  filename: string;
  source: string;
  row_count: number;
  columns: ColumnProfile[];
  /** A preview, not the file. A listing returns none of these. */
  sample: Record<string, string>[];
  warnings: string[];
  created_at: string;
  owner_email: string;
}

/** One declared input, and what the dataset offers for it. */
export interface MappingSuggestion {
  field: string;
  column: string | null;
  score: number;
  /** Above the confidence threshold — offered pre-selected rather than as a guess. */
  confident: boolean;
  reason: string;
  alternatives: { column: string; score: number; reason: string }[];
}

export interface MappingResult {
  usecase_id: string;
  dataset_id: string;
  suggestions: MappingSuggestion[];
  /** Fields whose match is absent or too close to call. */
  unresolved: string[];
  columns: string[];
}

export interface CredentialSummary {
  id: string;
  name: string;
  slots: string[];
  created_at: string;
  last_used_at: string | null;
}

// ---------------------------------------------------------------------------
// The agent
// ---------------------------------------------------------------------------

/** Everything a session needs to start. Budgets are per-session because they
 *  are a judgement: exploring an unfamiliar site is worth more steps than
 *  re-recording one somebody already knows. */
export interface StartAgentSession {
  task: string;
  target?: string;
  start_url?: string;
  name?: string;
  credential_id?: string | null;
  sample?: Record<string, string>;
  may_write?: boolean;
  budget_steps?: number;
  budget_usd?: number;
}

/** What a session cost so far. */
export interface AgentSpend {
  steps: number;
  tokens: number;
  usd: number;
  llm_calls: number;
  seconds: number;
}

/** Whether the distilled draft replays. The whole point of the phase: the
 *  agent does not get to claim it recorded something. */
export interface AgentVerification {
  ran: boolean;
  ok: boolean;
  failed_step: string;
  error: string;
  outputs: Record<string, unknown>;
  duration_ms: number;
  skipped: string;
}

export interface AgentSessionDetail {
  id: string;
  run_id: string;
  task: string;
  start_url: string;
  /** running · awaiting_approval · succeeded · partial · failed · cancelled */
  status: string;
  error: string | null;
  /** Set while it is stopped, waiting for a person. */
  awaiting: {
    approval_id: string;
    call: { id: string; name: string; input: Record<string, unknown> };
    categories: string;
  } | null;
  summary: string;
  stopped_by: string;
  spend: Partial<AgentSpend>;
  steps: number;
  marks: { kind: string; name: string; after_call: number }[];
  unfinished: string;
  use_case: UseCase | null;
  draft_warnings: string[];
  verification: Partial<AgentVerification>;
}


/** What a workspace has spent with a model this month, and its ceiling.
 *  `limit_usd` null means no ceiling was ever set, which is not the same as a
 *  ceiling of zero. */
export interface WorkspaceSpend {
  usd: number;
  tokens: number;
  runs: number;
  since: string;
  limit_usd: number | null;
  remaining_usd: number | null;
}
