import type { RunStatus } from '../lib/events';

const LABELS: Record<RunStatus, string> = {
  pending: 'pending',
  running: 'running',
  awaiting_approval: 'needs approval',
  succeeded: 'succeeded',
  failed: 'failed',
  cancelled: 'cancelled',
};

export function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={`badge ${status}`}>
      <span className="dot" />
      {LABELS[status] ?? status}
    </span>
  );
}

export function ConnectionIndicator({ state }: { state: string }) {
  const label =
    state === 'open'
      ? 'live'
      : state === 'reconnecting'
        ? 'reconnecting...'
        : state === 'connecting'
          ? 'connecting...'
          : state === 'closed'
            ? 'stream closed'
            : 'idle';
  return (
    <span className={`conn ${state}`}>
      <span className="dot" />
      {label}
    </span>
  );
}
