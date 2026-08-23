import { useEffect, useState } from 'react';
import type { ApprovalRequiredEvent } from '../lib/events';
import { prettyJson } from '../lib/format';

interface Props {
  approval: ApprovalRequiredEvent;
  onDecide: (decision: 'approve' | 'reject', note?: string) => Promise<void>;
}

/**
 * A blocking action bar, deliberately not a modal.
 *
 * The agent loop is paused until someone answers, and a dialog that can be
 * dismissed by a stray click (or that opens behind another tab) is exactly the
 * wrong affordance for that. This sits inline at the top of the run view,
 * stays until it is resolved, and shows the countdown to auto-rejection.
 */
export function ApprovalBar({ approval, onDecide }: Props) {
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [remaining, setRemaining] = useState(() => secondsUntil(approval.expires_at));

  useEffect(() => {
    setRemaining(secondsUntil(approval.expires_at));
    const timer = window.setInterval(
      () => setRemaining(secondsUntil(approval.expires_at)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [approval.expires_at, approval.approval_id]);

  async function decide(decision: 'approve' | 'reject') {
    setBusy(true);
    try {
      await onDecide(decision, note.trim() || undefined);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="approval-bar" role="alertdialog" aria-live="assertive">
      <h3>Approval required before the agent continues</h3>
      <p className="reason">{approval.reason}</p>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
        {approval.categories.map((category) => (
          <span className="tag sensitive" key={category}>
            {category.replace(/_/g, ' ')}
          </span>
        ))}
      </div>

      <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: '#ffd479' }}>
        {approval.name}
      </div>
      <pre>{prettyJson(approval.arguments)}</pre>

      <input
        value={note}
        onChange={(event) => setNote(event.target.value)}
        placeholder="Optional note recorded with the decision"
        style={{ marginTop: 10 }}
      />

      <div className="actions">
        <button className="approve" type="button" disabled={busy} onClick={() => decide('approve')}>
          Approve and continue
        </button>
        <button className="reject" type="button" disabled={busy} onClick={() => decide('reject')}>
          Reject
        </button>
        <span className="countdown">
          {remaining > 0 ? `auto-rejects in ${formatCountdown(remaining)}` : 'expired'}
        </span>
      </div>
    </div>
  );
}

function secondsUntil(iso: string): number {
  const target = new Date(iso).getTime();
  if (Number.isNaN(target)) return 0;
  return Math.max(0, Math.round((target - Date.now()) / 1000));
}

function formatCountdown(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return minutes > 0 ? `${minutes}m ${rest}s` : `${rest}s`;
}
