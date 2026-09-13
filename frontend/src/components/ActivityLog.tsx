/**
 * Who did what, when — the audit trail, rendered.
 *
 * The trail existed in the database from the moment tenancy did, and nothing
 * ever showed it. An audit log nobody can read is a compliance artifact rather
 * than a working tool, so this is deliberately part of the ordinary use case
 * view and not buried in an admin screen.
 *
 * Two things every row must answer, because they are the questions actually
 * asked afterwards: *who* did this, and *what was it given*. The second is why
 * inputs are rendered rather than summarised — "it ran with the wrong data" is
 * only answerable if the data is visible.
 */

import { useEffect, useState } from 'react';

import { api } from '../lib/api';
import type { AuditEntry } from '../lib/events';
import { BrandSpinner } from './BrandSpinner';

interface Props {
  /** Omit to show the whole workspace log (administrators only). */
  usecaseId?: string;
  limit?: number;
}

/** Past-tense phrasing, so a row reads as a sentence about a person. */
const ACTION_LABELS: Record<string, string> = {
  'usecase.distill': 'recorded this use case',
  'usecase.edit': 'edited the steps',
  'usecase.publish': 'published it',
  'usecase.archive': 'archived it',
  'usecase.purge': 'deleted it permanently',
  'usecase.repair': 'repaired it with AI',
  'usecase.execute': 'ran it',
  'usecase.scripts_enabled': 'allowed it to run JavaScript',
  'usecase.scripts_disabled': 'withdrew permission to run JavaScript',
  'batch.start': 'started a batch',
  'credential.save': 'saved a credential',
  'credential.delete': 'deleted a credential',
  'run.approved': 'approved a step',
  'run.rejected': 'rejected a step',
  'user.create': 'created an account',
  'user.update': 'changed an account',
};

/** Actions worth colouring: they change what the system is permitted to do. */
const SENSITIVE = new Set([
  'usecase.scripts_enabled',
  'usecase.purge',
  'credential.save',
  'credential.delete',
  'user.create',
  'user.update',
]);

function describe(entry: AuditEntry): string {
  return ACTION_LABELS[entry.action] ?? entry.action.replace(/[._]/g, ' ');
}

function when(iso: string): string {
  const date = new Date(iso);
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
}

/** The parts of `detail` worth showing inline, in a stable order. */
function summarise(detail: Record<string, unknown>): string {
  const parts: string[] = [];
  if (detail.inputs && typeof detail.inputs === 'object') {
    const inputs = Object.entries(detail.inputs as Record<string, unknown>);
    if (inputs.length) {
      parts.push(inputs.map(([k, v]) => `${k}=${String(v)}`).join(', '));
    }
  }
  if (typeof detail.rows === 'number') parts.push(`${detail.rows} rows`);
  if (typeof detail.version === 'number') parts.push(`v${detail.version}`);
  if (typeof detail.status === 'string') parts.push(String(detail.status));
  if (typeof detail.reason === 'string' && detail.reason) parts.push(String(detail.reason));
  return parts.join(' · ');
}

export function ActivityLog({ usecaseId, limit = 100 }: Props) {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = usecaseId
      ? api.useCaseActivity(usecaseId, limit).then((body) => body.entries)
      : api.auditLog({ limit }).then((body) => body.entries);

    load
      .then((rows) => alive && setEntries(rows))
      .catch((err: Error) => alive && setError(err.message));
    return () => {
      alive = false;
    };
  }, [usecaseId, limit]);

  if (error) return <p className="hint">Could not load the activity: {error}</p>;
  if (entries === null) return <BrandSpinner state="working" label="Loading activity…" />;
  if (entries.length === 0) {
    return (
      <p className="hint">
        Nothing recorded yet. Every publish, run, repair and credential change appears
        here with who did it.
      </p>
    );
  }

  return (
    <table className="runs activity">
      <thead>
        <tr>
          <th>When</th>
          <th>Who</th>
          <th>What</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((entry) => (
          <tr key={entry.id} className={SENSITIVE.has(entry.action) ? 'sensitive-row' : ''}>
            <td style={{ whiteSpace: 'nowrap', fontSize: 12 }}>{when(entry.created_at)}</td>
            <td style={{ fontSize: 12 }}>
              {/* An entry whose account has since been deleted still names
                  them: the email is stored on the row, not joined. */}
              {entry.actor_email || <span className="hint">(system)</span>}
            </td>
            <td>{describe(entry)}</td>
            <td style={{ fontFamily: 'var(--mono)', fontSize: 12, wordBreak: 'break-word' }}>
              {summarise(entry.detail)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
