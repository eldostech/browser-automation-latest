import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { RunStatus, RunSummary } from '../lib/events';
import { formatDuration, formatRelative, truncate } from '../lib/format';
import { StatusBadge } from './StatusBadge';

interface Props {
  onOpen: (runId: string) => void;
}

const FILTERS: { label: string; value: RunStatus | null }[] = [
  { label: 'All', value: null },
  { label: 'Running', value: 'running' },
  { label: 'Needs approval', value: 'awaiting_approval' },
  { label: 'Succeeded', value: 'succeeded' },
  { label: 'Failed', value: 'failed' },
  { label: 'Cancelled', value: 'cancelled' },
];

export function RunHistory({ onOpen }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [filter, setFilter] = useState<RunStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.listRuns(filter ?? undefined);
      setRuns(response.runs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  // Keep the list fresh while runs are in flight, without a websocket per row.
  useEffect(() => {
    const active = runs.some((run) => ['pending', 'running', 'awaiting_approval'].includes(run.status));
    if (!active) return;
    const timer = window.setInterval(load, 4000);
    return () => window.clearInterval(timer);
  }, [runs, load]);

  return (
    <div className="card">
      <h2>Run history</h2>
      <p className="hint">
        Every run, its events and its screenshots are persisted, so history survives a backend
        restart. Open any run to replay it step by step.
      </p>

      <div className="filters">
        {FILTERS.map((option) => (
          <button
            type="button"
            key={option.label}
            className={filter === option.value ? 'active' : ''}
            onClick={() => setFilter(option.value)}
          >
            {option.label}
          </button>
        ))}
        <button type="button" onClick={load} style={{ marginLeft: 'auto' }} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && <div className="banner error">{error}</div>}

      {runs.length === 0 && !loading ? (
        <div className="empty-state">No runs{filter ? ` with status "${filter}"` : ''} yet.</div>
      ) : (
        <table className="runs">
          <thead>
            <tr>
              <th>Status</th>
              <th>Task</th>
              <th>Started</th>
              <th>Steps</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td>
                  <StatusBadge status={run.status} />
                </td>
                <td className="task-cell">
                  <button type="button" onClick={() => onOpen(run.id)}>
                    {truncate(run.task, 140)}
                  </button>
                  {run.error && (
                    <div style={{ color: 'var(--danger)', fontSize: 12, marginTop: 4 }}>
                      {truncate(run.error, 160)}
                    </div>
                  )}
                </td>
                <td style={{ color: 'var(--text-faint)', whiteSpace: 'nowrap' }}>
                  {formatRelative(run.started_at ?? run.created_at)}
                </td>
                <td style={{ fontFamily: 'var(--mono)' }}>{run.steps}</td>
                <td style={{ fontFamily: 'var(--mono)', whiteSpace: 'nowrap' }}>
                  {formatDuration(run.duration_ms)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
