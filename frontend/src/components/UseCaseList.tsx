import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { UseCaseSummary } from '../lib/events';
import { formatRelative, truncate } from '../lib/format';

interface Props {
  onOpen: (usecaseId: string) => void;
}

const FILTERS: { label: string; value: string | null }[] = [
  { label: 'All', value: null },
  { label: 'Needs review', value: 'draft' },
  { label: 'Ready', value: 'ready' },
  { label: 'Archived', value: 'archived' },
];

/** Recorded use cases: the things that run without an LLM. */
export function UseCaseList({ onOpen }: Props) {
  const [rows, setRows] = useState<UseCaseSummary[]>([]);
  const [filter, setFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.listUseCases(filter ?? undefined);
      setRows(response.usecases);
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

  return (
    <div className="card">
      <h2>Use cases</h2>
      <p className="hint">
        A use case is a successful run recorded as repeatable steps. Running one costs{' '}
        <strong>no LLM tokens at all</strong> — give it inputs, or a CSV of them, and it replays.
        Record one from any succeeded run in <em>History</em>.
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

      {rows.length === 0 && !loading ? (
        <div className="empty-state">
          No use cases yet. Open a succeeded run and choose <strong>Save as use case</strong>.
        </div>
      ) : (
        <table className="runs">
          <thead>
            <tr>
              <th>Status</th>
              <th>Name</th>
              <th>Updated</th>
              <th>Version</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>
                  <span className={`badge ${row.status === 'ready' ? 'succeeded' : 'pending'}`}>
                    {row.status === 'draft' ? 'needs review' : row.status}
                  </span>
                </td>
                <td className="task-cell">
                  <button type="button" onClick={() => onOpen(row.id)}>
                    {truncate(row.name, 120)}
                  </button>
                  {row.description && (
                    <div style={{ color: 'var(--text-faint)', fontSize: 12, marginTop: 4 }}>
                      {truncate(row.description, 160)}
                    </div>
                  )}
                </td>
                <td style={{ color: 'var(--text-faint)', whiteSpace: 'nowrap' }}>
                  {formatRelative(row.updated_at)}
                </td>
                <td style={{ fontFamily: 'var(--mono)' }}>v{row.current_version}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
