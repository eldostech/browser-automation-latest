import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import type { UseCaseSummary } from '../lib/events';
import { formatRelative, truncate } from '../lib/format';
import { BrandSpinner } from './BrandSpinner';

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
  // What the import said this environment still has to supply. Kept on screen
  // rather than announced and dismissed: a missing target is the difference
  // between a use case that runs here and one that runs against the
  // environment it came from.
  const [imported, setImported] = useState<{
    id: string;
    version: number;
    from: string;
    warnings: string[];
  } | null>(null);
  const file = useRef<HTMLInputElement>(null);

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

  const importFile = async (chosen: File) => {
    setError(null);
    setImported(null);
    try {
      const parsed = JSON.parse(await chosen.text()) as Record<string, unknown>;
      const result = await api.importUseCase(parsed);
      setImported({
        id: result.usecase_id,
        version: result.version,
        from: result.from,
        warnings: result.warnings ?? [],
      });
      await load();
    } catch (err) {
      setError(
        err instanceof SyntaxError
          ? `${chosen.name} is not valid JSON. Export it again from the other environment.`
          : err instanceof Error
            ? err.message
            : String(err),
      );
    } finally {
      // Cleared so choosing the same file twice fires again -- re-importing
      // after adding the missing target is the obvious next thing to do.
      if (file.current) file.current.value = '';
    }
  };

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
        <input
          ref={file}
          type="file"
          accept="application/json,.json"
          style={{ display: 'none' }}
          onChange={(e) => {
            const chosen = e.target.files?.[0];
            if (chosen) void importFile(chosen);
          }}
        />
        <button
          type="button"
          style={{ marginLeft: 'auto' }}
          onClick={() => file.current?.click()}
          title="Land a use case exported from another environment. It arrives as a draft."
        >
          Import
        </button>
        <button type="button" onClick={load} disabled={loading}>
          {loading ? <BrandSpinner state="working" label="Refreshing…" /> : 'Refresh'}
        </button>
      </div>

      {imported && (
        <div className={imported.warnings.length ? 'banner warn' : 'banner'}>
          <strong>
            Imported as v{imported.version}
            {imported.from ? ` from ${imported.from}` : ''}, as a draft.
          </strong>
          <p style={{ margin: '6px 0' }}>
            Publishing is per environment, and script permission never travels: both are
            decisions for this environment rather than inherited ones.
          </p>
          {imported.warnings.length > 0 && (
            <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
              {imported.warnings.map((line, index) => (
                <li key={index}>{line}</li>
              ))}
            </ul>
          )}
          <p style={{ margin: '6px 0 0' }}>
            <button type="button" className="link" onClick={() => onOpen(imported.id)}>
              Open it
            </button>
          </p>
        </div>
      )}

      {error && <div className="banner error">{error}</div>}

      {rows.length === 0 && loading ? (
        <BrandSpinner layout="block" state="working" label="Loading use cases…" />
      ) : rows.length === 0 && !loading ? (
        <div className="empty-state">
          No use cases yet. Open a succeeded run and choose <strong>Save as use case</strong>.
        </div>
      ) : (
        <table className="runs">
          <thead>
            <tr>
              <th>Status</th>
              <th>Name</th>
              <th>Runs as</th>
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
                <td style={{ whiteSpace: 'nowrap' }}>
                  <span
                    className="tag mode"
                    title={
                      row.mode
                        ? row.mode === 'strict'
                          ? 'No model can run on this use case.'
                          : 'A repair may run when a step stops matching.'
                        : 'Nothing chosen here, so it follows the deployment.'
                    }
                  >
                    {row.mode ?? 'deployment'}
                  </span>
                  {row.authored_by === 'agent' && (
                    <span className="tag" style={{ marginLeft: 6 }} title="Distilled from an agent session">
                      agent
                    </span>
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
