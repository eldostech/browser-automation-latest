/**
 * Upload a file, see what is in it, and say which column fills which field.
 *
 * This is the step that makes the product usable by someone who did not record
 * the workflow. The recording named its inputs; the spreadsheet names its
 * columns; nobody involved agreed on either. Joining them is the whole job.
 *
 * Two decisions worth keeping:
 *
 * **The mapping is shown even when it is obvious.** A confident match is
 * pre-selected and marked, so agreeing to it is one glance rather than one
 * decision per field — but it is still shown. A mapping that is wrong and
 * unreviewed does not fail, it succeeds into the wrong fields a thousand
 * times, and this screen is the only place anyone would notice.
 *
 * **The choice is a dropdown of real columns, never free text.** The user is
 * picking from what the file actually contains, which is the same reason the
 * healer picks page elements by index rather than inventing a selector.
 */

import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { ColumnProfile, DatasetSummary, MappingSuggestion, UseCase } from '../lib/events';
import { BrandSpinner } from './BrandSpinner';

type Props = {
  useCase: UseCase;
  usecaseId: string;
  /** Called with the confirmed mapping when the user is ready to run. */
  onReady: (dataset: DatasetSummary, mapping: Record<string, string>) => void;
  /** How many rows are about to run, the moment that is known. The screen
   *  above uses it to price the batch before anybody presses the button. */
  onRowCount?: (rows: number) => void;
  busy?: boolean;
  disabled?: boolean;
  disabledReason?: string;
};

function describe(column: ColumnProfile): string {
  const parts: string[] = [column.shape ?? column.kind];
  if (column.nulls) parts.push(`${column.nulls} blank`);
  if (column.distinct) parts.push(`${column.distinct} distinct`);
  return parts.join(' · ');
}

export function DatasetMapper({
  useCase,
  usecaseId,
  onReady,
  onRowCount,
  busy = false,
  disabled = false,
  disabledReason,
}: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [suggestions, setSuggestions] = useState<MappingSuggestion[]>([]);
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setDatasets((await api.listDatasets()).datasets);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  /** Ask for a mapping and pre-select whatever came back confident. */
  const chooseDataset = useCallback(
    async (summary: DatasetSummary) => {
      setLoading(true);
      setError(null);
      try {
        const full = await api.getDataset(summary.id);
        const result = await api.suggestMapping(usecaseId, summary.id);
        setDataset(full);
        onRowCount?.(full.row_count ?? 0);
        setSuggestions(result.suggestions);
        setChosen(
          Object.fromEntries(
            result.suggestions
              .filter((s) => s.column !== null)
              .map((s) => [s.field, s.column as string]),
          ),
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [usecaseId],
  );

  const upload = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const uploaded = await api.uploadDataset(file, file.name);
      await refresh();
      await chooseDataset(uploaded);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  };

  const required = useCase.inputs.filter((input) => input.required).map((input) => input.name);
  const unmapped = required.filter((name) => !chosen[name]);
  const columns = dataset?.columns.map((column) => column.name) ?? [];

  return (
    <div className="dataset-mapper">
      <label className="file-drop">
        <input
          type="file"
          accept=".csv,.txt,.tsv,.xlsx,.xlsm,text/csv,text/plain"
          disabled={busy || loading}
          onChange={async (event) => {
            const file = event.target.files?.[0];
            // Cleared so choosing the same file twice still fires a change.
            event.target.value = '';
            if (file) await upload(file);
          }}
        />
        <span>Upload a CSV, spreadsheet or text file</span>
      </label>

      {datasets.length > 0 && (
        <div className="dataset-list">
          <span className="hint">or reuse a file you have already uploaded:</span>
          {datasets.map((summary) => (
            <button
              key={summary.id}
              type="button"
              className={dataset?.id === summary.id ? 'chip active' : 'chip'}
              disabled={busy || loading}
              onClick={() => void chooseDataset(summary)}
            >
              {summary.name} · {summary.row_count} rows
            </button>
          ))}
        </div>
      )}

      {error && <p className="error">{error}</p>}
      {loading && (
        <BrandSpinner
          state="validating"
          label="Reading the file and matching its columns…"
          detail="Tries obvious matches first; a model is only asked about the columns that stay unclear."
        />
      )}

      {dataset && (
        <>
          {dataset.warnings.map((warning) => (
            <p className="hint warning" key={warning}>
              {warning}
            </p>
          ))}

          <h4>
            {dataset.name} — {dataset.row_count} rows
          </h4>

          <table className="mapping">
            <thead>
              <tr>
                <th>Field the workflow needs</th>
                <th>Column from your file</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {suggestions.map((suggestion) => {
                const column = chosen[suggestion.field] ?? '';
                const profile = dataset.columns.find((c) => c.name === column);
                const isRequired = required.includes(suggestion.field);
                return (
                  <tr key={suggestion.field} className={!column && isRequired ? 'missing' : ''}>
                    <td>
                      <code>{suggestion.field}</code>
                      {isRequired && <span className="req"> required</span>}
                    </td>
                    <td>
                      <select
                        value={column}
                        disabled={busy}
                        onChange={(event) =>
                          setChosen((previous) => {
                            const next = { ...previous };
                            if (event.target.value) next[suggestion.field] = event.target.value;
                            else delete next[suggestion.field];
                            return next;
                          })
                        }
                      >
                        <option value="">— not in this file —</option>
                        {columns.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                      {profile && <div className="hint">{describe(profile)}</div>}
                      {profile && profile.examples.length > 0 && (
                        <div className="hint examples">e.g. {profile.examples.join(', ')}</div>
                      )}
                    </td>
                    <td className="hint">
                      {/* An unedited confident guess says so; anything else is
                          shown as what it is, a guess to check. */}
                      {column === suggestion.column
                        ? `${suggestion.confident ? '' : 'Best guess: '}${suggestion.reason}`
                        : 'You chose this.'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {unmapped.length > 0 && (
            <p className="hint warning">
              Still to map: {unmapped.join(', ')}. Every required field needs a column before this
              can run.
            </p>
          )}

          <button
            type="button"
            className="primary"
            disabled={busy || disabled || unmapped.length > 0}
            onClick={() => onReady(dataset, chosen)}
            title={disabled ? disabledReason : undefined}
          >
            {busy ? <BrandSpinner state="working" label="Starting…" /> : `Start ${dataset.row_count} rows`}
          </button>
          {disabled && disabledReason && <p className="hint">{disabledReason}</p>}
        </>
      )}
    </div>
  );
}
