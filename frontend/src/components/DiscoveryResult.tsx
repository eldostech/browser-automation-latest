/**
 * What a discovery run found, and the one button that makes it useful.
 *
 * A migration against a vendor who will not open their back end runs in two
 * passes: the first walks their list pages and reads identifiers out of them,
 * the second runs once per identifier to pull the detail. This is the join.
 * Without it the first pass produces a few hundred rows of JSON in a table
 * cell that nobody can act on.
 *
 * It renders only when an output actually holds rows, so an ordinary run that
 * extracts three scalar fields never sees it.
 */

import { useState } from 'react';
import { api } from '../lib/api';

type Props = {
  executionId: string;
  outputs: Record<string, unknown>;
  /** So the operator can go straight from "saved" to running the second pass. */
  onSaved?: (datasetId: string) => void;
};

/** Outputs whose value is a list of objects — the shape `extract_rows` writes. */
function rowOutputs(outputs: Record<string, unknown>): [string, Record<string, unknown>[]][] {
  return Object.entries(outputs ?? {}).filter(
    (entry): entry is [string, Record<string, unknown>[]] =>
      Array.isArray(entry[1]) &&
      entry[1].length > 0 &&
      typeof entry[1][0] === 'object' &&
      entry[1][0] !== null,
  );
}

export function DiscoveryResult({ executionId, outputs, onSaved }: Props) {
  const found = rowOutputs(outputs);
  const [busy, setBusy] = useState('');
  const [saved, setSaved] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  if (found.length === 0) return null;

  const save = async (output: string) => {
    setBusy(output);
    setError(null);
    try {
      const result = await api.datasetFromRun({ execution_id: executionId, output });
      setSaved((s) => ({ ...s, [output]: result.dataset_id }));
      onSaved?.(result.dataset_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy('');
    }
  };

  return (
    <div className="card discovery">
      <h3>This run found rows</h3>
      <p className="hint">
        Save them as a dataset and a second use case can run once per row — which is how a
        migration works when the vendor will not give you their back end: one pass to find
        what exists, one to pull the detail.
      </p>

      {error && <div className="banner error">{error}</div>}

      {found.map(([output, rows]) => {
        const columns = Object.keys(rows[0] ?? {});
        const preview = rows.slice(0, 5);
        return (
          <div key={output} className="discovery-output">
            <div className="discovery-head">
              <div>
                <code>{output}</code>
                <span className="hint"> — {rows.length} row{rows.length === 1 ? '' : 's'}</span>
              </div>
              {saved[output] ? (
                <span className="tag">Saved as a dataset</span>
              ) : (
                <button
                  type="button"
                  className="primary"
                  disabled={busy === output}
                  onClick={() => void save(output)}
                >
                  {busy === output ? 'Saving…' : 'Save as a dataset'}
                </button>
              )}
            </div>
            <table className="mapping">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c} style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                        {String(row[c] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {rows.length > preview.length && (
              <p className="hint">
                Showing {preview.length} of {rows.length}. All of them go into the dataset.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** A downloaded document, as a `download` step records it. */
export type DownloadedFile = {
  filename: string;
  artifact_id: string;
  url: string;
  bytes: number;
};

function isDownload(value: unknown): value is DownloadedFile {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as DownloadedFile).filename === 'string' &&
    typeof (value as DownloadedFile).artifact_id === 'string'
  );
}

/** Every file a run pulled down, with what to call it. */
export function downloadsIn(
  outputs: Record<string, unknown> | null | undefined,
): [string, DownloadedFile][] {
  return Object.entries(outputs ?? {}).filter(
    (entry): entry is [string, DownloadedFile] => isDownload(entry[1]),
  );
}

function readableSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** The files a run kept, as links that save under the vendor's own name. */
export function Downloads({ outputs }: { outputs: Record<string, unknown> }) {
  const files = downloadsIn(outputs);
  if (files.length === 0) return null;

  return (
    <div className="card">
      <h3>Files this run kept</h3>
      <p className="hint">
        Stored where every other artifact goes — a directory locally, S3 in a cluster — under
        the name the vendor gave them, which is what the next system will expect.
      </p>
      <ul className="downloads">
        {files.map(([output, file]) => (
          <li key={output}>
            <a href={file.url}>{file.filename}</a>
            <span className="hint">
              {' '}
              {readableSize(file.bytes)} · <code>{output}</code>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A one-line summary for a table cell, so an output is not a wall of JSON. */
export function summariseOutputs(outputs: Record<string, unknown> | null | undefined): string {
  if (!outputs) return '';
  return Object.entries(outputs)
    .map(([key, value]) => {
      if (Array.isArray(value)) return `${key}: ${value.length} rows`;
      // A download is an object; its filename is the readable part, and
      // String(value) would print "[object Object]".
      if (isDownload(value)) return `${key}: ${value.filename}`;
      return `${key}=${String(value)}`;
    })
    .join(', ');
}
