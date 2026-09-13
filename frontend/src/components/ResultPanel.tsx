import { useState } from 'react';
import type { RunFinishedEvent } from '../lib/events';
import { formatDuration, prettyJson } from '../lib/format';
import { BrandSpinner } from './BrandSpinner';

interface Props {
  finished: RunFinishedEvent | null;
  running: boolean;
}

/** Extracted answer and any structured JSON the agent produced. */
export function ResultPanel({ finished, running }: Props) {
  const [copied, setCopied] = useState(false);
  const data = finished?.result?.data;
  const answer = finished?.result?.answer ?? finished?.summary ?? null;

  async function copy() {
    if (data === undefined || data === null) return;
    await navigator.clipboard.writeText(prettyJson(data));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="panel">
      <header>
        <span>Result</span>
        {finished && (
          <span style={{ marginLeft: 'auto' }}>
            {finished.steps} steps / {formatDuration(finished.duration_ms)}
          </span>
        )}
      </header>
      <div className="body">
        {!finished && running && (
          <BrandSpinner state="working" label="The result appears here when it finishes." />
        )}

        {!finished && !running && (
          <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>No result yet.</div>
        )}

        {finished?.error && <div className="banner error">{finished.error}</div>}

        {answer && <p className="result-answer">{answer}</p>}

        {data !== undefined && data !== null && (
          <>
            <pre>{prettyJson(data)}</pre>
            <button type="button" onClick={copy} style={{ marginTop: 10 }}>
              {copied ? 'Copied' : 'Copy JSON'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
