/**
 * Record a workflow by doing it once.
 *
 * Three states, and the middle one is the point: a browser window is open and
 * the user is working in it, not here. This screen waits, then shows what was
 * captured and asks the one question the recording cannot answer itself —
 * which of the values they typed are per-row inputs, and which are the login.
 *
 * That question is asked *after* the recording rather than before, because
 * before it the user has no idea what they are about to type.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import type { RecordingDetail } from '../lib/events';

type Props = {
  onSaved: (usecaseId: string) => void;
  /** Back to the fork. Absent when this screen is reached directly. */
  onBack?: () => void;
};

type FieldChoice = {
  value: string;
  /** What was on screen at the control this went into; "" when it had none. */
  label: string;
  /** "fill" or "select" -- a dropdown is worth saying so in the review table. */
  action: string;
  /** Which of the recording's typed values this is. Sent with the field: two
   *  boxes filled with the same text are indistinguishable by value alone. */
  index: number;
  name: string;
  secret: boolean;
  use: boolean;
};

/** Trim a label down to something usable as a field name. */
function fromLabel(label: string): string {
  const cleaned = label
    // Placeholders are commonly written as a worked example -- "e.g. Legacy CRM
    // to Salesforce" -- and the example is not the field's name.
    .replace(/^\s*(e\.g\.|eg\.|example:|for example)\s*/i, '')
    .replace(/[.…]+$/, '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (!cleaned) return '';
  // A whole sentence is not a name either. The first few words carry it.
  return cleaned.split('_').slice(0, 3).join('_');
}

/** A sensible field name from a typed value, for the user to correct. */
function suggestName(value: string, index: number): string {
  if (/^[^@\s]+@[^@\s]+$/.test(value)) return 'email';
  if (/^https?:\/\//i.test(value)) return 'url';
  return `value_${index + 1}`;
}

/** Values that look like a credential, pre-marked as one. */
function looksSecret(value: string): boolean {
  // No spaces, mixed character classes, and not an address or URL: the shape
  // of a password rather than of a record field. The user corrects it either
  // way — this only decides which checkbox starts ticked.
  //
  // The address test is the full email shape, not merely "contains an @".
  // Rejecting every value with an @ in it excluded a large share of real
  // passwords, which is precisely backwards for a check whose whole job is to
  // spot one — `Autumn@2026` was read as a record field.
  if (/\s/.test(value) || /^https?:\/\//i.test(value)) return false;
  if (/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(value)) return false;
  return value.length >= 8 && /[^a-zA-Z0-9]/.test(value) && /\d/.test(value);
}

export function RecordWorkflow({ onSaved, onBack }: Props) {
  const [available, setAvailable] = useState<{ ok: boolean; reason: string } | null>(null);
  const [startUrl, setStartUrl] = useState('');
  const [name, setName] = useState('');
  const [recording, setRecording] = useState<RecordingDetail | null>(null);
  const [fields, setFields] = useState<FieldChoice[]>([]);
  // Elements pointed at with the recorder's assert buttons, and what to call
  // each in the results file. Empty name means "leave it as a check".
  const [reads, setReads] = useState<Record<number, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const poll = useRef<number | null>(null);

  useEffect(() => {
    api
      .listRecordings()
      .then((body) => setAvailable({ ok: body.available, reason: body.reason }))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const stopPolling = useCallback(() => {
    if (poll.current !== null) {
      window.clearInterval(poll.current);
      poll.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  /** Watch until the window closes. The user is in the browser, not here. */
  const watch = useCallback(
    (recordingId: string) => {
      stopPolling();
      poll.current = window.setInterval(async () => {
        try {
          const body = await api.getRecording(recordingId);
          setRecording(body);
          if (body.status !== 'recording') {
            stopPolling();
            setReads(
              Object.fromEntries(
                (body.captured ?? []).map((c) => [c.line, fromLabel(c.label)]),
              ),
            );
            setFields(
              (body.values ?? (body.typed ?? []).map((value) => ({ value, action: 'fill', label: '' }))).map(
                (entry, index) => ({
                  value: entry.value,
                  index,
                  label: entry.label,
                  action: entry.action,
                  // What the person saw on the control beats anything derivable
                  // from the value: a dropdown records the option's underlying
                  // id, which on a real application is often a UUID.
                  name:
                    fromLabel(entry.label) ||
                    (entry.action === 'select'
                      ? `dropdown_${index + 1}`
                      : suggestName(entry.value, index)),
                  secret: looksSecret(entry.value),
                  use: true,
                }),
              ),
            );
          }
        } catch (e) {
          stopPolling();
          setError(e instanceof Error ? e.message : String(e));
        }
      }, 1000);
    },
    [stopPolling],
  );

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const started = await api.startRecording(startUrl, name);
      setRecording(started);
      watch(started.recording_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!recording) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await api.saveRecording(recording.recording_id, {
        name: name || recording.name,
        fields: fields
          .filter((f) => f.use && f.name.trim())
          .map((f) => ({
            name: f.name.trim(),
            value: f.value,
            index: f.index,
            secret: f.secret,
          })),
        extractions: Object.entries(reads)
          .filter(([, name]) => name.trim())
          .map(([line, name]) => ({ line: Number(line), name: name.trim() })),
      });
      onSaved(saved.usecase_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const discard = async () => {
    if (!recording) return;
    stopPolling();
    if (recording.status === 'recording') await api.cancelRecording(recording.recording_id);
    await api.discardRecording(recording.recording_id).catch(() => undefined);
    setRecording(null);
    setFields([]);
  };

  if (available && !available.ok) {
    return (
      <div className="card">
        <h3>
        Record a workflow
        {onBack && (
          <button type="button" className="linkish" style={{ marginLeft: 12 }} onClick={onBack}>
            Back
          </button>
        )}
      </h3>
        <p className="hint">{available.reason}</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3>
        Record a workflow
        {onBack && (
          <button type="button" className="linkish" style={{ marginLeft: 12 }} onClick={onBack}>
            Back
          </button>
        )}
      </h3>

      {error && <p className="error">{error}</p>}

      {!recording && (
        <>
          <p className="hint">
            A browser window opens. Do the task once, by hand, then close the window. Nothing is
            sent to a model — what you do is captured directly, so recording costs nothing.
          </p>
          <div className="hint callout">
            <strong>To pull a value out into a spreadsheet:</strong> in the recorder&rsquo;s own
            toolbar, click <strong>Assert text</strong> (or <strong>Assert value</strong> for
            something you typed into), then click the thing on the page you want. Do that for
            each value. Afterwards you name them, and each becomes a column in the results
            file.
          </div>
          <label className="field">
            <span>Start at</span>
            <input
              value={startUrl}
              placeholder="https://example.com/login"
              onChange={(e) => setStartUrl(e.target.value)}
            />
          </label>
          <label className="field">
            <span>Call it</span>
            <input
              value={name}
              placeholder="Submit orders"
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <button
            type="button"
            className="primary"
            disabled={busy || !startUrl.trim()}
            onClick={start}
          >
            {busy ? 'Opening…' : 'Open the browser and record'}
          </button>
        </>
      )}

      {recording?.status === 'recording' && (
        <>
          <p className="hint">
            The browser window is open. Do the task, then <strong>close the window</strong> to
            finish. This page updates on its own.
          </p>
          <button type="button" onClick={discard}>
            Cancel
          </button>
        </>
      )}

      {recording && recording.status === 'failed' && (
        <>
          <p className="error">{recording.error}</p>
          <button type="button" onClick={discard}>
            Start again
          </button>
        </>
      )}

      {recording?.status === 'ready' && (
        <>
          <p className="hint">
            Captured {recording.summary}. Visiting: {(recording.domains ?? []).join(', ')}.
          </p>

          <ol className="recorded-steps">
            {(recording.steps ?? []).map((step) => (
              <li key={step.id}>
                <code>{step.action}</code> {step.url ?? step.locator}
                {step.value ? <span className="typed"> = {step.value}</span> : null}
              </li>
            ))}
          </ol>

          {(recording.unsupported ?? []).length > 0 && (
            <div className="hint warning">
              <p>These recorded lines could not be represented and were left out:</p>
              <ul>
                {recording.unsupported!.map((item) => (
                  <li key={item.line}>
                    <code>{item.source}</code> — {item.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {(recording.captured ?? []).length > 0 && (
            <>
              <h4>What should it read?</h4>
              <p className="hint">
                You pointed at these while recording. Name the ones whose value you want in
                the results file &mdash; each becomes a column. Leave a name blank and it
                stays a check that the page still says what it said.
              </p>
              <table className="mapping">
                <thead>
                  <tr>
                    <th>Where</th>
                    <th>It said</th>
                    <th>Column name</th>
                  </tr>
                </thead>
                <tbody>
                  {recording.captured!.map((c) => (
                    <tr key={c.line}>
                      <td style={{ fontSize: 13 }}>
                        {c.label || <code>{c.describe}</code>}
                        {c.kind === 'value' && <span className="hint"> (a field)</span>}
                      </td>
                      <td>
                        <code>{c.value}</code>
                      </td>
                      <td>
                        <input
                          value={reads[c.line] ?? ''}
                          placeholder="leave blank to just check it"
                          onChange={(e) =>
                            setReads((r) => ({ ...r, [c.line]: e.target.value }))
                          }
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {fields.length > 0 && (
            <>
              <h4>What did you type?</h4>
              <p className="hint">
                Name each value so a spreadsheet column can fill it. Mark the login as a
                credential: those are stored encrypted, entered once per session rather than once
                per row, and never written into the workflow.
              </p>
              <p className="hint">
                A dropdown records the option&rsquo;s underlying id rather than the words you
                picked &mdash; that is what the page sends, and the visible text is not in the
                recording. Your spreadsheet will hold those ids.
              </p>
              <table className="mapping">
                <thead>
                  <tr>
                    <th>Use it</th>
                    <th>Where</th>
                    <th>You typed</th>
                    <th>Call it</th>
                    <th>Credential</th>
                  </tr>
                </thead>
                <tbody>
                  {fields.map((field, index) => (
                    <tr key={`${field.value}-${index}`}>
                      <td>
                        <input
                          type="checkbox"
                          checked={field.use}
                          onChange={(e) =>
                            setFields((f) =>
                              f.map((item, i) =>
                                i === index ? { ...item, use: e.target.checked } : item,
                              ),
                            )
                          }
                        />
                      </td>
                      <td className="where">
                        {field.label ? (
                          field.label
                        ) : (
                          // A control with no accessible name -- most custom
                          // dropdowns. Saying which one it was is the only help
                          // available, and it beats leaving the cell blank.
                          <span className="hint">
                            {field.action === 'select' ? 'dropdown' : 'field'} #{index + 1},
                            unnamed
                          </span>
                        )}
                      </td>
                      <td>
                        <code>{field.secret ? '•'.repeat(8) : field.value}</code>
                      </td>
                      <td>
                        <input
                          value={field.name}
                          disabled={!field.use}
                          onChange={(e) =>
                            setFields((f) =>
                              f.map((item, i) =>
                                i === index ? { ...item, name: e.target.value } : item,
                              ),
                            )
                          }
                        />
                      </td>
                      <td>
                        <input
                          type="checkbox"
                          checked={field.secret}
                          disabled={!field.use}
                          onChange={(e) =>
                            setFields((f) =>
                              f.map((item, i) =>
                                i === index ? { ...item, secret: e.target.checked } : item,
                              ),
                            )
                          }
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          <div className="row end">
            <button type="button" onClick={discard} disabled={busy}>
              Throw it away
            </button>
            <button type="button" className="primary" onClick={save} disabled={busy}>
              {busy ? 'Saving…' : 'Save as a draft workflow'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
