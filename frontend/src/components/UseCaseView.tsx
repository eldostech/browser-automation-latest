import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, batchResultsUrl } from '../lib/api';
import type { BatchDetail, CredentialSummary, UseCase } from '../lib/events';
import { formatDuration } from '../lib/format';
import { CredentialsPanel } from './CredentialsPanel';
import { UseCaseSteps } from './UseCaseSteps';

interface Props {
  usecaseId: string;
  onBack: () => void;
  onOpenRun: (runId: string) => void;
}

type Mode = 'review' | 'single' | 'batch';

/**
 * Review a recorded use case, publish it, and run it.
 *
 * Review is not optional: distillation is a best guess over a noisy recording,
 * so a use case is created as a draft and only a person moves it to ready.
 */
export function UseCaseView({ usecaseId, onBack, onOpenRun }: Props) {
  const [useCase, setUseCase] = useState<UseCase | null>(null);
  const [credentials, setCredentials] = useState<CredentialSummary[]>([]);
  const [vaultAvailable, setVaultAvailable] = useState(true);
  const [mode, setMode] = useState<Mode>('review');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [credentialId, setCredentialId] = useState<string>('');
  const [csv, setCsv] = useState('');
  const [batch, setBatch] = useState<BatchDetail | null>(null);

  const load = useCallback(async () => {
    try {
      const [detail, creds] = await Promise.all([
        api.getUseCase(usecaseId),
        api.listCredentials(),
      ]);
      setUseCase(detail.definition);
      setCredentials(creds.credentials);
      setVaultAvailable(creds.vault_available);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [usecaseId]);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while a batch is in flight. One request every couple of seconds is
  // plenty for a job measured in minutes.
  useEffect(() => {
    if (!batch || batch.batch.finished_at) return;
    const timer = window.setInterval(async () => {
      try {
        setBatch(await api.getBatch(batch.batch.id));
      } catch {
        /* transient; the next tick retries */
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [batch]);

  const missingSlots = useMemo(() => {
    if (!useCase) return [];
    const chosen = credentials.find((c) => c.id === credentialId);
    const provided = new Set(chosen?.slots ?? []);
    return useCase.secrets.filter((s) => s.required && !provided.has(s.name)).map((s) => s.name);
  }, [useCase, credentials, credentialId]);

  const act = useCallback(
    async (fn: () => Promise<void>) => {
      setBusy(true);
      setError(null);
      setNotice(null);
      try {
        await fn();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const publish = () =>
    act(async () => {
      await api.publishUseCase(usecaseId);
      setNotice('Published. It can now be run against inputs.');
      await load();
    });

  const allowScripts = () =>
    act(async () => {
      if (!useCase) return;
      await api.updateUseCase(usecaseId, { ...useCase, allow_scripts: true });
      setNotice('Raw-JavaScript steps enabled for this use case.');
      await load();
    });

  const removeStep = (phase: 'setup_steps' | 'row_steps') => (stepId: string) =>
    act(async () => {
      if (!useCase) return;
      const next = { ...useCase, [phase]: useCase[phase].filter((s) => s.id !== stepId) };
      await api.updateUseCase(usecaseId, next as UseCase);
      setNotice(`Removed ${stepId}. Saved as a new version.`);
      await load();
    });

  const runOnce = () =>
    act(async () => {
      const result = await api.executeUseCase(usecaseId, {
        inputs,
        credential_id: credentialId || null,
      });
      setNotice(
        result.status === 'succeeded'
          ? `Succeeded using ${result.llm_tokens} LLM tokens. Outputs: ${JSON.stringify(result.outputs)}`
          : `Failed: ${result.error}`,
      );
      onOpenRun(result.run_id);
    });

  const runBatch = () =>
    act(async () => {
      const started = await api.startBatch(usecaseId, {
        csv,
        credential_id: credentialId || null,
      });
      setBatch(await api.getBatch(started.batch_id));
      setNotice(`Started ${started.total} rows on one shared browser session.`);
    });

  const resume = () =>
    act(async () => {
      if (!batch) return;
      const resumed = await api.resumeBatch(batch.batch.id, credentialId || null);
      setBatch(await api.getBatch(resumed.batch_id));
      setNotice(`Re-running ${resumed.rows} row(s) that had not succeeded.`);
    });

  if (!useCase) {
    return (
      <div className="card">
        <button type="button" onClick={onBack}>
          Back
        </button>
        {error ? <div className="banner error">{error}</div> : <p>Loading...</p>}
      </div>
    );
  }

  const isReady = useCase.status === 'ready';
  const blockedScripts = useCase.allow_scripts
    ? []
    : [...useCase.setup_steps, ...useCase.row_steps].filter((s) => s.action === 'script');

  return (
    <div>
      <div className="run-header">
        <button type="button" onClick={onBack}>
          Back
        </button>
        <div className="task">
          <div className="row">
            <span className={`badge ${isReady ? 'succeeded' : 'pending'}`}>
              {isReady ? 'ready' : 'needs review'}
            </span>
            <span style={{ fontSize: 12, color: 'var(--text-faint)' }}>
              v{useCase.version} · {useCase.setup_steps.length} setup ·{' '}
              {useCase.row_steps.length} per row
            </span>
          </div>
          <p>{useCase.name}</p>
        </div>
        {!isReady && (
          <button type="button" className="primary" onClick={publish} disabled={busy}>
            Publish
          </button>
        )}
      </div>

      {error && <div className="banner error">{error}</div>}
      {notice && <div className="banner">{notice}</div>}

      {useCase.warnings.length > 0 && (
        <div className="banner warn">
          <strong>Check these before publishing</strong>
          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
            {useCase.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {blockedScripts.length > 0 && (
        <div className="banner error">
          <strong>This use case contains raw JavaScript</strong>
          <p style={{ margin: '6px 0' }}>
            {blockedScripts.length} step(s) run arbitrary code against a live, signed-in session.
            They refuse to execute until you read the code below and enable them.
          </p>
          <button type="button" onClick={allowScripts} disabled={busy}>
            I have read the code — enable scripts
          </button>
        </div>
      )}

      <Blockers
        useCase={useCase}
        blockedScripts={blockedScripts.length}
        credentialCount={credentials.length}
        vaultAvailable={vaultAvailable}
      />

      <div className="filters" style={{ marginBottom: 16 }}>
        <button
          type="button"
          className={mode === 'review' ? 'active' : ''}
          onClick={() => setMode('review')}
        >
          Steps
        </button>
        <button
          type="button"
          className={mode === 'single' ? 'active' : ''}
          onClick={() => setMode('single')}
          disabled={!isReady}
        >
          Run one
        </button>
        <button
          type="button"
          className={mode === 'batch' ? 'active' : ''}
          onClick={() => setMode('batch')}
          disabled={!isReady}
        >
          Run a file
        </button>
      </div>

      {mode === 'review' && (
        <>
          <UseCaseSteps
            title="Setup — runs once per file"
            hint="The sign-in lives here. It runs once for a whole batch, not once per row."
            steps={useCase.setup_steps}
            onRemove={removeStep('setup_steps')}
          />
          {useCase.row_reset && (
            <UseCaseSteps
              title="Reset — before every row"
              hint="Puts the browser back to a known state so one row cannot inherit the last one's state."
              steps={[useCase.row_reset]}
            />
          )}
          <UseCaseSteps
            title="Per row — runs once for each input"
            steps={useCase.row_steps}
            onRemove={removeStep('row_steps')}
          />

          <div className="card">
            <h3>Inputs and credentials</h3>
            <p className="hint">
              Inputs change from row to row. Credentials are bound once per file and never stored
              in the recording.
            </p>
            <div className="kv">
              <div>
                <span className="label">Inputs</span>
                <div>
                  {useCase.inputs.length === 0
                    ? '(none)'
                    : useCase.inputs.map((i) => (
                        <code key={i.name} style={{ marginRight: 8 }}>
                          {i.name}
                          {i.required ? '' : '?'}
                        </code>
                      ))}
                </div>
              </div>
              <div>
                <span className="label">Credential slots</span>
                <div>
                  {useCase.secrets.length === 0
                    ? '(none)'
                    : useCase.secrets.map((s) => (
                        <code key={s.name} style={{ marginRight: 8 }}>
                          {s.name}
                        </code>
                      ))}
                </div>
              </div>
              <div>
                <span className="label">Allowed domains</span>
                <div>{useCase.allowed_domains.join(', ') || '(none)'}</div>
              </div>
              <div>
                <span className="label">Session check</span>
                <div>
                  {useCase.session_check
                    ? `${useCase.session_check.negate ? 'NOT ' : ''}${useCase.session_check.kind} ${useCase.session_check.value ?? ''}`
                    : '(none — a dropped session will not be noticed)'}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {mode !== 'review' && (
        <>
          <div className="card">
            <h3>Sign in as</h3>
            <select value={credentialId} onChange={(e) => setCredentialId(e.target.value)}>
              <option value="">(no credential)</option>
              {credentials.map((credential) => (
                <option key={credential.id} value={credential.id}>
                  {credential.name} — {credential.slots.join(', ')}
                </option>
              ))}
            </select>
            {missingSlots.length > 0 && (
              <p className="hint" style={{ color: 'var(--danger)' }}>
                This use case still needs: {missingSlots.join(', ')}. Add them below.
              </p>
            )}
          </div>

          {(missingSlots.length > 0 || credentials.length === 0 || !vaultAvailable) && (
            <CredentialsPanel
              requiredSlots={useCase.secrets.map((s) => s.name)}
              onChange={load}
            />
          )}
        </>
      )}

      {mode === 'single' && (
        <div className="card">
          <h3>Run one row</h3>
          <p className="hint">No LLM call is made. This is the same code path a batch uses.</p>
          {useCase.inputs.map((spec) => (
            <label key={spec.name}>
              <span>
                {spec.name}
                {spec.required ? ' *' : ''}
              </span>
              <input
                type="text"
                value={inputs[spec.name] ?? ''}
                placeholder={spec.example || spec.description || spec.type}
                onChange={(e) => setInputs({ ...inputs, [spec.name]: e.target.value })}
              />
            </label>
          ))}
          <button
            type="button"
            className="primary"
            onClick={runOnce}
            disabled={busy || missingSlots.length > 0}
          >
            {busy ? 'Running...' : 'Run'}
          </button>
        </div>
      )}

      {mode === 'batch' && (
        <div className="card">
          <h3>Run a file</h3>
          <p className="hint">
            One header row naming the inputs ({useCase.inputs.map((i) => i.name).join(', ') || 'none'}
            ), then one line per record. Every row is checked before the browser opens. Rows run in
            sequence on one shared session, signing in once.
          </p>
          <textarea
            rows={8}
            value={csv}
            spellCheck={false}
            placeholder={`${useCase.inputs.map((i) => i.name).join(',')}\n...`}
            onChange={(e) => setCsv(e.target.value)}
          />
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (file) setCsv(await file.text());
            }}
          />
          <button
            type="button"
            className="primary"
            onClick={runBatch}
            disabled={busy || !csv.trim() || missingSlots.length > 0}
          >
            {busy ? 'Starting...' : 'Start batch'}
          </button>

          {batch && <BatchProgressPanel batch={batch} onOpenRun={onOpenRun} onResume={resume} />}
        </div>
      )}
    </div>
  );
}

function BatchProgressPanel({
  batch,
  onOpenRun,
  onResume,
}: {
  batch: BatchDetail;
  onOpenRun: (runId: string) => void;
  onResume: () => void;
}) {
  const { batch: summary, executions, pending } = batch;
  const done = summary.succeeded + summary.failed;
  const percent = summary.total ? Math.round((done / summary.total) * 100) : 0;

  return (
    <div className="panel" style={{ marginTop: 16 }}>
      <header>
        <span>Batch</span>
        <span style={{ marginLeft: 'auto', fontFamily: 'var(--mono)' }}>
          {summary.succeeded} ok · {summary.failed} failed · {pending} not attempted
        </span>
      </header>
      <div className="body">
        <div className="progress">
          <div className="bar" style={{ width: `${percent}%` }} />
        </div>

        {summary.error && (
          <div className="banner warn" style={{ marginTop: 12 }}>
            {summary.error}
            {pending > 0 && (
              <p style={{ margin: '6px 0 0' }}>
                <button type="button" onClick={onResume}>
                  Resume — run the {pending} row(s) that did not succeed
                </button>
              </p>
            )}
          </div>
        )}

        <table className="runs" style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>Row</th>
              <th>Status</th>
              <th>Outputs</th>
              <th>Failed at</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {executions.map((execution) => (
              <tr key={execution.id}>
                <td style={{ fontFamily: 'var(--mono)' }}>{execution.row_index}</td>
                <td>
                  <span
                    className={`badge ${
                      execution.status === 'succeeded'
                        ? 'succeeded'
                        : execution.status === 'failed'
                          ? 'failed'
                          : 'pending'
                    }`}
                  >
                    {execution.status}
                  </span>
                </td>
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                  {execution.outputs ? JSON.stringify(execution.outputs) : ''}
                </td>
                <td style={{ fontSize: 12, color: 'var(--danger)' }}>
                  {execution.failed_step_id && (
                    <button type="button" className="link" onClick={() => execution.run_id && onOpenRun(execution.run_id)}>
                      {execution.failed_step_id}
                    </button>
                  )}
                  {execution.error && <div>{execution.error}</div>}
                </td>
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                  {formatDuration(execution.duration_ms)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {summary.finished_at && (
          <p style={{ marginTop: 12 }}>
            <a href={batchResultsUrl(summary.id)} download>
              Download results CSV
            </a>
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * Everything standing between this use case and a batch, in one place.
 *
 * A draft that cannot be published, or a published one that cannot be run,
 * is otherwise a dead end with no explanation: the buttons are simply
 * disabled and nothing says why.
 */
function Blockers({
  useCase,
  blockedScripts,
  credentialCount,
  vaultAvailable,
}: {
  useCase: UseCase;
  blockedScripts: number;
  credentialCount: number;
  vaultAvailable: boolean;
}) {
  const items: { text: string; fix: string }[] = [];

  if (blockedScripts > 0) {
    items.push({
      text: `${blockedScripts} step(s) run raw JavaScript and are not enabled`,
      fix: 'Read the code under "Steps", then choose "I have read the code — enable scripts".',
    });
  }
  if (useCase.status === 'draft') {
    items.push({
      text: 'It is still a draft',
      fix: 'Review the steps, then press Publish. Nothing runs a draft.',
    });
  }
  if (useCase.secrets.length > 0 && !vaultAvailable) {
    items.push({
      text: 'It signs in, but credential storage is switched off in the backend',
      fix: 'Set CREDENTIALS_KEY in .env and restart the backend. See "Credentials" below.',
    });
  } else if (useCase.secrets.length > 0 && credentialCount === 0) {
    items.push({
      text: `It needs credentials (${useCase.secrets.map((s) => s.name).join(', ')}) and none are stored`,
      fix: 'Add one under "Run one" or "Run a file".',
    });
  }
  if (useCase.setup_steps.concat(useCase.row_steps).every((s) => s.action !== 'assert')) {
    items.push({
      text: 'Nothing verifies that a row worked',
      fix: 'Without an assertion a batch reports success even when a row silently did nothing.',
    });
  }

  if (items.length === 0) return null;

  return (
    <div className="banner warn">
      <strong>Before this can run</strong>
      <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
        {items.map((item, index) => (
          <li key={index} style={{ marginBottom: 4 }}>
            {item.text}. <span style={{ opacity: 0.85 }}>{item.fix}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
