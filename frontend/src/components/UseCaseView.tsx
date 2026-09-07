import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, batchResultsUrl } from '../lib/api';
import type {
  BatchDetail,
  BatchSummary,
  CredentialSummary,
  DatasetSummary,
  BatchEstimate,
  Target,
  UseCase,
  UseCaseMode,
  UseCaseStep,
} from '../lib/events';
import { DatasetMapper } from './DatasetMapper';
import {
  DiscoveryResult,
  Downloads,
  downloadsIn,
  summariseOutputs,
} from './DiscoveryResult';
import { formatDuration } from '../lib/format';
import { ActivityLog } from './ActivityLog';
import { CredentialsPanel } from './CredentialsPanel';
import { session } from '../lib/session';
import { UseCaseSteps } from './UseCaseSteps';

interface Props {
  usecaseId: string;
  onBack: () => void;
  onOpenRun: (runId: string) => void;
}

type Mode = 'review' | 'single' | 'batch' | 'activity';

/**
 * Review a recorded use case, publish it, and run it.
 *
 * Review is not optional: distillation is a best guess over a noisy recording,
 * so a use case is created as a draft and only a person moves it to ready.
 */
/**
 * The three things a person can say, in the order of how much they permit.
 *
 * "Follow the deployment" is offered rather than hidden because it is what
 * every use case recorded before this existed is doing, and a screen that
 * showed one of the other two would be claiming a decision nobody made.
 */
const MODE_CHOICES: {
  value: UseCaseMode | null;
  label: string;
  description: string;
  price: string;
}[] = [
  {
    value: 'strict',
    label: 'Strict',
    description:
      'Follows the recorded steps. No model can run — the replay engine cannot reach one, so this is a property of the code rather than a setting.',
    price: 'free',
  },
  {
    value: 'guided',
    label: 'Guided',
    description:
      'The same, until a step stops matching. Then one budgeted call re-finds the control from a list of what is actually on the page, and the run carries on.',
    price: 'free on a good row',
  },
  {
    value: 'explore',
    label: 'Explore',
    description:
      'No plan at all: it works each row out from the page and the task text. For work that genuinely cannot be recorded — a page that differs per record, a next step that depends on what the last one said.',
    price: 'costs on every row',
  },
  {
    value: null,
    label: 'Follow the deployment',
    description:
      'Whatever this installation allows. What every use case did before this choice existed.',
    price: 'depends',
  },
];


export function UseCaseView({ usecaseId, onBack, onOpenRun }: Props) {
  const [useCase, setUseCase] = useState<UseCase | null>(null);
  const [credentials, setCredentials] = useState<CredentialSummary[]>([]);
  const [vaultAvailable, setVaultAvailable] = useState(true);
  // Resource-level permission, distinct from `allow_scripts` in the definition.
  // Both must be true before a script step runs; only an admin can set this one.
  const [scriptsEnabled, setScriptsEnabled] = useState(false);
  const [mode, setMode] = useState<Mode>('review');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  //: Non-null while the title is being edited in place.
  const [draftName, setDraftName] = useState<string | null>(null);

  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [credentialId, setCredentialId] = useState<string>('');
  // Watching the browser work is the fastest way to understand why a step
  // fails, so this is offered on both run paths rather than buried in config.
  const [watch, setWatch] = useState(false);
  const [batch, setBatch] = useState<BatchDetail | null>(null);
  // Batches this use case has already run. `batch` above is only ever set by
  // starting or resuming one, so without this a batch became unreachable the
  // moment the tab closed -- no progress, no resume, no results, on a job that
  // can run for hours.
  const [pastBatches, setPastBatches] = useState<BatchSummary[]>([]);
  // A run that found rows stays on screen, because the useful thing to do next
  // -- turn them into a dataset -- is here rather than in the run view.
  const [lastDiscovery, setLastDiscovery] = useState<{
    execution_id: string;
    outputs: Record<string, unknown>;
  } | null>(null);
  const [rowDelay, setRowDelay] = useState('');
  // Whether this deployment permits healing at all. The card below has to be
  // able to say what "following the deployment" actually resolves to, and it
  // has to grey out Guided where the installation has turned it off -- a
  // choice the screen offers but the run would not honour is a lie.
  const [healingAllowed, setHealingAllowed] = useState<boolean | null>(null);
  // What a batch of the size on screen would cost. Fetched when a file is
  // mapped rather than up front, because the row count is most of the answer.
  const [estimate, setEstimate] = useState<BatchEstimate | null>(null);
  // Which sites this deployment knows about, so the target can be chosen from
  // a list rather than typed from memory.
  const [targets, setTargets] = useState<Target[]>([]);
  const [lastFailure, setLastFailure] = useState<{ execution_id: string; error: string } | null>(
    null,
  );

  const load = useCallback(async () => {
    try {
      const [detail, creds] = await Promise.all([
        api.getUseCase(usecaseId),
        api.listCredentials(),
      ]);
      setUseCase(detail.definition);
      // Best effort: a use case that has never run has none, and failing to
      // list them must not stop the screen loading.
      api
        .listTargets()
        .then((body) => setTargets(body.targets))
        .catch(() => setTargets([]));
      api
        .getConfig()
        .then((body) => setHealingAllowed(Boolean(body.defaults?.healing)))
        .catch(() => setHealingAllowed(null));
      api
        .listBatches(usecaseId)
        .then((body) => setPastBatches(body.batches))
        .catch(() => setPastBatches([]));
      setRowDelay(
        detail.definition.row_delay_seconds === null ||
          detail.definition.row_delay_seconds === undefined
          ? ''
          : String(detail.definition.row_delay_seconds),
      );
      setScriptsEnabled(Boolean(detail.meta?.scripts_enabled));
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

  /**
   * Does this use case sign in at all?
   *
   * The distillation records a `secrets` slot for every value the recording
   * typed into a password field. No slots means no sign-in, and every piece of
   * credential UI below is then not merely unnecessary but misleading -- it
   * reads as something the run is waiting for.
   */
  const needsCredentials = (useCase?.secrets.length ?? 0) > 0;

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

  /** The author half: this definition is allowed to contain scripts. */
  const allowScripts = () =>
    act(async () => {
      if (!useCase) return;
      await api.updateUseCase(usecaseId, { ...useCase, allow_scripts: true });
      setNotice('Raw-JavaScript steps marked as reviewed in this definition.');
      await load();
    });

  /** The administrator half: this use case may actually execute them. */
  const enableScripts = (enabled: boolean) =>
    act(async () => {
      await api.setScriptsEnabled(usecaseId, enabled, 'Reviewed in the use case editor.');
      setNotice(
        enabled
          ? 'Script execution enabled. This use case can now run its JavaScript steps.'
          : 'Script execution disabled for this use case.',
      );
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

  // A recorded URL or typed value is sometimes just wrong -- the browser was
  // on the wrong tab, an env-specific address got baked in -- and until now
  // the only fix was re-recording the whole step. Same shape as every other
  // edit here: patch the definition, save as a new version, reload.
  const editStep =
    (phase: 'setup_steps' | 'row_steps' | 'row_reset') =>
    (stepId: string, field: 'url' | 'value', value: string) =>
      act(async () => {
        if (!useCase) return;
        const patch = (step: UseCaseStep) =>
          step.id === stepId ? { ...step, [field]: value } : step;
        const next =
          phase === 'row_reset'
            ? { ...useCase, row_reset: useCase.row_reset ? patch(useCase.row_reset) : null }
            : { ...useCase, [phase]: useCase[phase].map(patch) };
        await api.updateUseCase(usecaseId, next as UseCase);
        setNotice(`Updated ${stepId}. Saved as a new version.`);
        await load();
      });

  // Saved on blur rather than behind a button: it is one number, and a
  // "Save" next to a single field is ceremony. A new version is written, as
  // for any other edit, so the change is versioned and auditable.
  const saveRowDelay = async () => {
    if (!useCase) return;
    const trimmed = rowDelay.trim();
    const next = trimmed === '' ? null : Number(trimmed);
    if (next !== null && (Number.isNaN(next) || next < 0)) {
      setError('Seconds between rows must be a number, or empty to use the default.');
      return;
    }
    if (next === (useCase.row_delay_seconds ?? null)) return;
    await act(async () => {
      await api.updateUseCase(usecaseId, { ...useCase, row_delay_seconds: next } as UseCase);
      setNotice(
        next === null
          ? 'Pace cleared; this use case uses the deployment default.'
          : `Pace set to ${next}s between rows. Saved as a new version.`,
      );
      await load();
    });
  };

  // Like the target and the pace, the mode is part of the definition: it
  // changes what executes, so it is versioned and audited as any other edit
  // is. Null clears the choice and hands the decision back to the deployment.
  const saveMode = (next: UseCaseMode | null) =>
    act(async () => {
      if (!useCase) return;
      if (next === (useCase.mode ?? null)) return;
      await api.updateUseCase(usecaseId, { ...useCase, mode: next } as UseCase);
      setNotice(
        next === null
          ? 'Cleared; this use case follows the deployment. Saved as a new version.'
          : next === 'strict'
            ? 'Set to Strict. No model can run on this use case. Saved as a new version.'
            : 'Set to Guided. A repair may run when a step stops matching. Saved as a new version.',
      );
      await load();
    });

  // The target lives in the definition, because promotion carries the document
  // and each deployment answers the name for itself. Changing it is therefore
  // an ordinary edit: a new version, versioned and audited like any other.
  const saveTarget = (next: string) =>
    act(async () => {
      if (!useCase) return;
      await api.updateUseCase(usecaseId, { ...useCase, target: next } as UseCase);
      setNotice(
        next
          ? `This use case now runs against ${next}. Saved as a new version.`
          : 'Cleared. It runs against the address it was recorded on.',
      );
      await load();
    });

  const runOnce = () =>
    act(async () => {
      const result = await api.executeUseCase(usecaseId, {
        inputs,
        credential_id: credentialId || null,
        headless: !watch,
      });
      if (result.status === 'succeeded') {
        setLastFailure(null);
        const outputs = (result.outputs ?? {}) as Record<string, unknown>;
        // Stay here for anything worth acting on: rows to turn into a
        // dataset, or files to open. Otherwise go to the run view as before.
        const foundRows =
          Object.values(outputs).some((v) => Array.isArray(v) && v.length > 0) ||
          downloadsIn(outputs).length > 0;
        setNotice(
          `Succeeded using ${result.llm_tokens} LLM tokens. ` +
            `Outputs: ${summariseOutputs(outputs) || '(none)'}`,
        );
        // Rows are the first pass of a two-pass migration and the next step is
        // on this screen, so stay here. Anything else goes to the run view as
        // before.
        if (foundRows) {
          setLastDiscovery({ execution_id: result.execution_id, outputs });
          return;
        }
        setLastDiscovery(null);
        onOpenRun(result.run_id);
        return;
      }
      // Stay on this screen when it fails: the repair button is here.
      setLastFailure({ execution_id: result.execution_id, error: result.error ?? 'it failed' });
      setError(`Failed: ${result.error}`);
    });

  const runBatch = (dataset: DatasetSummary, mapping: Record<string, string>) =>
    act(async () => {
      const started = await api.startBatch(usecaseId, {
        dataset_id: dataset.id,
        mapping,
        credential_id: credentialId || null,
        headless: !watch,
      });
      setBatch(await api.getBatch(started.batch_id));
      setNotice(`Queued ${started.total} rows. They run on one shared browser session.`);
    });

  const resume = () =>
    act(async () => {
      if (!batch) return;
      const resumed = await api.resumeBatch(batch.batch.id, credentialId || null);
      setBatch(await api.getBatch(resumed.batch_id));
      setNotice(`Re-running ${resumed.rows} row(s) that had not succeeded.`);
    });

  const fixIt = (executionId?: string) =>
    act(async () => {
      const result = await api.repairUseCase(usecaseId, { execution_id: executionId });
      if (!result.repaired) {
        setError(
          `${result.diagnosis} ${result.unfixable_reason ?? ''} ` +
            `(${result.llm_tokens} tokens, ${result.confidence} confidence)`,
        );
        return;
      }
      setLastFailure(null);
      setMode('review');
      setNotice(
        [
          result.diagnosis,
          ...(result.applied ?? []).map((line) => `• ${line}`),
          `Saved as v${result.version}, back to draft for you to review. ` +
            `${result.llm_tokens} tokens, ${result.confidence} confidence.`,
        ].join('\n'),
      );
      await load();
    });

  const rename = (name: string) =>
    act(async () => {
      setDraftName(null);
      if (!useCase || name === useCase.name) return;
      // A rename creates no version: the name is a label, not part of what
      // executes, so it does not belong in a history of behaviour.
      await api.renameUseCase(usecaseId, name);
      setNotice(`Renamed to ${name}.`);
      await load();
    });

  const archive = () =>
    act(async () => {
      await api.archiveUseCase(usecaseId);
      onBack();
    });

  const destroy = () =>
    act(async () => {
      if (
        !window.confirm(
          [
            'Delete this use case permanently?',
            '',
            'Its steps, every saved version and its execution records go for good.',
            'The runs and screenshots they produced are kept.',
            '',
            'Archive instead if you only want it out of the way.',
          ].join('\n'),
        )
      ) {
        return;
      }
      await api.deleteUseCase(usecaseId);
      onBack();
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
  const scriptSteps = [...useCase.setup_steps, ...useCase.row_steps].filter(
    (s) => s.action === 'script',
  );
  const blockedScripts = useCase.allow_scripts ? [] : scriptSteps;
  // Two separate permissions, and running needs both. `allow_scripts` lives in
  // the definition an author edits; `scripts_enabled` lives on the resource and
  // only an administrator can set it -- otherwise an author could grant
  // themselves code execution by editing JSON.
  const scriptsNeedAdmin = scriptSteps.length > 0 && !scriptsEnabled;

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
          {draftName === null ? (
            <p>
              {useCase.name}{' '}
              <button
                type="button"
                className="link"
                onClick={() => setDraftName(useCase.name)}
                title="Rename — this does not create a new version"
              >
                rename
              </button>
            </p>
          ) : (
            <p className="row">
              <input
                type="text"
                autoFocus
                value={draftName}
                maxLength={200}
                onChange={(event) => setDraftName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && draftName.trim()) rename(draftName.trim());
                  if (event.key === 'Escape') setDraftName(null);
                }}
              />
              <button
                type="button"
                onClick={() => draftName.trim() && rename(draftName.trim())}
                disabled={busy || !draftName.trim()}
              >
                Save
              </button>
              <button type="button" onClick={() => setDraftName(null)} disabled={busy}>
                Cancel
              </button>
            </p>
          )}
        </div>
        {!isReady && (
          <button type="button" className="primary" onClick={publish} disabled={busy}>
            Publish
          </button>
        )}
        {useCase.status !== 'archived' && (
          <button type="button" onClick={archive} disabled={busy} title="Reversible — hides it from the list">
            Archive
          </button>
        )}
        <button type="button" className="danger" onClick={destroy} disabled={busy}>
          Delete
        </button>
      </div>

      {error && (
        <div className="banner error">
          <div style={{ whiteSpace: 'pre-wrap' }}>{error}</div>
          {lastFailure && (
            <p style={{ margin: '8px 0 0' }}>
              <button type="button" onClick={() => fixIt(lastFailure.execution_id)} disabled={busy}>
                {busy ? 'Looking at it...' : 'Fix it with AI'}
              </button>
              <span className="hint" style={{ display: 'inline', marginLeft: 8 }}>
                Reads the page as it was when it broke and proposes a repair. One LLM call.
              </span>
            </p>
          )}
        </div>
      )}
      {notice && <div className="banner" style={{ whiteSpace: 'pre-wrap' }}>{notice}</div>}

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

      {blockedScripts.length === 0 && scriptsNeedAdmin && (
        <div className="banner error">
          <strong>Script execution is not enabled for this use case</strong>
          <p style={{ margin: '6px 0' }}>
            You have marked the {scriptSteps.length} JavaScript step(s) as reviewed, which
            is the author's half. Executing them is a separate permission held by an
            administrator, so that editing a definition cannot grant code execution to
            whoever edited it.
          </p>
          {session.can('script:enable') ? (
            <button type="button" onClick={() => enableScripts(true)} disabled={busy}>
              I have read the code — allow this use case to run it
            </button>
          ) : (
            <p style={{ margin: 0 }}>
              Ask an administrator to enable it, or remove the script steps.
            </p>
          )}
        </div>
      )}

      {scriptSteps.length > 0 && scriptsEnabled && session.can('script:enable') && (
        <div className="banner">
          Script execution is enabled for this use case.{' '}
          <button
            type="button"
            className="linkish"
            onClick={() => enableScripts(false)}
            disabled={busy}
          >
            Withdraw it
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
        <button
          type="button"
          className={mode === 'activity' ? 'active' : ''}
          onClick={() => setMode('activity')}
          title="Who published, ran, repaired or changed this use case"
        >
          Activity
        </button>
      </div>

      {mode === 'activity' && (
        <div className="card">
          <h3>Activity</h3>
          <p className="hint">
            Every publish, run, repair and permission change on this use case, with who
            did it. Entries are kept after an account is deleted — the person's address
            is stored on the entry rather than looked up.
          </p>
          <ActivityLog usecaseId={usecaseId} />
        </div>
      )}

      {mode === 'review' && (
        <>
          <UseCaseSteps
            title="Setup — runs once per file"
            hint="The sign-in lives here. It runs once for a whole batch, not once per row."
            steps={useCase.setup_steps}
            onRemove={removeStep('setup_steps')}
            onEditField={editStep('setup_steps')}
          />
          {useCase.row_reset && (
            <UseCaseSteps
              title="Reset — before every row"
              hint="Puts the browser back to a known state so one row cannot inherit the last one's state."
              steps={[useCase.row_reset]}
              onEditField={editStep('row_reset')}
            />
          )}
          <UseCaseSteps
            title="Per row — runs once for each input"
            steps={useCase.row_steps}
            onRemove={removeStep('row_steps')}
            onEditField={editStep('row_steps')}
          />

          {(useCase.dropped ?? []).length > 0 && (
            <details className="card dropped">
              <summary>
                {useCase.dropped.length} recorded call
                {useCase.dropped.length === 1 ? '' : 's'} did not become a step
              </summary>
              <p className="hint">
                A recording keeps only what actually worked. Everything the agent tried and
                failed, and everything that only looked at the page, is left out — replaying a
                failed action wastes time and can leave the page in a state the next step does
                not expect. Check here if a step you expected is missing.
              </p>
              <ul>
                {useCase.dropped.map((line, index) => (
                  <li key={index}>{line}</li>
                ))}
              </ul>
            </details>
          )}

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
          {/* Only for a use case that actually signs in. A recording with no
              secrets has nothing to bind a credential to, and offering the
              choice reads as a requirement -- which is how a use case that
              needed no credentials ended up looking like it was asking for
              one. */}
          {needsCredentials && (
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
          )}

          <div className="card">
            <h3>Where it runs</h3>
            <p className="hint">
              A use case names a target; this deployment says what address that target has.
              That is what lets the same use case run in dev, UAT and production without the
              definition changing.
            </p>
            <label className="field" style={{ maxWidth: 380 }}>
              <span>Target</span>
              <select
                value={useCase.target ?? ''}
                disabled={!session.can('usecase:create')}
                onChange={(e) => void saveTarget(e.target.value)}
              >
                <option value="">
                  (none — run against the address it was recorded on)
                </option>
                {targets.map((target) => (
                  <option key={target.name} value={target.name}>
                    {target.name} — {target.base_url}
                  </option>
                ))}
                {/* A target the definition names but this deployment has no
                    address for. Showing it is what makes the run-time refusal
                    legible: you can see what it is asking for, and change it. */}
                {useCase.target && !targets.some((t) => t.name === useCase.target) && (
                  <option value={useCase.target}>
                    {useCase.target} — not defined in this deployment
                  </option>
                )}
              </select>
            </label>
            {useCase.target && !targets.some((t) => t.name === useCase.target) ? (
              <p className="hint" style={{ color: 'var(--danger)' }}>
                This deployment has no address for <code>{useCase.target}</code>, so runs will
                refuse. Add it under Targets, or pick one above.
              </p>
            ) : (
              <p className="hint">
                {useCase.target
                  ? `Runs against whatever ${useCase.target} points at here.`
                  : `Runs against ${useCase.base_url || 'the recorded address'}. That is right for a single environment, and what you change before promoting.`}
              </p>
            )}
          </div>

          <div className="card">
            <h3>How it runs</h3>
            <p className="hint">
              How much a model is allowed to do. This belongs to the workflow rather than
              to the installation: one site is rebuilt every sprint and another has not
              changed in four years, and one setting cannot be right for both.
            </p>

            <div className="modes">
              {MODE_CHOICES.map((choice) => {
                const chosen = (useCase.mode ?? null) === choice.value;
                // Both modes that can reach a model are unavailable where the
                // deployment has switched healing off: the ceiling only ever
                // restricts, and offering a choice a run would not honour is
                // worse than not offering it.
                const unavailable =
                  (choice.value === 'guided' || choice.value === 'explore') &&
                  healingAllowed === false;
                return (
                  <label
                    key={String(choice.value)}
                    className={`mode-choice${chosen ? ' chosen' : ''}${unavailable ? ' unavailable' : ''}`}
                  >
                    <input
                      type="radio"
                      name="usecase-mode"
                      checked={chosen}
                      disabled={!session.can('usecase:create') || unavailable}
                      onChange={() => void saveMode(choice.value)}
                    />
                    <span>
                      <strong>{choice.label}</strong>
                      <span className="hint" style={{ margin: '2px 0 0' }}>
                        {choice.description}
                      </span>
                      {unavailable && (
                        <span className="hint" style={{ margin: '2px 0 0' }}>
                          This deployment has healing switched off, so a repair would not
                          run even if it were chosen here.
                        </span>
                      )}
                    </span>
                    <span
                      className={`mode-price${choice.value === 'explore' ? ' paid' : ''}`}
                    >
                      {choice.price}
                    </span>
                  </label>
                );
              })}
            </div>

            {useCase.mode === null || useCase.mode === undefined ? (
              <p className="hint">
                Nothing chosen, so this follows the deployment
                {healingAllowed === null
                  ? '.'
                  : healingAllowed
                    ? ' — which currently allows a repair (Guided).'
                    : ' — which currently allows no model at all (Strict).'}{' '}
                Pick one above to decide it here instead.
              </p>
            ) : (
              <p className="hint">
                Chosen on this use case, so it stays{' '}
                {useCase.mode === 'strict'
                  ? 'Strict'
                  : useCase.mode === 'guided'
                    ? 'Guided'
                    : 'Explore'}{' '}
                wherever it is promoted.
              </p>
            )}
          </div>

          <div className="card">
            <h3>Pace</h3>
            <p className="hint">
              How long to wait between rows. Politeness is a property of the site rather
              than of this installation: one vendor tolerates a request a second, another
              starts refusing after three, and a long extraction that reads as an attack
              gets the account blocked. The person who recorded this knows which site it is.
            </p>
            <label className="field" style={{ maxWidth: 260 }}>
              <span>Seconds between rows</span>
              <input
                type="number"
                min={0}
                max={600}
                step={0.1}
                value={rowDelay}
                placeholder="server default"
                disabled={!session.can('usecase:create')}
                onChange={(e) => setRowDelay(e.target.value)}
                onBlur={() => void saveRowDelay()}
              />
            </label>
            <p className="hint">
              {useCase.row_delay_seconds === null || useCase.row_delay_seconds === undefined
                ? 'Empty uses the deployment default (REPLAY_ROW_DELAY_SECONDS).'
                : `This use case waits ${useCase.row_delay_seconds}s between rows.`}
            </p>
          </div>

          <div className="card">
            <h3>Browser</h3>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={watch}
                onChange={(e) => setWatch(e.target.checked)}
              />
              <span>
                Show the browser while it runs
                <span className="hint" style={{ margin: '2px 0 0' }}>
                  Opens a visible window instead of running headless. The quickest way to see
                  why a step fails — though a long batch will hold a window open throughout.
                </span>
              </span>
            </label>
          </div>

          {/* `credentials.length === 0` and `!vaultAvailable` say something
              about the *installation*, not about this use case -- so they only
              matter once this use case needs a credential at all. Without the
              first clause, a fresh install showed the add-a-credential form on
              every use case, including ones that never sign in. */}
          {needsCredentials &&
            (missingSlots.length > 0 || credentials.length === 0 || !vaultAvailable) && (
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
          {lastDiscovery && <Downloads outputs={lastDiscovery.outputs} />}

          {lastDiscovery && (
            <DiscoveryResult
              executionId={lastDiscovery.execution_id}
              outputs={lastDiscovery.outputs}
              onSaved={() => setNotice('Saved. It is now under Run a file, below.')}
            />
          )}

          {pastBatches.length > 0 && (
            <div className="card">
              <h3>Earlier runs</h3>
              <p className="hint">
                A batch keeps running whether or not this page is open. Open one to watch it,
                resume what a stopped run never attempted, or take the results.
              </p>
              <table className="mapping">
                <thead>
                  <tr>
                    <th>Started</th>
                    <th>Status</th>
                    <th>Rows</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {pastBatches.map((past) => (
                    <tr key={past.id}>
                      <td style={{ fontSize: 13 }}>
                        {new Date(past.created_at).toLocaleString()}
                      </td>
                      <td>
                        <span className={`badge ${past.status}`}>{past.status}</span>
                      </td>
                      <td style={{ fontSize: 13 }}>
                        {past.succeeded}/{past.total} done
                        {past.failed > 0 && (
                          <span style={{ color: 'var(--danger)' }}> · {past.failed} failed</span>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 6 }}>
                          <button
                            type="button"
                            onClick={() =>
                              act(async () => setBatch(await api.getBatch(past.id)))
                            }
                          >
                            Open
                          </button>
                          <a className="linkish" href={batchResultsUrl(past.id)}>
                            Results
                          </a>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <h3>Run a file</h3>
          <p className="hint">
            Upload your records, check that each field is reading the right column, then start.
            Every row is validated before the browser opens, and rows run in sequence on one
            shared session so the workflow signs in once.
          </p>

          {estimate && <CostBeforeCommitting estimate={estimate} />}

          <DatasetMapper
            useCase={useCase}
            usecaseId={usecaseId}
            onRowCount={(rows) => {
              // A limit is what stops a mistake; this is what prevents one.
              void api
                .estimateBatch(usecaseId, Math.max(1, rows))
                .then(setEstimate)
                .catch(() => setEstimate(null));
            }}
            busy={busy}
            disabled={missingSlots.length > 0}
            disabledReason={
              missingSlots.length > 0
                ? `This workflow signs in. Choose a credential providing: ${missingSlots.join(', ')}.`
                : undefined
            }
            onReady={runBatch}
          />

          {batch && <BatchProgressPanel batch={batch} onOpenRun={onOpenRun} onResume={resume} />}
        </div>
      )}
    </div>
  );
}

/** What this batch is expected to cost, said before the button rather than
 *  after the bill. A range, because how many turns a row takes depends on the
 *  site and a single figure would imply an accuracy this cannot have. */
function CostBeforeCommitting({ estimate }: { estimate: BatchEstimate }) {
  const free = estimate.high_usd === 0;
  return (
    <div className={estimate.over_budget ? 'banner error' : free ? 'verified' : 'banner'}>
      <strong>
        {free
          ? 'This batch costs nothing.'
          : estimate.low_usd === estimate.high_usd
            ? `About $${estimate.high_usd.toFixed(2)}.`
            : `Between $${estimate.low_usd.toFixed(2)} and $${estimate.high_usd.toFixed(2)}.`}
      </strong>{' '}
      {estimate.note}
      {estimate.over_budget && (
        <>
          {' '}
          <strong>
            That is more than this workspace has left this month, so it would stop part
            way through.
          </strong>
        </>
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
              <th>Inputs</th>
              <th>Outputs</th>
              <th>Run by</th>
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
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12, maxWidth: 260 }}>
                  {/* What this row was actually given. Half of "what happened
                      to record 700" is what it was run with. */}
                  {Object.entries(execution.inputs ?? {})
                    .map(([k, v]) => `${k}=${String(v)}`)
                    .join(', ')}
                </td>
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                  {/* A discovery run's output is hundreds of rows; the count
                      is the readable thing in a cell this size. */}
                  {summariseOutputs(execution.outputs as Record<string, unknown>)}
                </td>
                <td style={{ fontSize: 12 }}>
                  {execution.owner_email || <span className="hint">—</span>}
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
