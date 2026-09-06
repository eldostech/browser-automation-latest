/**
 * Describing a task, and watching an agent work it out.
 *
 * Two panes, because watching is how trust gets built. The transcript is the
 * existing run event stream -- an agent session writes the same events a
 * replay does, so this needed no new plumbing and a session can be reopened
 * later from the run view like anything else.
 *
 * The screen is honest about three things a person needs and would otherwise
 * have to guess at: what it has spent, when it is waiting for them, and
 * whether what it produced actually replays.
 */

import { useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type {
  AgentEvent,
  AgentSessionDetail,
  CredentialSummary,
  Target,
  WorkspaceSpend,
} from '../lib/events';
import { useRunStream } from '../lib/useRunStream';

type Props = {
  onSaved: (usecaseId: string) => void;
  onCancel: () => void;
};

export function AgentSessionView({ onSaved, onCancel }: Props) {
  const [session, setSession] = useState<AgentSessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // The form
  const [task, setTask] = useState('');
  const [target, setTarget] = useState('');
  const [startUrl, setStartUrl] = useState('');
  const [name, setName] = useState('');
  const [credentialId, setCredentialId] = useState('');
  const [mayWrite, setMayWrite] = useState(false);
  const [steps, setSteps] = useState('40');
  const [spendCap, setSpendCap] = useState('1.00');
  const [targets, setTargets] = useState<Target[]>([]);
  const [credentials, setCredentials] = useState<CredentialSummary[]>([]);
  // What the workspace has left this month. Shown before the button rather
  // than discovered by being refused, which is a worse way to learn it.
  const [spend, setSpend] = useState<WorkspaceSpend | null>(null);

  useEffect(() => {
    api.listTargets().then((b) => setTargets(b.targets)).catch(() => setTargets([]));
    api.listCredentials().then((b) => setCredentials(b.credentials)).catch(() => undefined);
    api.getSpend().then(setSpend).catch(() => setSpend(null));
  }, []);

  // Poll while it is working. The transcript is live over the websocket; this
  // is for the parts that are not events -- the spend, the draft, the verdict.
  useEffect(() => {
    if (!session || !isLive(session.status)) return;
    const timer = window.setInterval(async () => {
      try {
        setSession(await api.getAgentSession(session.id));
      } catch {
        /* a poll that fails is not worth interrupting the session for */
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [session?.id, session?.status]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setSession(
        await api.startAgentSession({
          task: task.trim(),
          target: target || undefined,
          start_url: startUrl.trim() || undefined,
          name: name.trim() || undefined,
          credential_id: credentialId || undefined,
          may_write: mayWrite,
          budget_steps: Number(steps) || 40,
          budget_usd: Number(spendCap) || 1,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const decide = async (decision: 'approved' | 'rejected') => {
    if (!session) return;
    await api.decideAgentSession(session.id, decision);
    setSession(await api.getAgentSession(session.id));
  };

  const save = async () => {
    if (!session) return;
    setBusy(true);
    try {
      const saved = await api.saveAgentSession(session.id, name.trim());
      onSaved(saved.usecase_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!session) {
    return (
      <div className="card">
        <h2>Describe it</h2>
        <p className="hint">
          Say what to do. An agent works it out in a browser you can watch, marking what
          varies per row and what to read out. It costs tokens once; what it produces
          replays for nothing, as many times as you like.
        </p>

        {error && <div className="banner error">{error}</div>}

        <label className="field">
          <span>What should it do?</span>
          <textarea
            rows={3}
            value={task}
            placeholder="Open the accounts list, find account A-1001, and read its balance."
            onChange={(e) => setTask(e.target.value)}
          />
        </label>

        <div className="row" style={{ alignItems: 'flex-start', gap: 16 }}>
          <label className="field" style={{ flex: 1 }}>
            <span>Where</span>
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">An address, below</option>
              {targets.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
          {!target && (
            <label className="field" style={{ flex: 2 }}>
              <span>Starting address</span>
              <input
                value={startUrl}
                placeholder="https://vendor.example.com/accounts"
                onChange={(e) => setStartUrl(e.target.value)}
              />
            </label>
          )}
        </div>

        <div className="row" style={{ alignItems: 'flex-start', gap: 16 }}>
          <label className="field" style={{ flex: 1 }}>
            <span>Sign in as</span>
            <select value={credentialId} onChange={(e) => setCredentialId(e.target.value)}>
              <option value="">No sign-in needed</option>
              {credentials.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ flex: 1 }}>
            <span>Call it</span>
            <input
              value={name}
              placeholder="taken from the task"
              onChange={(e) => setName(e.target.value)}
            />
          </label>
        </div>

        <h4>Stop it before it costs too much</h4>
        <p className="hint">
          Checked before every call, not reported after. Reaching a limit ends the session
          and keeps everything it did &mdash; often that is a complete recording.
        </p>
        {spend && spend.limit_usd !== null && (
          <p className={spend.remaining_usd === 0 ? 'banner error' : 'hint'}>
            This workspace has spent ${spend.usd.toFixed(2)} of its $
            {spend.limit_usd.toFixed(2)} limit this month.{' '}
            {spend.remaining_usd === 0
              ? 'There is nothing left, so a session cannot start until an administrator raises it.'
              : `A session here can spend at most $${(spend.remaining_usd ?? 0).toFixed(2)}, whatever you set below.`}
          </p>
        )}

        <div className="row" style={{ gap: 16 }}>
          <label className="field" style={{ maxWidth: 140 }}>
            <span>Steps</span>
            <input type="number" min={1} max={500} value={steps} onChange={(e) => setSteps(e.target.value)} />
          </label>
          <label className="field" style={{ maxWidth: 140 }}>
            <span>Spend (USD)</span>
            <input
              type="number"
              min={0.01}
              step={0.25}
              value={spendCap}
              onChange={(e) => setSpendCap(e.target.value)}
            />
          </label>
        </div>

        <label className="checkbox">
          <input type="checkbox" checked={mayWrite} onChange={(e) => setMayWrite(e.target.checked)} />
          <span>
            Let it change things
            <span className="hint" style={{ margin: '2px 0 0' }}>
              Off means it can look but not click, type or submit. Anything irreversible
              &mdash; submitting, deleting, paying &mdash; stops and asks you either way.
            </span>
          </span>
        </label>

        <div className="row end" style={{ marginTop: 16 }}>
          <button type="button" className="ghost" onClick={onCancel}>
            Back
          </button>
          <button
            type="button"
            className="primary"
            disabled={busy || !task.trim() || (!target && !startUrl.trim())}
            onClick={() => void start()}
          >
            {busy ? 'Starting…' : 'Start'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <Working
      session={session}
      error={error}
      busy={busy}
      onDecide={decide}
      onSave={save}
      onStop={async () => {
        await api.cancelAgentSession(session.id).catch(() => undefined);
        setSession(await api.getAgentSession(session.id));
      }}
    />
  );
}

function isLive(status: string): boolean {
  return status === 'running' || status === 'awaiting_approval';
}

function Working({
  session,
  error,
  busy,
  onDecide,
  onSave,
  onStop,
}: {
  session: AgentSessionDetail;
  error: string | null;
  busy: boolean;
  onDecide: (d: 'approved' | 'rejected') => Promise<void>;
  onSave: () => Promise<void>;
  onStop: () => Promise<void>;
}) {
  const { events } = useRunStream(session.run_id);
  const spend = session.spend ?? {};
  const verification = session.verification ?? {};

  return (
    <>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2 style={{ margin: 0 }}>{titleFor(session.status)}</h2>
          <span className="meter">
            <span>{session.steps} steps</span>
            <span>{(spend.tokens ?? 0).toLocaleString()} tokens</span>
            <span className="cost">${(spend.usd ?? 0).toFixed(2)}</span>
          </span>
        </div>
        <p className="hint">{session.task}</p>

        {error && <div className="banner error">{error}</div>}
        {session.error && <div className="banner error">{session.error}</div>}
        {session.stopped_by && <div className="banner">{session.stopped_by}</div>}

        {session.awaiting && (
          <div className="approval">
            <strong>
              It wants to {readable(session.awaiting.call.name)}
              {session.awaiting.categories ? ` (${session.awaiting.categories})` : ''}.
            </strong>
            <p className="hint">
              This was classified from what the call does, not from the agent&rsquo;s
              opinion of it. Nothing has happened yet, and the browser is holding its
              place while you decide.
            </p>
            <pre className="args">{JSON.stringify(session.awaiting.call.input, null, 2)}</pre>
            <div className="row">
              <button type="button" className="primary" onClick={() => void onDecide('approved')}>
                Allow once
              </button>
              <button type="button" className="ghost" onClick={() => void onDecide('rejected')}>
                Refuse
              </button>
            </div>
          </div>
        )}

        {isLive(session.status) && !session.awaiting && (
          <div className="row end">
            <button type="button" className="ghost" onClick={() => void onStop()}>
              Stop
            </button>
          </div>
        )}
      </div>

      <div className="card">
        <h3>What it did</h3>
        <Transcript events={events} />
      </div>

      {session.use_case && (
        <div className="card">
          <h3>What it recorded</h3>
          {verification.ran ? (
            <div className={verification.ok ? 'verified' : 'banner error'}>
              {verification.ok
                ? `Replayed cleanly by the engine in ${((verification.duration_ms ?? 0) / 1000).toFixed(1)}s, no model involved.`
                : `Did not replay. ${verification.failed_step ? `Step ${verification.failed_step}: ` : ''}${verification.error}`}
            </div>
          ) : (
            <div className="banner">Not verified: {verification.skipped}</div>
          )}

          {session.draft_warnings.length > 0 && (
            <ul className="warnings">
              {session.draft_warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}

          <Steps session={session} />

          <div className="row end" style={{ marginTop: 12 }}>
            <button type="button" className="primary" disabled={busy} onClick={() => void onSave()}>
              {busy ? 'Saving…' : 'Save as a use case'}
            </button>
          </div>
          <p className="hint">
            It saves as a draft, and you review it like any recording. Nothing runs a
            thousand rows until a person publishes it.
          </p>
        </div>
      )}
    </>
  );
}

function Steps({ session }: { session: AgentSessionDetail }) {
  const useCase = session.use_case;
  if (!useCase) return null;
  const rows = [
    ...useCase.setup_steps.map((s) => ['once per batch', s] as const),
    ...useCase.row_steps.map((s) => ['once per row', s] as const),
  ];
  return (
    <table className="mapping">
      <thead>
        <tr>
          <th>When</th>
          <th>Does</th>
          <th>To</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([phase, step]) => (
          <tr key={step.id}>
            <td className="hint" style={{ whiteSpace: 'nowrap' }}>{phase}</td>
            <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
              {step.action}
              {step.value ? ` ${step.value}` : ''}
              {step.output ? ` → ${step.output}` : ''}
            </td>
            <td style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-dim)' }}>
              {step.description || step.url || ''}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** The tool calls, as they happen. Marks are highlighted, because they are the
 *  part that turns doing the task into recording it. */
function Transcript({ events }: { events: AgentEvent[] }) {
  const lines = useMemo(
    () =>
      events.filter(
        (e) => e.type === 'tool_call' || e.type === 'thinking' || e.type === 'approval_required',
      ),
    [events],
  );

  if (lines.length === 0) return <p className="hint">Waiting for it to start…</p>;

  return (
    <ol className="transcript">
      {lines.map((event, index) => {
        if (event.type === 'thinking') {
          return (
            <li key={index} className="thought">
              {event.text}
            </li>
          );
        }
        if (event.type === 'approval_required') {
          return (
            <li key={index} className="asked">
              asked about {readable(event.name)}
            </li>
          );
        }
        const mark = event.name.startsWith('mark_') || event.name.endsWith('_row');
        return (
          <li key={index} className={mark ? 'mark' : undefined}>
            <code>{event.name}</code> <span className="hint">{describeArgs(event.arguments)}</span>
          </li>
        );
      })}
    </ol>
  );
}

function describeArgs(args: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const key of ['url', 'target', 'ref', 'text', 'name', 'column', 'slot', 'key']) {
    const value = args?.[key];
    if (value !== undefined && value !== null && value !== '') {
      parts.push(`${key}=${String(value).slice(0, 60)}`);
    }
  }
  return parts.join(' · ');
}

function readable(tool: string): string {
  return tool.replace(/^browser_/, '').replace(/_/g, ' ');
}

function titleFor(status: string): string {
  if (status === 'running') return 'Working';
  if (status === 'awaiting_approval') return 'Waiting for you';
  if (status === 'succeeded') return 'Finished';
  if (status === 'partial') return 'Stopped early';
  if (status === 'cancelled') return 'Stopped';
  return 'Failed';
}
