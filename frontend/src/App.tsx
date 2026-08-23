import { useCallback, useEffect, useState } from 'react';
import { api } from './lib/api';
import { RunHistory } from './components/RunHistory';
import { RunView } from './components/RunView';
import { TaskComposer } from './components/TaskComposer';
import { UseCaseList } from './components/UseCaseList';
import { UseCaseView } from './components/UseCaseView';

type View =
  | { name: 'compose' }
  | { name: 'history' }
  | { name: 'run'; runId: string }
  | { name: 'usecases' }
  | { name: 'usecase'; usecaseId: string };

/** The run id lives in the URL hash so a live run can be shared or reloaded. */
function viewFromHash(): View {
  const hash = window.location.hash.replace(/^#\/?/, '');
  if (hash.startsWith('runs/')) {
    const runId = hash.slice('runs/'.length);
    if (runId) return { name: 'run', runId };
  }
  if (hash.startsWith('usecases/')) {
    const usecaseId = hash.slice('usecases/'.length);
    if (usecaseId) return { name: 'usecase', usecaseId };
  }
  if (hash === 'usecases') return { name: 'usecases' };
  if (hash === 'history') return { name: 'history' };
  return { name: 'compose' };
}

function hashFor(view: View): string {
  if (view.name === 'run') return `#/runs/${view.runId}`;
  if (view.name === 'usecase') return `#/usecases/${view.usecaseId}`;
  if (view.name === 'usecases') return '#/usecases';
  if (view.name === 'history') return '#/history';
  return '#/';
}

export default function App() {
  const [view, setView] = useState<View>(viewFromHash);
  const [health, setHealth] = useState<{ status?: string; mcp?: { ok?: boolean | null } } | null>(
    null,
  );

  const navigate = useCallback((next: View) => {
    setView(next);
    const hash = hashFor(next);
    if (window.location.hash !== hash) window.location.hash = hash;
  }, []);

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    api
      .health()
      .then((body) => setHealth(body as { status?: string }))
      .catch(() => setHealth({ status: 'unreachable' }));
  }, []);

  const mcpOk = health?.mcp?.ok;

  return (
    <div className="app">
      <header className="topbar">
        <h1>Browser Agent</h1>
        <nav>
          <button
            type="button"
            className={view.name === 'compose' ? 'active' : ''}
            onClick={() => navigate({ name: 'compose' })}
          >
            New run
          </button>
          <button
            type="button"
            className={view.name === 'history' ? 'active' : ''}
            onClick={() => navigate({ name: 'history' })}
          >
            History
          </button>
          <button
            type="button"
            className={view.name === 'usecases' || view.name === 'usecase' ? 'active' : ''}
            onClick={() => navigate({ name: 'usecases' })}
            title="Recorded steps that replay without an LLM"
          >
            Use cases
          </button>
        </nav>
        <span className="spacer" />
        <span className="meta">
          {health === null
            ? 'checking backend...'
            : health.status === 'unreachable'
              ? 'backend unreachable'
              : `backend ${health.status} - mcp ${mcpOk === true ? 'ok' : mcpOk === false ? 'down' : 'unknown'}`}
        </span>
      </header>

      <main className={view.name === 'run' || view.name === 'usecase' ? 'page' : 'page narrow'}>
        {view.name === 'compose' && (
          <>
            {health?.status === 'degraded' && (
              <div className="banner error">
                The backend reports a degraded state. Check <code>/healthz</code> -- usually a
                missing <code>ANTHROPIC_API_KEY</code> or an MCP server that will not start.
              </div>
            )}
            <TaskComposer onStarted={(runId) => navigate({ name: 'run', runId })} />
          </>
        )}

        {view.name === 'history' && (
          <RunHistory onOpen={(runId) => navigate({ name: 'run', runId })} />
        )}

        {view.name === 'run' && (
          <RunView
            runId={view.runId}
            onBack={() => navigate({ name: 'history' })}
            onRecorded={(usecaseId) => navigate({ name: 'usecase', usecaseId })}
          />
        )}

        {view.name === 'usecases' && (
          <UseCaseList onOpen={(usecaseId) => navigate({ name: 'usecase', usecaseId })} />
        )}

        {view.name === 'usecase' && (
          <UseCaseView
            usecaseId={view.usecaseId}
            onBack={() => navigate({ name: 'usecases' })}
            onOpenRun={(runId) => navigate({ name: 'run', runId })}
          />
        )}
      </main>
    </div>
  );
}
