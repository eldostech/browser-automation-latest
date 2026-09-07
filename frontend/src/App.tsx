import { useCallback, useEffect, useState } from 'react';
import { api } from './lib/api';
import { session, type CurrentUser } from './lib/session';
import { HealingMemory } from './components/HealingMemory';
import { Targets } from './components/Targets';
import { Help } from './components/Help';
import { NewUseCase } from './components/NewUseCase';
import { RunHistory } from './components/RunHistory';
import { SignIn } from './components/SignIn';
import { RunView } from './components/RunView';
import { UseCaseList } from './components/UseCaseList';
import { UseCaseView } from './components/UseCaseView';

type View =
  | { name: 'record' }
  | { name: 'history' }
  | { name: 'memory' }
  | { name: 'targets' }
  | { name: 'run'; runId: string }
  | { name: 'usecases' }
  | { name: 'usecase'; usecaseId: string }
  | { name: 'help'; section: 'user' | 'technical' };

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
  if (hash === 'history') return { name: 'history' };
  if (hash === 'memory') return { name: 'memory' };
  if (hash === 'targets') return { name: 'targets' };
  if (hash === 'usecases') return { name: 'usecases' };
  if (hash === 'help/technical') return { name: 'help', section: 'technical' };
  if (hash === 'help' || hash === 'help/user') return { name: 'help', section: 'user' };
  return { name: 'record' };
}

function hashFor(view: View): string {
  if (view.name === 'run') return `#/runs/${view.runId}`;
  if (view.name === 'usecase') return `#/usecases/${view.usecaseId}`;
  if (view.name === 'usecases') return '#/usecases';
  if (view.name === 'history') return '#/history';
  if (view.name === 'memory') return '#/memory';
  if (view.name === 'targets') return '#/targets';
  if (view.name === 'help') return view.section === 'technical' ? '#/help/technical' : '#/help';
  return '#/record';
}

export default function App() {
  const [user, setUser] = useState<CurrentUser | null>(session.user);
  const [view, setView] = useState<View>(viewFromHash);
  const [health, setHealth] = useState<{
    status?: string;
    queue?: { queued?: number; running?: number };
  } | null>(null);
  const [environment, setEnvironment] = useState('');

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

  // One subscription, so an expired token discovered by *any* request drops
  // the whole app back to the sign-in screen rather than leaving one panel
  // showing an error while the rest keeps retrying.
  useEffect(() => session.subscribe(setUser), []);

  // A stored token may have expired while the tab was closed. Asking the
  // server who we are is the only way to find out, and a 401 clears it.
  useEffect(() => {
    if (session.isSignedIn) api.me().then(setUser).catch(() => undefined);
  }, []);

  // Polled rather than fetched once. A single attempt at mount meant a backend
  // that was still starting left the header saying "unreachable" for the whole
  // session -- tolerable for a status line, not for the environment badge,
  // which is there so nobody starts a batch against the wrong deployment.
  useEffect(() => {
    if (!user) return;
    let live = true;
    const poll = () => {
      api
        .health()
        .then((body) => live && setHealth(body as { status?: string }))
        .catch(() => live && setHealth({ status: 'unreachable' }));
      api
        .getConfig()
        .then((body) => live && setEnvironment(body.environment ?? ''))
        .catch(() => undefined);
    };
    poll();
    const timer = window.setInterval(poll, 30_000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [user]);

  const queued = health?.queue?.queued ?? 0;
  const running = health?.queue?.running ?? 0;

  if (!user) return <SignIn onSignedIn={setUser} />;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3l7.5 3v6c0 4.4-3 7.9-7.5 9-4.5-1.1-7.5-4.6-7.5-9V6z" />
              <path d="M9 12l2.2 2.2L15.5 10" />
            </svg>
          </span>
          <h1>TRACE</h1>
          <span className="qualifier">Automation Platform</span>
          {environment && (
            <span className="env" title="The deployment this dashboard is pointed at">
              <span className="dot" />
              {environment}
            </span>
          )}
          <span className="spacer" />
          <span className="meta">
            {health === null
              ? 'checking backend...'
              : health.status === 'unreachable'
                ? 'backend unreachable'
                : `backend ${health.status} - ${running} running, ${queued} queued`}
          </span>
          <span className="who" title={`${user.email} (${user.role})`}>
            {user.email} <span className="role-chip">{user.role}</span>
          </span>
          <button
            type="button"
            className={`help-button${view.name === 'help' ? ' active' : ''}`}
            onClick={() => navigate({ name: 'help', section: 'user' })}
            title="How to use this, and how it's built"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="12" cy="12" r="9" />
              <path d="M9.5 9.5a2.5 2.5 0 113.5 2.3c-.7.3-1 .9-1 1.7" />
              <path d="M12 17v.5" />
            </svg>
            Help
          </button>
          <button type="button" className="linkish" onClick={() => api.logout()}>
            Sign out
          </button>
        </div>
        <nav>
          <button
            type="button"
            className={view.name === 'usecases' || view.name === 'usecase' ? 'active' : ''}
            onClick={() => navigate({ name: 'usecases' })}
            title="Recorded steps that replay without an LLM"
          >
            Use cases
          </button>
          <button
            type="button"
            className={view.name === 'record' ? 'active' : ''}
            onClick={() => navigate({ name: 'record' })}
            title="Do the task once by hand; no model, no tokens"
          >
            Record
          </button>
          <button
            type="button"
            className={view.name === 'history' ? 'active' : ''}
            onClick={() => navigate({ name: 'history' })}
          >
            Runs
          </button>
          <button
            type="button"
            className={view.name === 'targets' ? 'active' : ''}
            onClick={() => navigate({ name: 'targets' })}
            title="Which site each use case runs against, in this deployment"
          >
            Targets
          </button>
          <button
            type="button"
            className={view.name === 'memory' ? 'active' : ''}
            onClick={() => navigate({ name: 'memory' })}
            title="Locators that broke, and what fixed them"
          >
            Learned fixes
          </button>
        </nav>
      </header>

      <main
        className={
          view.name === 'run' || view.name === 'usecase' || view.name === 'help'
            ? 'page'
            : 'page narrow'
        }
      >
        {view.name === 'record' && !session.can('usecase:create') && (
          <div className="banner">
            Your role ({user.role}) cannot record workflows. Ask an administrator for the
            operator role.
          </div>
        )}

        {view.name === 'record' && session.can('usecase:create') && (
          <NewUseCase onSaved={(usecaseId) => navigate({ name: 'usecase', usecaseId })} />
        )}

        {view.name === 'memory' && <HealingMemory />}

        {view.name === 'targets' && <Targets />}

        {view.name === 'history' && (
          <RunHistory onOpen={(runId) => navigate({ name: 'run', runId })} />
        )}

        {view.name === 'run' && (
          <RunView
            runId={view.runId}
            onBack={() => navigate({ name: 'history' })}
            onOpenUseCase={(usecaseId) => navigate({ name: 'usecase', usecaseId })}
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

        {view.name === 'help' && (
          <Help section={view.section} onSection={(section) => navigate({ name: 'help', section })} />
        )}
      </main>
    </div>
  );
}
