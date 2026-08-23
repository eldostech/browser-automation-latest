import { useEffect, useState } from 'react';
import { api, ApiError } from '../lib/api';
import type { CreateRunPayload } from '../lib/api';
import type { ServerConfig } from '../lib/events';

interface Props {
  onStarted: (runId: string) => void;
}

const EXAMPLE_TASK =
  'Search for wireless headphones under $50 and export the top 5 results as JSON ' +
  'with fields: name, price, rating, url.';

export function TaskComposer({ onStarted }: Props) {
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [task, setTask] = useState('');
  const [startUrl, setStartUrl] = useState('');
  const [maxSteps, setMaxSteps] = useState(30);
  const [timeoutSeconds, setTimeoutSeconds] = useState(300);
  const [allowedDomains, setAllowedDomains] = useState('');
  const [headless, setHeadless] = useState(true);
  const [requireApproval, setRequireApproval] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getConfig()
      .then((loaded) => {
        setConfig(loaded);
        setMaxSteps(loaded.defaults.max_steps);
        setTimeoutSeconds(loaded.defaults.timeout_seconds);
        setAllowedDomains(loaded.defaults.allowed_domains.join(', '));
        setHeadless(loaded.defaults.headless);
        setRequireApproval(loaded.defaults.require_approval);
      })
      .catch((err: Error) => setError(`Could not reach the backend: ${err.message}`));
  }, []);

  const domains = allowedDomains
    .split(',')
    .map((d) => d.trim())
    .filter(Boolean);

  async function submit(submitEvent: React.FormEvent) {
    submitEvent.preventDefault();
    setError(null);
    setSubmitting(true);

    const payload: CreateRunPayload = {
      task: task.trim(),
      start_url: startUrl.trim() || null,
      max_steps: maxSteps,
      timeout_seconds: timeoutSeconds,
      allowed_domains: domains,
      require_approval: requireApproval,
      headless,
    };

    try {
      const { run_id } = await api.createRun(payload);
      onStarted(run_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  const wildcardOnly = domains.length === 1 && domains[0] === '*';

  return (
    <form className="card" onSubmit={submit}>
      <h2>New run</h2>
      <p className="hint">
        The agent plans and executes this task against a real browser, one step at a time.
        {config ? ` Model: ${config.model} (${config.provider}). Transport: ${config.transport}.` : ''}
      </p>

      {error && <div className="banner error">{error}</div>}

      <div className="field">
        <label htmlFor="task">Task</label>
        <textarea
          id="task"
          value={task}
          placeholder={EXAMPLE_TASK}
          onChange={(e) => setTask(e.target.value)}
          required
        />
      </div>

      <div className="field">
        <label htmlFor="start-url">Starting URL (optional)</label>
        <input
          id="start-url"
          value={startUrl}
          placeholder="https://example.com"
          onChange={(e) => setStartUrl(e.target.value)}
        />
      </div>

      <div className="field">
        <label htmlFor="domains">Allowed domains</label>
        <input
          id="domains"
          value={allowedDomains}
          placeholder="example.com, *.example.com"
          onChange={(e) => setAllowedDomains(e.target.value)}
        />
        {wildcardOnly && (
          <div className="banner error" style={{ marginTop: 10, marginBottom: 0 }}>
            <strong>*</strong> disables the allowlist. The agent will be able to navigate
            anywhere, including sites you do not control. Keep a real list unless you know
            exactly why you need this.
          </div>
        )}
      </div>

      <div className="grid-2">
        <div className="field">
          <label htmlFor="max-steps">Max steps</label>
          <input
            id="max-steps"
            type="number"
            min={1}
            max={200}
            value={maxSteps}
            onChange={(e) => setMaxSteps(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="timeout">Time budget (seconds)</label>
          <input
            id="timeout"
            type="number"
            min={10}
            max={3600}
            value={timeoutSeconds}
            onChange={(e) => setTimeoutSeconds(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="grid-2">
        <label className="checkbox" htmlFor="headless">
          <input
            id="headless"
            type="checkbox"
            checked={headless}
            onChange={(e) => setHeadless(e.target.checked)}
          />
          <span>
            <strong>Headless</strong>
            Uncheck to watch the real browser window on the machine running the backend.
          </span>
        </label>

        <label className="checkbox" htmlFor="approval">
          <input
            id="approval"
            type="checkbox"
            checked={requireApproval}
            onChange={(e) => setRequireApproval(e.target.checked)}
          />
          <span>
            <strong>Require approval for sensitive actions</strong>
            Pauses on form submits, credentials, payments, deletions and off-allowlist
            navigation.
          </span>
        </label>
      </div>

      <div className="row end" style={{ marginTop: 8 }}>
        <button className="primary" type="submit" disabled={submitting || !task.trim()}>
          {submitting ? 'Starting...' : 'Start run'}
        </button>
      </div>
    </form>
  );
}
