import { useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { ModelCatalogue, ModelChoice, ModelInfo } from '../lib/events';
import { onChoiceChanged, readChoice, writeChoice } from '../lib/model';

/**
 * Choose which model the next thing you do will spend on.
 *
 * Two providers, and they are offered differently because they *are*
 * different. Bedrock is a short configured list. OpenRouter is several hundred
 * models fetched live, so it gets a filter box rather than a dropdown you
 * scroll: picking out `anthropic/claude-sonnet-4.5` from four hundred entries
 * by eye is not a thing a select element is good at.
 *
 * The choice lives in this browser and travels on each request. It is not
 * server state, deliberately -- see `lib/model.ts` for why.
 *
 * Price is shown because it is the number that decides what you are willing to
 * try. OpenRouter's range spans four orders of magnitude, and a model picked
 * for accuracy without seeing its price is how a comparison becomes a bill.
 */

interface Props {
  onClose: () => void;
}

function priceLabel(model: ModelInfo): string {
  if (model.input_per_million === null || model.output_per_million === null) {
    return 'price not published';
  }
  const money = (value: number) => (value >= 1 ? value.toFixed(2) : value.toFixed(3));
  return `$${money(model.input_per_million)} in / $${money(model.output_per_million)} out per M`;
}

function contextLabel(model: ModelInfo): string {
  if (!model.context) return '';
  return model.context >= 1000 ? `${Math.round(model.context / 1000)}k context` : `${model.context} context`;
}

export function ModelPicker({ onClose }: Props) {
  const [catalogue, setCatalogue] = useState<ModelCatalogue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<string>(() => readChoice()?.provider ?? '');
  const [filter, setFilter] = useState('');
  const [checking, setChecking] = useState<string | null>(null);
  const [checked, setChecked] = useState<Record<string, string>>({});

  const chosen = readChoice();

  useEffect(() => {
    let alive = true;
    api
      .models()
      .then((body) => {
        if (!alive) return;
        setCatalogue(body);
        // Default the provider tab to whatever is already chosen, else to the
        // deployment's own default, so the list opens where you left it.
        setProvider((current) => current || chosen?.provider || body.default.provider);
      })
      .catch((err) => alive && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const forProvider = useMemo(() => {
    if (!catalogue) return [];
    const wanted = filter.trim().toLowerCase();
    return catalogue.models
      .filter((model) => model.provider === provider)
      .filter(
        (model) =>
          !wanted ||
          model.id.toLowerCase().includes(wanted) ||
          model.name.toLowerCase().includes(wanted),
      );
  }, [catalogue, provider, filter]);

  const choose = (model: ModelInfo) => {
    writeChoice({ provider: model.provider, model: model.id });
    onClose();
  };

  const check = async (model: ModelInfo) => {
    setChecking(model.id);
    try {
      const result = await api.checkModel(model.provider, model.id);
      setChecked((previous) => ({
        ...previous,
        [model.id]: result.ok ? 'reachable' : result.error || 'could not be called',
      }));
    } catch (err) {
      setChecked((previous) => ({
        ...previous,
        [model.id]: err instanceof Error ? err.message : String(err),
      }));
    } finally {
      setChecking(null);
    }
  };

  return (
    <div className="panel model-picker">
      <header>
        <span>Model for this browser</span>
        <button type="button" className="link" style={{ marginLeft: 'auto' }} onClick={onClose}>
          Close
        </button>
      </header>
      <div className="body">
        <p className="hint" style={{ marginTop: 0 }}>
          Applies to everything this browser starts from now on — recording with AI,
          repairing a step, and healing during a run. It is not shared with anyone else,
          so two people can compare two models at the same time.
        </p>

        {error && <div className="banner error">{error}</div>}
        {!catalogue && !error && <p className="hint">Reading what this deployment can reach…</p>}

        {catalogue && (
          <>
            <div className="model-providers">
              {catalogue.providers.map((name) => {
                const problem = catalogue.problems[name];
                return (
                  <button
                    key={name}
                    type="button"
                    className={`model-tab${provider === name ? ' active' : ''}`}
                    onClick={() => setProvider(name)}
                    disabled={Boolean(problem)}
                    title={problem || `Models from ${name}`}
                  >
                    {name}
                    {problem && <span className="model-tab-note"> unavailable</span>}
                  </button>
                );
              })}
              <span style={{ marginLeft: 'auto' }}>
                {chosen ? (
                  <button type="button" className="link" onClick={() => { writeChoice(null); onClose(); }}>
                    Use the deployment default ({catalogue.default.provider}:{catalogue.default.model || 'none set'})
                  </button>
                ) : (
                  <span className="hint">
                    Using the deployment default: {catalogue.default.provider}
                    {catalogue.default.model ? `:${catalogue.default.model}` : ' (no model set)'}
                  </span>
                )}
              </span>
            </div>

            {catalogue.problems[provider] && (
              <div className="banner warn">{catalogue.problems[provider]}</div>
            )}

            {forProvider.length > 8 && (
              <label className="rung-field rung-wide" style={{ marginTop: 10 }}>
                <span>find a model</span>
                <input
                  autoFocus
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                  placeholder="claude, gpt, llama, qwen…"
                />
              </label>
            )}

            <ul className="model-list">
              {forProvider.map((model) => {
                const active = chosen?.provider === model.provider && chosen?.model === model.id;
                const verdict = checked[model.id];
                return (
                  <li key={model.id} className={active ? 'model-row active' : 'model-row'}>
                    <div className="model-row-main">
                      <button type="button" className="model-name" onClick={() => choose(model)}>
                        {model.name}
                      </button>
                      <code className="model-id">{model.id}</code>
                    </div>
                    <div className="model-row-meta">
                      <span>{priceLabel(model)}</span>
                      {contextLabel(model) && <span>{contextLabel(model)}</span>}
                      <button
                        type="button"
                        className="link"
                        disabled={checking === model.id}
                        onClick={() => check(model)}
                        title="Make one tiny call to see whether this deployment can actually use it"
                      >
                        {checking === model.id ? 'Checking…' : 'Check'}
                      </button>
                      {verdict && (
                        <span className={verdict === 'reachable' ? 'model-ok' : 'model-bad'}>
                          {verdict}
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>

            {forProvider.length === 0 && !catalogue.problems[provider] && (
              <p className="hint">
                {filter ? `Nothing matching “${filter}”.` : 'No models listed for this provider.'}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** The chrome's read-only view of the choice, with a way in. */
export function ModelBadge({ onOpen }: { onOpen: () => void }) {
  const [choice, setChoice] = useState<ModelChoice | null>(() => readChoice());

  useEffect(() => {
    // Subscribed rather than read once: the badge and the requests must never
    // disagree about which model is in play, and the picker is what changes it.
    return onChoiceChanged(setChoice);
  }, []);

  const label = choice ? choice.model.split('/').pop() || choice.model : 'default model';
  return (
    <button
      type="button"
      className="model-badge"
      onClick={onOpen}
      title={
        choice
          ? `Everything this browser starts uses ${choice.provider}:${choice.model}. Click to change.`
          : 'Using the model this deployment is configured for. Click to choose another.'
      }
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" />
      </svg>
      {label}
    </button>
  );
}
