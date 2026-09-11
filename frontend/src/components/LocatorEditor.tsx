import { useState } from 'react';
import type { Locator, LocatorCheckReport, LocatorStrategy } from '../lib/events';

/**
 * Editing a step's locator ladder by hand.
 *
 * Recording produces a ladder from what `playwright codegen` happened to write,
 * and healing produces one from what a model picked off the page. Both are
 * usually right and neither is always right, and until now the only remedy for
 * a wrong rung was re-recording the whole workflow — which throws away every
 * other step to fix one.
 *
 * Two things make this an editor rather than a JSON field:
 *
 * **The ladder is ordered, and the order is the point.** Rungs are tried top
 * to bottom and the first to match exactly one element wins, so moving a rung
 * is as much of an edit as changing one. Showing them as a list you can
 * reorder says that; a textarea does not.
 *
 * **Nothing is saved until it has been checked.** "Check on a page" resolves
 * the ladder against a live page through the executor's own composer and says
 * what each rung actually matches. A person editing a locator without that is
 * guessing, and the cost of guessing wrong is a batch that fails on row one.
 */

const STRATEGIES: { value: LocatorStrategy; label: string; hint: string }[] = [
  { value: 'role', label: 'role + name', hint: 'What the control is and what it is called. Survives a redesign best.' },
  { value: 'label', label: 'label', hint: "A form field's visible label." },
  { value: 'placeholder', label: 'placeholder', hint: 'The greyed-out text inside an empty field.' },
  { value: 'alt_text', label: 'alt text', hint: "An image's alt attribute." },
  { value: 'test_id', label: 'test id', hint: "A data-testid the site's own authors maintain." },
  { value: 'text', label: 'text', hint: 'Any element containing this wording. Breaks on a rewording.' },
  { value: 'css', label: 'CSS selector', hint: 'Markup, not meaning. Breaks whenever the markup does.' },
];

/** Which single field this strategy carries, if it carries one. */
function valueField(strategy: LocatorStrategy): 'text' | 'selector' | null {
  if (strategy === 'css') return 'selector';
  if (strategy === 'role' || strategy === 'nth') return null;
  return 'text';
}

function blankLocator(): Locator {
  return { strategy: 'role', role: 'button', name: '', exact: false, nth: 0 };
}

/** The rungs that break on cosmetic change. Worth flagging before a batch runs. */
export function isBrittle(locator: Locator): boolean {
  if (locator.strategy === 'text' || locator.strategy === 'nth' || locator.strategy === 'css') {
    return true;
  }
  return locator.within ? isBrittle(locator.within) : false;
}

/** How a locator reads in the review UI. Mirrors `Locator.describe()`. */
export function describeLocator(locator: Locator): string {
  let base: string;
  switch (locator.strategy) {
    case 'role':
      base = locator.name ? `role=${locator.role} "${locator.name}"` : `role=${locator.role}`;
      break;
    case 'css':
      base = `css=${locator.selector}`;
      break;
    case 'nth':
      base = `nth=${locator.nth ?? 0}`;
      break;
    default:
      base = `${locator.strategy}='${locator.text}'`;
  }
  if (locator.exact && locator.strategy !== 'css' && locator.strategy !== 'test_id') {
    base += ' exact';
  }
  if (locator.has_text) base += ` has_text='${locator.has_text}'`;
  if ((locator.nth ?? 0) !== 0 && locator.strategy !== 'nth') base += ` [${locator.nth}]`;
  if (locator.within) base = `${base} in ${describeLocator(locator.within)}`;
  if (locator.frames?.length) base = `${base} in frame ${locator.frames.join(' > ')}`;
  return base;
}

interface RungProps {
  value: Locator;
  onChange: (next: Locator) => void;
  /** How deep this sits inside `within`. Bounded to match the schema. */
  depth: number;
}

const MAX_SCOPE_DEPTH = 3;

/** One rung's fields. Recursive, because `within` is another rung. */
function RungFields({ value, onChange, depth }: RungProps) {
  const field = valueField(value.strategy);
  const named = value.strategy !== 'css' && value.strategy !== 'test_id' && value.strategy !== 'nth';

  const set = (patch: Partial<Locator>) => onChange({ ...value, ...patch });

  return (
    <div className="rung-fields">
      <label className="rung-field">
        <span>finds it by</span>
        <select
          value={value.strategy}
          onChange={(e) => {
            // Clearing the other strategies' fields on the way through: a
            // `role` left behind on a `css` rung is dead weight the API
            // refuses, and it reads as though it still applies.
            const strategy = e.target.value as LocatorStrategy;
            set({
              strategy,
              role: strategy === 'role' ? value.role || 'button' : null,
              name: strategy === 'role' ? value.name ?? '' : null,
              text: valueField(strategy) === 'text' ? value.text ?? '' : null,
              selector: strategy === 'css' ? value.selector ?? '' : null,
            });
          }}
        >
          {STRATEGIES.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      {value.strategy === 'role' && (
        <>
          <label className="rung-field">
            <span>role</span>
            <input
              value={value.role ?? ''}
              onChange={(e) => set({ role: e.target.value })}
              placeholder="button"
            />
          </label>
          <label className="rung-field">
            <span>name</span>
            <input
              value={value.name ?? ''}
              onChange={(e) => set({ name: e.target.value })}
              placeholder="Sign in"
            />
          </label>
        </>
      )}

      {field && (
        <label className="rung-field">
          <span>{field === 'selector' ? 'selector' : 'text'}</span>
          <input
            value={(field === 'selector' ? value.selector : value.text) ?? ''}
            onChange={(e) => set({ [field]: e.target.value } as Partial<Locator>)}
            placeholder={field === 'selector' ? '#order-form button' : 'Sign in'}
          />
        </label>
      )}

      {named && (
        <label className="rung-field rung-check" title="Playwright matches a name as a substring unless told otherwise, so 'Invite' also finds '+ Invite User'.">
          <input
            type="checkbox"
            checked={value.exact ?? false}
            onChange={(e) => set({ exact: e.target.checked })}
          />
          <span>whole name only</span>
        </label>
      )}

      <label className="rung-field" title="Keep only matches containing this text. How a row is picked out of a table.">
        <span>containing</span>
        <input
          value={value.has_text ?? ''}
          onChange={(e) => set({ has_text: e.target.value || null })}
          placeholder="(any text)"
        />
      </label>

      <label className="rung-field rung-narrow" title="Which of several identical matches. 0 means no position is given, and a step whose rung matches several is refused rather than guessing.">
        <span>position</span>
        <input
          type="number"
          min={-1}
          value={value.nth ?? 0}
          onChange={(e) => set({ nth: Number(e.target.value) || 0 })}
        />
      </label>

      {depth === 0 && (
        <label className="rung-field" title="iframes to descend through, outermost first, as CSS selectors. An element inside a frame is not on the page as far as every other rung is concerned.">
          <span>inside frames</span>
          <input
            value={(value.frames ?? []).join(' > ')}
            onChange={(e) =>
              set({
                frames: e.target.value
                  .split('>')
                  .map((part) => part.trim())
                  .filter(Boolean),
              })
            }
            placeholder="(not in a frame)"
          />
        </label>
      )}

      {/* The scope. This is the answer to almost every real ambiguity — "the
          Invite button in the dialog", "the Edit link in the Acme row" — and
          before it existed the only alternatives were a positional index,
          which is a claim about ordering, and refusing the step. */}
      <div className="rung-scope">
        {value.within ? (
          <>
            <div className="rung-scope-head">
              <span className="label">inside</span>
              <button type="button" className="link danger" onClick={() => set({ within: null })}>
                Search the whole page instead
              </button>
            </div>
            <RungFields
              value={value.within}
              onChange={(next) => set({ within: next })}
              depth={depth + 1}
            />
          </>
        ) : (
          depth < MAX_SCOPE_DEPTH - 1 && (
            <button
              type="button"
              className="link"
              onClick={() => set({ within: { strategy: 'role', role: 'dialog', name: '', nth: 0 } })}
            >
              Only look inside a particular element
            </button>
          )
        )}
      </div>
    </div>
  );
}

interface Props {
  stepId: string;
  locators: Locator[];
  /** Where to open when checking. Usually the step's own page. */
  defaultUrl: string;
  onCheck: (locators: Locator[], url: string) => Promise<LocatorCheckReport>;
  onSave: (locators: Locator[]) => void;
  onCancel: () => void;
}

export function LocatorEditor({ stepId, locators, defaultUrl, onCheck, onSave, onCancel }: Props) {
  const [draft, setDraft] = useState<Locator[]>(() =>
    locators.length ? locators.map((l) => ({ ...l })) : [blankLocator()],
  );
  const [url, setUrl] = useState(defaultUrl);
  const [report, setReport] = useState<LocatorCheckReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Any edit invalidates the last check. Showing a green tick beside a rung
  // that has been changed since it was checked is worse than showing nothing:
  // it is the screen asserting something it does not know.
  const edit = (index: number, next: Locator) => {
    setReport(null);
    setDraft(draft.map((rung, i) => (i === index ? next : rung)));
  };

  const move = (index: number, by: number) => {
    const to = index + by;
    if (to < 0 || to >= draft.length) return;
    const next = [...draft];
    [next[index], next[to]] = [next[to], next[index]];
    setReport(null);
    setDraft(next);
  };

  const remove = (index: number) => {
    setReport(null);
    setDraft(draft.filter((_, i) => i !== index));
  };

  const check = async () => {
    setChecking(true);
    setError(null);
    try {
      setReport(await onCheck(draft, url));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setReport(null);
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="locator-editor">
      <p className="hint">
        Rungs are tried from the top, and the first that matches exactly one element is
        used. A rung that matches several is refused rather than guessing, so the ones
        below it are what the step falls back on when a site changes.
      </p>

      <ol className="rungs">
        {draft.map((rung, index) => {
          const verdict = report?.results[index];
          return (
            <li key={index} className={verdict ? (verdict.ok ? 'rung ok' : 'rung bad') : 'rung'}>
              <div className="rung-head">
                <span className="rung-position">{index === 0 ? 'tried first' : `fallback ${index}`}</span>
                <code className={isBrittle(rung) ? 'locator brittle' : 'locator'}>
                  {describeLocator(rung)}
                </code>
                <span className="rung-actions">
                  <button type="button" className="link" disabled={index === 0} onClick={() => move(index, -1)}>
                    Up
                  </button>
                  <button
                    type="button"
                    className="link"
                    disabled={index === draft.length - 1}
                    onClick={() => move(index, 1)}
                  >
                    Down
                  </button>
                  <button
                    type="button"
                    className="link danger"
                    disabled={draft.length === 1}
                    onClick={() => remove(index)}
                  >
                    Remove
                  </button>
                </span>
              </div>

              <RungFields value={rung} onChange={(next) => edit(index, next)} depth={0} />

              {verdict && (
                <p className="rung-verdict">
                  {verdict.ok
                    ? `Matches one element: ${verdict.matches[0] ?? '(no text)'}`
                    : verdict.reason}
                  {verdict.total !== verdict.visible && (
                    <span className="rung-counts">
                      {' '}
                      ({verdict.total} in the page, {verdict.visible} visible)
                    </span>
                  )}
                </p>
              )}
            </li>
          );
        })}
      </ol>

      <div className="rung-add">
        <button type="button" className="link" onClick={() => setDraft([...draft, blankLocator()])}>
          Add a fallback
        </button>
      </div>

      <div className="locator-check">
        <label className="rung-field rung-wide">
          <span>check against</span>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/the-page-this-step-runs-on"
          />
        </label>
        <button type="button" className="linkish" disabled={checking || !url} onClick={check}>
          {checking ? 'Opening the page…' : 'Check on a page'}
        </button>
      </div>

      {error && <p className="error-text">{error}</p>}
      {report && (
        <p className="hint">
          Checked against {report.page_title || report.page_url}. Nothing was clicked or
          typed, and nothing has been saved yet.
        </p>
      )}

      <div className="locator-editor-actions">
        <button type="button" className="primary" onClick={() => onSave(draft)}>
          Save {stepId} as a new version
        </button>
        <button type="button" className="link" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
