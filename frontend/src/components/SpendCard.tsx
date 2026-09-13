/**
 * What this workspace has spent with a model, and what it may.
 *
 * It sits on the Targets screen because that is the one place this product
 * keeps settings a deployment owns, and a spending ceiling is exactly that. It
 * is not on an admin screen because there is not one; when there is, this moves
 * whole rather than being rebuilt.
 *
 * Two things it is careful about. **No ceiling and a ceiling of zero are
 * different**, and the copy says which you have — an installation that never
 * thought about spending should not be told it has a budget it did not set.
 * And the figure is **an estimate**: it is priced from this application's own
 * table, not from AWS, and saying so is the difference between a guard rail
 * and a number somebody reconciles against an invoice.
 */

import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { WorkspaceSpend } from '../lib/events';
import { session } from '../lib/session';

export function SpendCard() {
  const [spend, setSpend] = useState<WorkspaceSpend | null>(null);
  const [limit, setLimit] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const body = await api.getSpend();
      setSpend(body);
      setLimit(body.limit_usd === null ? '' : String(body.limit_usd));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const save = async () => {
    const trimmed = limit.trim();
    const next = trimmed === '' ? null : Number(trimmed);
    if (next !== null && (Number.isNaN(next) || next < 0)) {
      setError('A monthly limit is a number of dollars, or empty for no limit.');
      return;
    }
    setError(null);
    try {
      const body = await api.setSpendLimit(next);
      setSpend(body);
      setNotice(
        next === null
          ? 'Limit removed. Sessions are bounded only by their own budgets.'
          : `Limit set to $${next.toFixed(2)} a month.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (!spend) return null;

  const canSet = session.can('user:manage');
  const used = spend.usd;
  const limitUsd = spend.limit_usd;
  const share = limitUsd && limitUsd > 0 ? Math.min(1, used / limitUsd) : 0;

  return (
    <div className="card">
      <h3>Spending</h3>
      <p className="hint">
        Two things here can spend money: an agent recording a workflow, and a Guided
        replay asking a model to re-find a control that moved. A Strict replay cannot
        reach a model at all, so it contributes nothing and says so.
      </p>

      {error && <div className="banner error">{error}</div>}
      {notice && <div className="banner">{notice}</div>}

      <div className="spend-figure">
        <span className="amount">${used.toFixed(2)}</span>
        <span className="hint">
          across {spend.runs} run{spend.runs === 1 ? '' : 's'} this month
          {spend.tokens > 0 && ` · ${spend.tokens.toLocaleString()} tokens`}
        </span>
      </div>

      {limitUsd !== null && limitUsd > 0 && (
        <div className="spend-bar" aria-hidden="true">
          <span style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
      )}

      <label className="field" style={{ maxWidth: 240 }}>
        <span>Monthly limit (USD)</span>
        <input
          type="number"
          min={0}
          step={5}
          value={limit}
          placeholder="no limit"
          disabled={!canSet}
          onChange={(e) => setLimit(e.target.value)}
          onBlur={() => void save()}
        />
      </label>
      <p className="hint">
        {limitUsd === null
          ? 'No limit is set, so a session is bounded only by its own budget. Empty means no limit; zero means stop all spending, which is a different thing.'
          : `A session started here can spend at most what is left of this, whatever budget it asks for. It resets on the first of the month.`}
        {' '}These figures are this application&rsquo;s own estimate of what a model turn
        costs. The AWS bill is the authority.
      </p>
      {!canSet && (
        <p className="hint">Only an administrator can change the limit.</p>
      )}
    </div>
  );
}
