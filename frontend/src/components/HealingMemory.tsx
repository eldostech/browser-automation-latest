/**
 * What the system has learned about broken locators.
 *
 * Healing writes here on its own when it is confident. This screen is the
 * other half: what it learned, whether that was right, and taking it out when
 * it was not.
 *
 * **Forgetting is the important button.** A fix that was right last month and
 * wrong now does not fail loudly — it gets recalled, put in front of the model
 * as precedent, and quietly makes the next repair worse. Somebody has to be
 * able to see what is in here and take it back out.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { RememberedFix } from '../lib/events';

export function HealingMemory() {
  const [fixes, setFixes] = useState<RememberedFix[]>([]);
  const [domain, setDomain] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setFixes((await api.listFixes(domain || undefined)).fixes);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [domain]);

  useEffect(() => {
    void load();
  }, [load]);

  const forget = async (id: string) => {
    try {
      await api.forgetFix(id);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const domains = useMemo(
    () => Array.from(new Set(fixes.map((f) => f.domain))).sort(),
    [fixes],
  );

  return (
    <div className="card">
      <h3>What this workspace has learned</h3>
      <p className="hint">
        When a step breaks and the repair is confirmed, it is recorded here. The next time
        something breaks on the same site, these go to the model as evidence — so a redesign
        costs one call across every workflow that hits it, rather than one per workflow.
      </p>

      {error && <p className="error">{error}</p>}

      {domains.length > 1 && (
        <div className="dataset-list">
          <button
            type="button"
            className={domain === '' ? 'chip active' : 'chip'}
            onClick={() => setDomain('')}
          >
            Every site
          </button>
          {domains.map((name) => (
            <button
              key={name}
              type="button"
              className={domain === name ? 'chip active' : 'chip'}
              onClick={() => setDomain(name)}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      {loading && <p className="hint">Reading…</p>}

      {!loading && fixes.length === 0 && (
        <p className="hint">
          Nothing learned yet. This fills in the first time a locator breaks and somebody —
          or the model — works out what changed.
        </p>
      )}

      {fixes.length > 0 && (
        <table className="mapping">
          <thead>
            <tr>
              <th>Site</th>
              <th>What changed</th>
              <th>Confirmed by</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {fixes.map((fix) => (
              <tr key={fix.id}>
                <td>
                  <code>{fix.domain}</code>
                  <div className="hint">step {fix.step_id}</div>
                </td>
                <td>
                  {fix.explanation}
                  <div className="hint examples">
                    {fix.old_locator?.name ?? '—'} → {fix.new_locator?.name ?? '—'}
                  </div>
                </td>
                <td className="hint">
                  {/* A fix somebody looked at outranks one nobody checked, and
                      the ranking in the prompt depends on this distinction. */}
                  {fix.confirmed_by === 'model' ? 'the model' : fix.confirmed_by}
                </td>
                <td>
                  <button type="button" onClick={() => forget(fix.id)} title="Stop recalling this">
                    Forget
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
