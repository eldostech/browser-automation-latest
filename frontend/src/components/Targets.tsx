/**
 * Where a use case actually points, in this deployment.
 *
 * A use case names a target — `schemora`, `bank` — and each deployment answers
 * that name with its own address. The definition carries no URL, so promoting a
 * workflow from dev to production moves nothing about where it runs.
 *
 * **This is deliberately a screen and not a config file.** The base URL used to
 * come from a map in the environment, which works while every workflow in an
 * environment shares one site and needs a variable and a release for each one
 * after that. The people who run the workflows change these, in the environment
 * they are running them in.
 */

import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { Target } from '../lib/events';

export function Targets() {
  const [targets, setTargets] = useState<Target[]>([]);
  const [name, setName] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setTargets((await api.listTargets()).targets);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    try {
      await api.saveTarget(name.trim(), baseUrl.trim(), description.trim());
      setNotice(`Saved ${name.trim()}.`);
      setName('');
      setBaseUrl('');
      setDescription('');
      setError(null);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const remove = async (target: string) => {
    try {
      await api.deleteTarget(target);
      // Deliberately not a warning about use cases naming it: they are left
      // alone and refuse on their next run, which says exactly what is wrong.
      setNotice(`Removed ${target}. Use cases naming it will refuse to run.`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const editing = targets.find((t) => t.name === name.trim());

  return (
    <div className="panel">
      <h2>Targets</h2>
      <p className="hint">
        A use case names a target; this deployment says where that target is. The same use
        case can then run in dev, UAT and production without the definition changing — and a
        single run can be pointed somewhere else entirely when it is started.
      </p>

      {error && <div className="banner error">{error}</div>}
      {notice && <div className="banner info">{notice}</div>}

      <div className="card">
        <h3>{editing ? `Change ${editing.name}` : 'Add a target'}</h3>
        <div className="grid-2">
          <label className="field">
            <span>Name</span>
            <input
              value={name}
              placeholder="schemora"
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label className="field">
            <span>Address here</span>
            <input
              value={baseUrl}
              placeholder="https://uat.schemora.ai"
              onChange={(e) => setBaseUrl(e.target.value)}
            />
          </label>
        </div>
        <label className="field">
          <span>What it is (optional)</span>
          <input
            value={description}
            placeholder="The product's UAT deployment"
            onChange={(e) => setDescription(e.target.value)}
          />
        </label>
        <p className="hint">
          An origin, not a page: <code>https://uat.schemora.ai</code>, never
          <code> https://uat.schemora.ai/login</code>. Each use case&rsquo;s steps carry their
          own paths.
        </p>
        <button
          type="button"
          className="primary"
          disabled={!name.trim() || !baseUrl.trim()}
          onClick={() => void save()}
        >
          {editing ? 'Update it' : 'Add it'}
        </button>
      </div>

      {loading ? (
        <p className="hint">Loading…</p>
      ) : targets.length === 0 ? (
        <div className="empty-state">
          <p>No targets yet.</p>
          <p className="hint">
            Without one, a use case runs against the address it was recorded on. That is fine
            for a single environment and is what you want to change before promoting anything.
          </p>
        </div>
      ) : (
        <table className="mapping">
          <thead>
            <tr>
              <th>Name</th>
              <th>Address here</th>
              <th>What it is</th>
              <th>Last changed</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {targets.map((target) => (
              <tr key={target.name}>
                <td>
                  <code>{target.name}</code>
                </td>
                <td>
                  <code>{target.base_url}</code>
                </td>
                <td>{target.description || <span className="hint">—</span>}</td>
                <td className="hint">{target.updated_by || '—'}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => {
                      setName(target.name);
                      setBaseUrl(target.base_url);
                      setDescription(target.description);
                    }}
                  >
                    Edit
                  </button>{' '}
                  <button type="button" className="danger" onClick={() => void remove(target.name)}>
                    Remove
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
