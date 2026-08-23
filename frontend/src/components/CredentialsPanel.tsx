import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { CredentialSummary } from '../lib/events';
import { formatRelative } from '../lib/format';

interface Props {
  /** Slot names the current use case needs, so the form can prefill them. */
  requiredSlots?: string[];
  onChange?: () => void;
}

/**
 * Create and remove the credentials a use case signs in with.
 *
 * Values are write-only: they go to the backend, are encrypted at rest, and no
 * endpoint returns them. This form is the only place they are ever typed, and
 * it never displays one back.
 */
export function CredentialsPanel({ requiredSlots = [], onChange }: Props) {
  const [rows, setRows] = useState<CredentialSummary[]>([]);
  const [available, setAvailable] = useState(true);
  const [name, setName] = useState('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [extraSlot, setExtraSlot] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const response = await api.listCredentials();
      setRows(response.credentials);
      setAvailable(response.vault_available);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const slots = Array.from(new Set([...requiredSlots, ...Object.keys(values)]));

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const filled = Object.fromEntries(
        Object.entries(values).filter(([, value]) => value.trim() !== ''),
      );
      await api.createCredential(name.trim(), filled);
      setName('');
      setValues({});
      await load();
      onChange?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    setBusy(true);
    try {
      await api.deleteCredential(id);
      await load();
      onChange?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const canSave =
    name.trim() !== '' && Object.values(values).some((value) => value.trim() !== '');

  return (
    <div className="card">
      <h3>Credentials</h3>

      {!available && (
        <div className="banner error">
          <strong>Credential storage is switched off</strong>
          <p style={{ margin: '6px 0' }}>
            The backend has no <code>CREDENTIALS_KEY</code>, so it refuses to store passwords
            rather than writing them to disk in the clear. Any use case that signs in cannot run
            until you set one.
          </p>
          <p style={{ margin: '6px 0 0' }}>Generate a key:</p>
          <pre className="snippet">
{`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`}
          </pre>
          <p style={{ margin: '6px 0 0' }}>
            Put it in <code>.env</code> as <code>CREDENTIALS_KEY=...</code> and restart the
            backend. Keep it with your other secrets — losing it makes stored credentials
            unreadable.
          </p>
        </div>
      )}

      {error && <div className="banner error">{error}</div>}

      {rows.length > 0 && (
        <table className="runs" style={{ marginBottom: 16 }}>
          <thead>
            <tr>
              <th>Name</th>
              <th>Slots</th>
              <th>Last used</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.name}</td>
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                  {row.slots.join(', ')}
                </td>
                <td style={{ color: 'var(--text-faint)', whiteSpace: 'nowrap' }}>
                  {row.last_used_at ? formatRelative(row.last_used_at) : 'never'}
                </td>
                <td>
                  <button type="button" className="link danger" onClick={() => remove(row.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <fieldset disabled={!available} style={{ border: 0, padding: 0, margin: 0 }}>
        <p className="hint">
          One credential covers a whole sign-in. Values are encrypted immediately and are never
          shown again — not here, and not in any run's timeline.
        </p>

        <label>
          <span>Name</span>
          <input
            type="text"
            value={name}
            placeholder="IXL account"
            onChange={(e) => setName(e.target.value)}
          />
        </label>

        {slots.map((slot) => (
          <label key={slot}>
            <span>{slot}</span>
            <input
              type="password"
              autoComplete="new-password"
              value={values[slot] ?? ''}
              onChange={(e) => setValues({ ...values, [slot]: e.target.value })}
            />
          </label>
        ))}

        <label>
          <span>Add another slot</span>
          <span style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              value={extraSlot}
              placeholder="pin"
              onChange={(e) => setExtraSlot(e.target.value)}
            />
            <button
              type="button"
              onClick={() => {
                const slot = extraSlot.trim();
                if (slot) setValues({ ...values, [slot]: '' });
                setExtraSlot('');
              }}
            >
              Add
            </button>
          </span>
        </label>

        <button type="button" className="primary" onClick={save} disabled={busy || !canSave}>
          {busy ? 'Saving...' : 'Save credential'}
        </button>
      </fieldset>
    </div>
  );
}
