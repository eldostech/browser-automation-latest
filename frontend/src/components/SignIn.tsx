/**
 * Sign-in.
 *
 * Shown instead of the application whenever there is no session. Deliberately
 * plain: one job, no navigation away from it, and no hints about which half of
 * a failed attempt was wrong -- the server does not distinguish them either.
 */

import { useState, type FormEvent } from 'react';

import { api, ApiError } from '../lib/api';
import type { CurrentUser } from '../lib/session';

interface Props {
  onSignedIn: (user: CurrentUser) => void;
}

export function SignIn({ onSignedIn }: Props) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.login(email.trim(), password));
    } catch (exc) {
      setError(
        exc instanceof ApiError
          ? exc.message
          : 'Could not reach the server. Is the backend running?',
      );
      setPassword('');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="signin">
      <form className="signin-card" onSubmit={submit}>
        <h1>Understudy</h1>
        <p className="signin-sub">Sign in to record and run browser tasks.</p>

        <label htmlFor="signin-email">Email</label>
        <input
          id="signin-email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="username"
          autoFocus
          required
          disabled={busy}
        />

        <label htmlFor="signin-password">Password</label>
        <input
          id="signin-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          required
          disabled={busy}
        />

        {/* role="alert" so a screen reader announces a failed attempt. */}
        {error && (
          <p className="signin-error" role="alert">
            {error}
          </p>
        )}

        <button type="submit" disabled={busy || !email || !password}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>

        <p className="signin-hint">
          No account yet? The first administrator is created on the backend's first
          start, and its one-time password is printed to the server log.
        </p>
      </form>
    </div>
  );
}
