/**
 * Keep or discard the credentials a recording used.
 *
 * Shown once, immediately before the recording becomes a use case, because
 * that is the only moment the choice exists: the values have been held in the
 * backend's memory since the run started and are dropped the instant this
 * resolves either way.
 *
 * The dialog offers two outcomes rather than three. "Decide later" would be a
 * lie — there is no later — so cancelling backs out of recording entirely and
 * leaves the values held until they expire.
 */

import { useState } from 'react';

interface Props {
  /** Slot names only. The values are on the server and never come back here. */
  slots: string[];
  /** Suggested name for the saved credential, usually the site. */
  suggestion?: string;
  busy?: boolean;
  onDecide: (saveAs: string | null) => void;
  onCancel: () => void;
}

export function KeepCredentials({
  slots,
  suggestion = '',
  busy = false,
  onDecide,
  onCancel,
}: Props) {
  const [name, setName] = useState(suggestion);

  return (
    <div className="card">
      <h3>Keep the sign-in details?</h3>
      <p className="hint">
        This recording used {slots.length === 1 ? 'a credential' : 'credentials'} for{' '}
        {slots.map((slot) => (
          <code key={slot} style={{ marginRight: 6 }}>
            {slot}
          </code>
        ))}
        . They have not been written anywhere yet.
      </p>

      <div className="field">
        <label htmlFor="credential-name">Save them as</label>
        <input
          id="credential-name"
          value={name}
          placeholder="Example account"
          disabled={busy}
          onChange={(e) => setName(e.target.value)}
        />
        <p className="hint">
          Encrypted, and reusable by any use case in this workspace that needs the same
          slots. You will not be able to read them back — only use them.
        </p>
      </div>

      <div className="row end" style={{ gap: 8 }}>
        <button type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
        <button type="button" onClick={() => onDecide(null)} disabled={busy}>
          {busy ? 'Working…' : 'Discard them'}
        </button>
        <button
          type="button"
          className="primary"
          disabled={busy || !name.trim()}
          onClick={() => onDecide(name.trim())}
        >
          {busy ? 'Working…' : 'Save and record'}
        </button>
      </div>
    </div>
  );
}
