import { useEffect, useRef, useState } from 'react';
import type { DistillResult } from '../lib/events';

interface Props {
  result: DistillResult;
  busy?: boolean;
  onConfirm: (name: string) => void;
  onCancel: () => void;
}

/**
 * Confirm the name of a use case that has just been recorded.
 *
 * The suggestion can only exist *after* distillation — the model reads the
 * recording to write it — so this is the first moment a name can be offered.
 * It is pre-filled and selected, so keeping it costs one keypress and
 * replacing it costs none of the usual "clear the field first" friction.
 */
export function NameUseCase({ result, busy, onConfirm, onCancel }: Props) {
  const [name, setName] = useState(result.suggested_name || result.name);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    input.current?.focus();
    input.current?.select();
  }, []);

  const suggestion = result.suggested_name || result.name;
  const trimmed = name.trim();

  return (
    <div className="card naming">
      <h3>Name this use case</h3>
      <p className="hint">
        Recorded {result.setup_steps} setup step{result.setup_steps === 1 ? '' : 's'} and{' '}
        {result.row_steps} per-row step{result.row_steps === 1 ? '' : 's'}. The name is just a
        label — you can change it later.
      </p>

      <label>
        <span>Name</span>
        <input
          ref={input}
          type="text"
          value={name}
          maxLength={200}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && trimmed) onConfirm(trimmed);
            if (event.key === 'Escape') setName(suggestion);
          }}
        />
      </label>

      {trimmed !== suggestion && (
        <p className="hint">
          Suggested: <code>{suggestion}</code>{' '}
          <button type="button" className="link" onClick={() => setName(suggestion)}>
            use that
          </button>
        </p>
      )}

      <div className="row">
        <button
          type="button"
          className="primary"
          disabled={busy || !trimmed}
          onClick={() => onConfirm(trimmed)}
        >
          {busy ? 'Saving...' : 'Save and review'}
        </button>
        <button type="button" onClick={onCancel} disabled={busy}>
          Skip
        </button>
      </div>
    </div>
  );
}
