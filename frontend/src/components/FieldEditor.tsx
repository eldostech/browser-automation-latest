/**
 * The values a recording will use, named one row at a time.
 *
 * The instruction and the data used to be one blob of prose. Separating them
 * is what makes two later things possible: the recording can be parameterised
 * without anyone guessing which words were data, and a password can be handled
 * as a password rather than as part of a sentence.
 *
 * A row marked **secret** behaves differently in three places, and the UI has
 * to make that legible rather than leaving it as a checkbox nobody reads:
 * the value is masked here, the agent is shown a placeholder instead of it,
 * and the finished use case asks for a *credential* rather than a CSV column.
 */

import type { DeclaredField } from '../lib/api';

interface Props {
  fields: DeclaredField[];
  onChange: (fields: DeclaredField[]) => void;
  disabled?: boolean;
}

const BLANK: DeclaredField = { name: '', value: '', secret: false, description: '' };

/** Suggests a usable field name from a label someone might type. */
export function toFieldName(raw: string): string {
  const cleaned = raw
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  // It becomes a `{{input.x}}` template and a CSV header, so it must start
  // with a letter.
  return /^[a-z]/.test(cleaned) ? cleaned : cleaned ? `f_${cleaned}` : '';
}

export function FieldEditor({ fields, onChange, disabled = false }: Props) {
  function update(index: number, patch: Partial<DeclaredField>) {
    onChange(fields.map((field, i) => (i === index ? { ...field, ...patch } : field)));
  }

  function remove(index: number) {
    onChange(fields.filter((_, i) => i !== index));
  }

  function add(secret: boolean) {
    onChange([...fields, { ...BLANK, secret }]);
  }

  const duplicates = new Set(
    fields
      .map((f) => f.name)
      .filter((name, i, all) => name && all.indexOf(name) !== i),
  );

  return (
    <div className="field">
      <label>Values this task uses</label>
      <p className="hint">
        Name each value the instruction refers to. Names become the input columns of the
        reusable use case, so use the name you want in your spreadsheet later. Mark a value
        secret and it is never shown to the model, never stored, and never written to the
        run log.
      </p>

      {fields.length > 0 && (
        <div className="fieldrows">
          {fields.map((field, index) => (
            <div className={`fieldrow${field.secret ? ' secret' : ''}`} key={index}>
              <input
                aria-label="Field name"
                className="fieldrow-name"
                placeholder={field.secret ? 'password' : 'full_name'}
                value={field.name}
                disabled={disabled}
                onChange={(e) => update(index, { name: e.target.value })}
                onBlur={(e) => update(index, { name: toFieldName(e.target.value) })}
              />
              <input
                aria-label={field.secret ? 'Secret value' : 'Value'}
                className="fieldrow-value"
                // type="password" keeps it off the screen and out of the
                // browser's autofill history for ordinary form fields.
                type={field.secret ? 'password' : 'text'}
                autoComplete={field.secret ? 'new-password' : 'off'}
                placeholder={field.secret ? '••••••••' : 'Nitin Asati'}
                value={field.value}
                disabled={disabled}
                onChange={(e) => update(index, { value: e.target.value })}
              />
              <label className="fieldrow-secret" title="Never shown to the model or stored">
                <input
                  type="checkbox"
                  checked={field.secret}
                  disabled={disabled}
                  onChange={(e) => update(index, { secret: e.target.checked })}
                />
                <span>secret</span>
              </label>
              <button
                type="button"
                className="fieldrow-remove"
                aria-label={`Remove ${field.name || 'field'}`}
                disabled={disabled}
                onClick={() => remove(index)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {duplicates.size > 0 && (
        <p className="hint" style={{ color: 'var(--danger)' }}>
          Duplicate name{duplicates.size > 1 ? 's' : ''}: {[...duplicates].join(', ')}. Each
          value needs its own name.
        </p>
      )}

      <div className="row" style={{ gap: 8, marginTop: 8 }}>
        <button type="button" onClick={() => add(false)} disabled={disabled}>
          + Input field
        </button>
        <button type="button" onClick={() => add(true)} disabled={disabled}>
          + Credential
        </button>
      </div>

      {fields.some((f) => f.secret) && (
        <p className="hint" style={{ marginTop: 10 }}>
          Credentials are held only while this run is in flight. When you save the recording
          as a use case you can choose to keep them; otherwise they are discarded the moment
          you decide.
        </p>
      )}
    </div>
  );
}
