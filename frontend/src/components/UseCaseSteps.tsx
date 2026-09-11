import { useState } from 'react';
import type { Locator, LocatorCheckReport, UseCaseStep } from '../lib/events';
import { LocatorEditor, describeLocator, isBrittle } from './LocatorEditor';

// `describeLocator` used to live here and now lives beside the editor, so the
// list and the thing that edits it cannot disagree about what a rung says.
export { describeLocator };

function summarise(step: UseCaseStep): string {
  if (step.description) return step.description;
  if (step.action === 'navigate') return `go to ${step.url}`;
  if (step.action === 'assert' && step.assert) {
    const base = `${step.assert.kind} ${step.assert.value ?? ''}`.trim();
    return step.assert.negate ? `NOT ${base}` : base;
  }
  if (step.locators.length) return describeLocator(step.locators[0]);
  return step.action;
}

/** Which field a step's single "value" line actually edits, by action. */
function editableField(step: UseCaseStep): 'url' | 'value' | null {
  if (step.action === 'navigate') return 'url';
  if (step.value != null) return 'value';
  return null;
}

interface Props {
  title: string;
  hint?: string;
  steps: UseCaseStep[];
  onRemove?: (stepId: string) => void;
  /** Saves a single step field (a wrong recorded URL, a typo'd typed value)
   * as a new draft version. Absent means read-only, same as today. */
  onEditField?: (stepId: string, field: 'url' | 'value', value: string) => void;
  /** Saves a rewritten locator ladder as a new draft version.
   *
   * A recorded ladder is what codegen happened to write, and a healed one is
   * what a model picked off a page. Both are usually right and neither is
   * always right — and before this the only remedy for one wrong rung was
   * re-recording the whole workflow. */
  onEditLocators?: (stepId: string, locators: Locator[]) => void;
  /** Opens a page and reports what each rung matches. Absent hides the check,
   * which is what a deployment with no browser should show. */
  onCheckLocators?: (locators: Locator[], url: string) => Promise<LocatorCheckReport>;
  /** Where the step under review runs, so the check has somewhere to open. */
  pageUrlFor?: (stepId: string) => string;
}

/**
 * One phase of a use case. Shows the locator ladder rather than hiding it:
 * which rung a step relies on is the single best predictor of whether it will
 * still work next month.
 */
export function UseCaseSteps({
  title,
  hint,
  steps,
  onRemove,
  onEditField,
  onEditLocators,
  onCheckLocators,
  pageUrlFor,
}: Props) {
  // Which step is mid-edit, and the draft text for it. One at a time: a step
  // list is reviewed top to bottom, and editing two at once just makes it
  // easy to lose track of which unsaved change belongs to which step.
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  // Which step's ladder is open. Separate from `editing` above because the
  // two edit different things and a step can want either.
  const [editingLocators, setEditingLocators] = useState<string | null>(null);

  const startEdit = (step: UseCaseStep) => {
    const field = editableField(step);
    if (!field) return;
    setEditing(step.id);
    setDraft((field === 'url' ? step.url : step.value) ?? '');
  };

  const save = (step: UseCaseStep) => {
    const field = editableField(step);
    if (field && onEditField) onEditField(step.id, field, draft);
    setEditing(null);
  };

  if (steps.length === 0) {
    return (
      <div className="panel" style={{ marginBottom: 16 }}>
        <header>
          <span>{title}</span>
        </header>
        <div className="body">
          <div className="empty-state" style={{ padding: 12 }}>
            No steps in this phase.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ marginBottom: 16 }}>
      <header>
        <span>{title}</span>
        <span style={{ marginLeft: 'auto', color: 'var(--text-faint)' }}>
          {steps.length} step{steps.length === 1 ? '' : 's'}
        </span>
      </header>
      <div className="body">
        {hint && <p className="hint" style={{ marginTop: 0 }}>{hint}</p>}
        <ol className="usecase-steps">
          {steps.map((step) => (
            <li key={step.id}>
              <div className="step-head">
                <code className="action">{step.action}</code>
                <span className="summary">{summarise(step)}</span>
                {step.optional && <span className="tag">optional</span>}
                {onEditField && editableField(step) && editing !== step.id && (
                  <button
                    type="button"
                    className="link"
                    onClick={() => startEdit(step)}
                    title={
                      step.action === 'navigate'
                        ? 'Fix the recorded URL'
                        : 'Fix the recorded value'
                    }
                  >
                    Edit
                  </button>
                )}
                {onRemove && (
                  <button
                    type="button"
                    className="link danger"
                    onClick={() => onRemove(step.id)}
                    title="Remove this step from the use case"
                  >
                    Remove
                  </button>
                )}
              </div>

              {editing === step.id ? (
                <div className="step-detail step-detail-edit">
                  <span className="label">{step.action === 'navigate' ? 'url' : 'value'}</span>
                  <input
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') save(step);
                      if (e.key === 'Escape') setEditing(null);
                    }}
                  />
                  <button type="button" className="linkish" onClick={() => save(step)}>
                    Save
                  </button>
                  <button type="button" className="link" onClick={() => setEditing(null)}>
                    Cancel
                  </button>
                </div>
              ) : (
                (step.value || step.url) && (
                  <div className="step-detail">
                    <span className="label">value</span>
                    <code>{step.value ?? step.url}</code>
                  </div>
                )
              )}

              {step.fields.length > 0 && (
                <div className="step-detail">
                  <span className="label">fields</span>
                  <span>
                    {step.fields.map((field) => (
                      <code key={field.name} style={{ marginRight: 8 }}>
                        {field.name}={field.value}
                      </code>
                    ))}
                  </span>
                </div>
              )}

              {/* Where the step's result lands. For an extract or a download
                  that is the whole point of the step, and a review screen that
                  does not show it makes the reviewer open the JSON. */}
              {step.output && (
                <div className="step-detail">
                  <span className="label">saves as</span>
                  <code>{step.output}</code>
                </div>
              )}

              {/* Which fields a discovery step reads out of each row. Without
                  these the step reads as "finds some rows" and a reviewer
                  cannot tell whether it captures the identifier they need. */}
              {(step.columns ?? []).length > 0 && (
                <div className="step-detail">
                  <span className="label">reads</span>
                  <span>
                    {step.columns!.map((column) => (
                      <code key={column.name} style={{ marginRight: 8 }}>
                        {column.name}={column.selector}
                        {column.attribute ? `@${column.attribute}` : ''}
                      </code>
                    ))}
                  </span>
                </div>
              )}

              {editingLocators === step.id && onEditLocators ? (
                <LocatorEditor
                  stepId={step.id}
                  locators={step.locators}
                  defaultUrl={pageUrlFor?.(step.id) ?? ''}
                  onCheck={
                    onCheckLocators ??
                    (async () => ({ page_url: '', page_title: '', results: [] }))
                  }
                  onSave={(locators) => {
                    onEditLocators(step.id, locators);
                    setEditingLocators(null);
                  }}
                  onCancel={() => setEditingLocators(null)}
                />
              ) : (
                step.locators.length > 0 && (
                  <div className="step-detail">
                    <span className="label">finds it by</span>
                    <span>
                      {step.locators.map((locator, index) => (
                        <code
                          key={index}
                          className={isBrittle(locator) ? 'locator brittle' : 'locator'}
                          title={
                            index === 0
                              ? 'Tried first'
                              : `Fallback ${index}: used only if the ones above stop matching`
                          }
                        >
                          {describeLocator(locator)}
                        </code>
                      ))}
                    </span>
                    {onEditLocators && (
                      <button
                        type="button"
                        className="link"
                        onClick={() => setEditingLocators(step.id)}
                        title="Change how this step finds the element, and check it against a real page before saving"
                      >
                        Edit
                      </button>
                    )}
                  </div>
                )
              )}

              {/* What the recording said and the draft took out. Shown because a
                  reviewer deciding whether that was the right call needs to see
                  it -- a locator made of row one's data reads perfectly well,
                  and "role=option" on its own does not explain itself. Never
                  executed: these are not fallbacks. */}
              {(step.rejected_locators ?? []).length > 0 && (
                <div className="step-detail">
                  <span className="label">not used</span>
                  <span>
                    {step.rejected_locators.map((locator, index) => (
                      <code
                        key={index}
                        className="locator rejected"
                        title="Recorded, then taken out of this step. It is never tried."
                      >
                        {describeLocator(locator)}
                      </code>
                    ))}
                  </span>
                </div>
              )}

              {step.action === 'script' && (
                <details className="step-script">
                  <summary>Raw JavaScript — read this before enabling scripts</summary>
                  <pre>{step.code}</pre>
                </details>
              )}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
