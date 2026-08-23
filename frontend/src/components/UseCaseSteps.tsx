import type { Locator, UseCaseStep } from '../lib/events';

/** How a locator reads in the review UI. */
export function describeLocator(locator: Locator): string {
  switch (locator.strategy) {
    case 'role':
      return locator.name ? `role=${locator.role} "${locator.name}"` : `role=${locator.role}`;
    case 'css':
      return `css ${locator.selector}`;
    case 'text':
      return `text "${locator.text}"`;
    default:
      return `nth ${locator.nth ?? 0}`;
  }
}

/** The rungs that break on cosmetic change. Worth flagging before a batch runs. */
function isBrittle(locator: Locator): boolean {
  return locator.strategy === 'text' || locator.strategy === 'nth';
}

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

interface Props {
  title: string;
  hint?: string;
  steps: UseCaseStep[];
  onRemove?: (stepId: string) => void;
}

/**
 * One phase of a use case. Shows the locator ladder rather than hiding it:
 * which rung a step relies on is the single best predictor of whether it will
 * still work next month.
 */
export function UseCaseSteps({ title, hint, steps, onRemove }: Props) {
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

              {(step.value || step.url) && (
                <div className="step-detail">
                  <span className="label">value</span>
                  <code>{step.value ?? step.url}</code>
                </div>
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

              {step.locators.length > 0 && (
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
