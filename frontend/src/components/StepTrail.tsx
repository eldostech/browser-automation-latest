/**
 * What the browser did, step by step, beside what it did last time.
 *
 * The timeline next to this one streams events while a run is in flight. This
 * is the finished article: one row per step, read back from `run_steps`, with
 * the screenshot this run took and the one from the last run that worked.
 *
 * **It opens on the first divergence.** A thousand-row batch produces tens of
 * thousands of steps, and "scroll until something looks wrong" is not a
 * workflow. The server names the first step that failed or moved more than a
 * couple of percent, and this starts there.
 *
 * **The ratio is shown in words.** Four decimal places is not a thing to make
 * somebody read, and the number is not a verdict anyway: a rendered clock
 * changes a few pixels and means nothing, a form that silently failed to
 * submit can change very few and mean everything. A person decides.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { RunStep } from '../lib/events';
import { AuthedImage } from './AuthedImage';
import { formatDuration } from '../lib/format';
import { BrandSpinner } from './BrandSpinner';

type Props = {
  runId: string;
  /** Stamped on anything the user records, so a fix traces to its recipe. */
  usecaseId?: string;
};

const STATUS_LABEL: Record<string, string> = {
  succeeded: 'ok',
  failed: 'failed',
  skipped: 'skipped',
  healed: 'healed',
};

export function StepTrail({ runId, usecaseId }: Props) {
  const [steps, setSteps] = useState<RunStep[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const body = await api.getRunSteps(runId);
      setSteps(body.steps);
      // Open where something changed, rather than at the top.
      setSelected(body.first_divergence ?? (body.steps.length ? body.steps[0].seq : null));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  const current = useMemo(
    () => steps.find((step) => step.seq === selected) ?? null,
    [steps, selected],
  );

  // What a person worked out when the model could not. Recording it means the
  // next run has a human-confirmed precedent, which outranks anything the
  // model guessed on its own.
  const [explanation, setExplanation] = useState('');
  const [saved, setSaved] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const remember = async () => {
    if (!current || !explanation.trim()) return;
    setSaving(true);
    try {
      const body = await api.rememberFix({
        step_id: current.step_id,
        // The page this step happened on, which is what files the fix
        // against the right site.
        page_url: current.page_url,
        step_summary: `${current.action} ${current.locator}`.trim(),
        wanted: current.locator,
        explanation: explanation.trim(),
        usecase_id: usecaseId ?? null,
      });
      setSaved(`Recorded for ${body.domain}. The next run will be shown it.`);
      setExplanation('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <BrandSpinner state="working" label="Reading the step trail…" />;
  if (error) return <p className="error">{error}</p>;
  if (!steps.length) {
    return (
      <p className="hint">
        No steps were recorded for this run. Screenshots are off, or it failed before it
        started.
      </p>
    );
  }

  return (
    <div className="step-trail">
      <ol className="steps">
        {steps.map((step) => (
          <li key={step.id} className={step.seq === selected ? 'selected' : ''}>
            <button type="button" onClick={() => setSelected(step.seq)}>
              <span className={`dot ${step.status}`} aria-hidden="true" />
              <span className="what">
                <code>{step.action}</code> {step.locator || step.step_id}
              </span>
              <span className="meta">
                {STATUS_LABEL[step.status] ?? step.status}
                {step.duration_ms ? ` · ${formatDuration(step.duration_ms)}` : ''}
                {step.row_index !== null ? ` · row ${step.row_index + 1}` : ''}
              </span>
              {/* Only worth showing when there is something to say. */}
              {step.pixel_diff !== null && step.pixel_diff > 0 && (
                <span className="drift">{step.diff}</span>
              )}
              {step.locator_rung !== null && step.locator_rung > 0 && (
                <span className="drift" title="The preferred locator no longer matched">
                  locator drifted
                </span>
              )}
            </button>
          </li>
        ))}
      </ol>

      <div className="compare">
        {current === null ? (
          <p className="hint">Pick a step.</p>
        ) : (
          <>
            <h4>
              {current.action} {current.locator || current.step_id} —{' '}
              {STATUS_LABEL[current.status] ?? current.status}
            </h4>
            {current.error && <p className="error">{current.error}</p>}

            {current.status === 'failed' && (
              <div className="teach">
                <h5>Do you know what changed?</h5>
                <p className="hint">
                  One sentence, for whoever sees this next — &ldquo;they renamed the button
                  and moved it into the dialog&rdquo;. It is kept against this site and shown
                  to the repair the next time something here breaks.
                </p>
                <textarea
                  rows={2}
                  value={explanation}
                  placeholder="What changed on the page?"
                  onChange={(event) => setExplanation(event.target.value)}
                />
                <button
                  type="button"
                  className="primary"
                  disabled={saving || !explanation.trim() || !current.page_url}
                  onClick={remember}
                  title={current.page_url ? undefined : 'This step recorded no page URL'}
                >
                  {saving ? 'Recording…' : 'Remember this'}
                </button>
                {saved && <p className="hint">{saved}</p>}
              </div>
            )}

            <div className="shots">
              <figure>
                <figcaption>Last time it worked</figcaption>
                {current.baseline_url ? (
                  <AuthedImage src={current.baseline_url} alt="the baseline page" />
                ) : (
                  <p className="hint">
                    Nothing to compare against — this step has not succeeded before.
                  </p>
                )}
              </figure>
              <figure>
                <figcaption>
                  This run{current.pixel_diff !== null ? ` — ${current.diff}` : ''}
                </figcaption>
                {current.screenshot_url ? (
                  <AuthedImage src={current.screenshot_url} alt="the page this run" />
                ) : (
                  <p className="hint">No screenshot for this step.</p>
                )}
              </figure>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
