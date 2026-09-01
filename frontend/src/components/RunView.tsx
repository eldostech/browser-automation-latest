/**
 * One execution, as it happens.
 *
 * A run is something you watch now, not something you steer. It replays steps
 * a person recorded and reviewed, so there is nothing to approve mid-run and
 * no task to compose — what used to live here (the approval bar, the "keep
 * these credentials?" prompt, the name-your-recording form) all belonged to an
 * agent working out what to do next.
 *
 * What is left is the part that was always the point: a step-by-step timeline
 * with a screenshot, so a person can see what the browser actually did to
 * record 700.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { RunDetail, RunFinishedEvent, ScreenshotEvent } from '../lib/events';
import { useRunStream, isTerminal } from '../lib/useRunStream';
import { formatDuration } from '../lib/format';
import { ResultPanel } from './ResultPanel';
import { ScreenshotPane } from './ScreenshotPane';
import { StepTrail } from './StepTrail';
import { ConnectionIndicator, StatusBadge } from './StatusBadge';
import { Timeline } from './Timeline';

interface Props {
  runId: string;
  onBack: () => void;
  onOpenUseCase?: (usecaseId: string) => void;
}

export function RunView({ runId, onBack, onOpenUseCase }: Props) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stream = useRunStream(runId);

  const load = useCallback(async () => {
    try {
      setDetail(await api.getRun(runId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  // Re-read the record once the stream says it is over: the terminal event
  // carries the outcome, but the row carries the counts and the artifacts.
  useEffect(() => {
    if (stream.finished) {
      void load();
      setTab('trail');
    }
  }, [stream.finished, load]);

  const cancel = async () => {
    try {
      await api.cancelRun(runId);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const running = !isTerminal(stream.status) && !stream.finished;
  const usecaseId = detail?.options?.usecase_id as string | undefined;

  // Every screenshot the run produced, so a person can step back through the
  // rows rather than only seeing the last one.
  const shots = useMemo(
    () => stream.events.filter((e): e is ScreenshotEvent => e.type === 'screenshot'),
    [stream.events],
  );
  const [selected, setSelected] = useState<number | null>(null);
  // Live events while it runs; the recorded trail once it has. The trail
  // is the one that can show what changed since last time.
  const [tab, setTab] = useState<'live' | 'trail'>('live');
  const finishedEvent = useMemo(
    () =>
      (stream.events.find((e) => e.type === 'run_finished') as RunFinishedEvent | undefined) ??
      null,
    [stream.events],
  );

  return (
    <div className="run-view">
      <header className="run-header">
        <button type="button" onClick={onBack}>
          &larr; Back
        </button>
        <h2>{detail?.task ?? 'Run'}</h2>
        <StatusBadge status={stream.status} />
        <ConnectionIndicator state={stream.connection} />
        {detail?.duration_ms ? <span className="meta">{formatDuration(detail.duration_ms)}</span> : null}
        <span className="spacer" />
        {usecaseId && onOpenUseCase && (
          <button type="button" onClick={() => onOpenUseCase(usecaseId)}>
            Open the workflow
          </button>
        )}
        {running && (
          <button type="button" onClick={cancel}>
            Stop
          </button>
        )}
      </header>

      {error && <p className="error">{error}</p>}

      <div className="tabs">
        <button
          type="button"
          className={tab === 'live' ? 'active' : ''}
          onClick={() => setTab('live')}
        >
          Timeline
        </button>
        <button
          type="button"
          className={tab === 'trail' ? 'active' : ''}
          onClick={() => setTab('trail')}
          title="Each step beside the last run that worked"
        >
          Compare with last run
        </button>
      </div>

      {tab === 'trail' && (
        <StepTrail runId={runId} usecaseId={usecaseId} />
      )}

      {tab === 'live' && (
      <div className="grid-2">
        <section className="panel scroll">
          <Timeline events={stream.events} />
        </section>
        <section className="panel">
          <ScreenshotPane
            screenshot={selected === null ? stream.latestScreenshot : shots[selected]}
            history={shots}
            selectedIndex={selected}
            onSelect={setSelected}
          />
          <ResultPanel finished={finishedEvent} running={running} />
        </section>
      </div>
      )}
    </div>
  );
}
