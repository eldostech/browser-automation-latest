import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { DistillResult, RunDetail, ScreenshotEvent } from '../lib/events';
import { useRunStream, isTerminal } from '../lib/useRunStream';
import { formatDuration } from '../lib/format';
import { ApprovalBar } from './ApprovalBar';
import { NameUseCase } from './NameUseCase';
import { ResultPanel } from './ResultPanel';
import { ScreenshotPane } from './ScreenshotPane';
import { ConnectionIndicator, StatusBadge } from './StatusBadge';
import { Timeline } from './Timeline';

interface Props {
  runId: string;
  onBack: () => void;
  /** Opens the review screen for a use case distilled from this run. */
  onRecorded?: (usecaseId: string) => void;
}

/**
 * Live run view. The same component replays a finished run: the WebSocket
 * replays the persisted history from seq 0 and then closes, so there is one
 * rendering path instead of two.
 */
export function RunView({ runId, onBack, onRecorded }: Props) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pinnedShot, setPinnedShot] = useState<number | null>(null);
  const [recording, setRecording] = useState(false);
  // Held between distillation and the name being confirmed: the model's
  // suggestion cannot exist until the recording has been read.
  const [recorded, setRecorded] = useState<DistillResult | null>(null);

  const stream = useRunStream(runId);

  useEffect(() => {
    let alive = true;
    api
      .getRun(runId)
      .then((loaded) => alive && setDetail(loaded))
      .catch((error: Error) => alive && setActionError(error.message));
    return () => {
      alive = false;
    };
  }, [runId]);

  // Refresh the stored record once the stream reports a terminal state, so the
  // header shows persisted values rather than derived ones.
  useEffect(() => {
    if (!stream.finished) return;
    api.getRun(runId).then(setDetail).catch(() => undefined);
  }, [stream.finished, runId]);

  const screenshots = useMemo(
    () => stream.events.filter((e) => e.type === 'screenshot') as ScreenshotEvent[],
    [stream.events],
  );

  // Following live means "stay pinned to the newest shot".
  useEffect(() => {
    setPinnedShot(null);
  }, [runId]);

  const cancel = useCallback(async () => {
    setActionError(null);
    try {
      await api.cancelRun(runId);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    }
  }, [runId]);

  const decide = useCallback(
    async (decision: 'approve' | 'reject', note?: string) => {
      if (!stream.pendingApproval) return;
      setActionError(null);
      try {
        await api.resolveApproval(runId, stream.pendingApproval.approval_id, decision, note);
      } catch (error) {
        setActionError(error instanceof Error ? error.message : String(error));
      }
    },
    [runId, stream.pendingApproval],
  );

  /**
   * Turn this run into a reusable use case. This is the only LLM call the
   * replay feature ever makes: everything the use case is later run with
   * costs nothing.
   */
  const record = useCallback(async () => {
    setRecording(true);
    setActionError(null);
    try {
      setRecorded(await api.distillRun(runId));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setRecording(false);
    }
  }, [runId, onRecorded]);

  /** Apply the confirmed name, then open the use case for review. */
  const confirmName = useCallback(
    async (name: string) => {
      if (!recorded) return;
      setRecording(true);
      try {
        if (name !== recorded.name) {
          await api.renameUseCase(recorded.usecase_id, name);
        }
        onRecorded?.(recorded.usecase_id);
      } catch (error) {
        setActionError(error instanceof Error ? error.message : String(error));
      } finally {
        setRecording(false);
      }
    },
    [recorded, onRecorded],
  );

  const status = stream.status;
  const running = !isTerminal(status);

  return (
    <div>
      <div className="run-header">
        <button type="button" onClick={onBack}>
          Back
        </button>
        <div className="task">
          <div className="row">
            <StatusBadge status={status} />
            <ConnectionIndicator state={stream.connection} />
            {detail?.duration_ms != null && (
              <span style={{ fontSize: 12, color: 'var(--text-faint)' }}>
                {formatDuration(detail.duration_ms)}
              </span>
            )}
          </div>
          <p>{detail?.task ?? 'Loading task...'}</p>
        </div>
        {status === 'succeeded' && (
          <button
            type="button"
            className="primary"
            onClick={record}
            disabled={recording}
            title="Record these steps so they can be replayed with no LLM calls"
          >
            {recording ? 'Recording...' : 'Save as use case'}
          </button>
        )}
        {running && (
          <button type="button" className="danger" onClick={cancel}>
            Cancel run
          </button>
        )}
        {stream.connection === 'closed' && running && (
          <button type="button" onClick={stream.reconnect}>
            Reconnect
          </button>
        )}
      </div>

      {actionError && <div className="banner error">{actionError}</div>}

      {recorded && (
        <NameUseCase
          result={recorded}
          busy={recording}
          onConfirm={confirmName}
          onCancel={() => onRecorded?.(recorded.usecase_id)}
        />
      )}

      {stream.pendingApproval && <ApprovalBar approval={stream.pendingApproval} onDecide={decide} />}

      <div className="run-view">
        <div className="panel scroll">
          <header>
            <span>Steps</span>
            <span style={{ marginLeft: 'auto' }}>{stream.events.length} events</span>
          </header>
          <div className="body">
            <Timeline events={stream.events} autoScroll={running && pinnedShot === null} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <ScreenshotPane
            screenshot={stream.latestScreenshot}
            history={screenshots}
            selectedIndex={pinnedShot}
            onSelect={setPinnedShot}
          />
          <ResultPanel finished={stream.finished} running={running} />
        </div>
      </div>
    </div>
  );
}
