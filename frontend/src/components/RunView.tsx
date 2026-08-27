import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { DistillResult, RunDetail, ScreenshotEvent } from '../lib/events';
import { useRunStream, isTerminal } from '../lib/useRunStream';
import { formatDuration } from '../lib/format';
import { ApprovalBar } from './ApprovalBar';
import { KeepCredentials } from './KeepCredentials';
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
  /** Opens the use case a replay run was executing. */
  onOpenUseCase?: (usecaseId: string) => void;
}

/**
 * Live run view. The same component replays a finished run: the WebSocket
 * replays the persisted history from seq 0 and then closes, so there is one
 * rendering path instead of two.
 */
/** The site a run started on, as a credential-name suggestion. */
function hostOf(url: string | null | undefined): string {
  if (!url) return '';
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return '';
  }
}

export function RunView({ runId, onBack, onRecorded, onOpenUseCase }: Props) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pinnedShot, setPinnedShot] = useState<number | null>(null);
  const [recording, setRecording] = useState(false);
  // Held between distillation and the name being confirmed: the model's
  // suggestion cannot exist until the recording has been read.
  const [recorded, setRecorded] = useState<DistillResult | null>(null);
  // Credential slots this run still holds values for. Asked once the run has
  // finished, so the record button can offer the choice rather than silently
  // discarding a login the user just typed.
  const [heldSlots, setHeldSlots] = useState<string[]>([]);
  const [decidingCredentials, setDecidingCredentials] = useState(false);

  const stream = useRunStream(runId);

  /**
   * Did this run *execute* a stored use case, rather than record a new one?
   *
   * The two look identical in the timeline, but only one of them can become a
   * use case. Offering "Save as use case" on a replay invited spending an LLM
   * call to derive a use case from a use case -- a copy of the original with
   * one row's values already baked into its steps. The API refuses it; this
   * stops the button being there to press.
   */
  const isReplay = Boolean(detail?.options?.replay);
  const sourceUseCaseId =
    typeof detail?.options?.usecase_id === 'string' ? detail.options.usecase_id : null;

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
    // Names only -- there is no endpoint that returns a held value. An empty
    // list is the normal answer both for a run that used no credentials and
    // for one whose values have since expired; either way there is nothing to
    // offer to save.
    api
      .heldCredentialSlots(runId)
      .then(({ slots }) => setHeldSlots(slots))
      .catch(() => setHeldSlots([]));
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
  const distil = useCallback(
    async (saveCredentialAs: string | null) => {
      setRecording(true);
      setActionError(null);
      try {
        setRecorded(await api.distillRun(runId, { saveCredentialAs }));
        // Whichever way it went, the backend has now consumed them.
        setHeldSlots([]);
        setDecidingCredentials(false);
      } catch (error) {
        setActionError(error instanceof Error ? error.message : String(error));
      } finally {
        setRecording(false);
      }
    },
    [runId],
  );

  /**
   * Turn this run into a use case. When the recording used credentials, the
   * choice of whether to keep them has to be made first -- distilling is what
   * consumes them, so afterwards is too late.
   */
  const record = useCallback(async () => {
    if (heldSlots.length > 0) {
      setDecidingCredentials(true);
      return;
    }
    await distil(null);
  }, [heldSlots, distil]);

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
        {status === 'succeeded' && !isReplay && (
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
        {isReplay && sourceUseCaseId && (
          <button
            type="button"
            onClick={() => onOpenUseCase?.(sourceUseCaseId)}
            title="This run executed a stored use case"
          >
            Open the use case
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

      {decidingCredentials && (
        <KeepCredentials
          slots={heldSlots}
          suggestion={hostOf(detail?.start_url) || ''}
          busy={recording}
          onDecide={distil}
          onCancel={() => setDecidingCredentials(false)}
        />
      )}

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
