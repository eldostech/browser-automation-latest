/**
 * Live event stream for one run.
 *
 * Reconnection is lossless without server-side session state: every event
 * carries a monotonic `seq`, and the socket is reopened with
 * `?after_seq=<highest seen>`, so the backend replays exactly what was missed.
 *
 * Events are stored keyed by `seq`, which is also how streaming works --
 * a `thinking` block re-sends the same `seq` with more text, so it updates in
 * place instead of appending a new bubble (and the timeline never jumps).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { streamUrl } from './api';
import { isAgentEvent, TERMINAL_STATUSES } from './events';
import type {
  AgentEvent,
  ApprovalRequiredEvent,
  RunFinishedEvent,
  RunStatus,
  ScreenshotEvent,
} from './events';

export type ConnectionState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed';

const BACKOFF_MS = [500, 1000, 2000, 4000, 8000, 10000];

export interface RunStream {
  events: AgentEvent[];
  connection: ConnectionState;
  lastSeq: number;
  finished: RunFinishedEvent | null;
  status: RunStatus;
  pendingApproval: ApprovalRequiredEvent | null;
  latestScreenshot: ScreenshotEvent | null;
  reconnect: () => void;
  reset: () => void;
}

export function useRunStream(runId: string | null, initialStatus: RunStatus = 'pending'): RunStream {
  const [eventMap, setEventMap] = useState<Map<number, AgentEvent>>(() => new Map());
  const [connection, setConnection] = useState<ConnectionState>('idle');

  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const lastSeqRef = useRef(0);
  const closedForGoodRef = useRef(false);

  const reset = useCallback(() => {
    setEventMap(new Map());
    lastSeqRef.current = 0;
    retryRef.current = 0;
    closedForGoodRef.current = false;
  }, []);

  const connect = useCallback(() => {
    if (!runId || closedForGoodRef.current) return;

    setConnection(retryRef.current === 0 ? 'connecting' : 'reconnecting');
    const socket = new WebSocket(streamUrl(runId, lastSeqRef.current));
    socketRef.current = socket;

    socket.onopen = () => {
      retryRef.current = 0;
      setConnection('open');
    };

    socket.onmessage = (message) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(message.data as string);
      } catch {
        return;
      }
      // Heartbeats and any other transport frame are not agent events.
      if (!isAgentEvent(parsed)) return;

      const event = parsed;
      if (event.seq > lastSeqRef.current) lastSeqRef.current = event.seq;
      if (event.type === 'run_finished') closedForGoodRef.current = true;

      setEventMap((previous) => {
        const next = new Map(previous);
        next.set(event.seq, event);
        return next;
      });
    };

    socket.onerror = () => {
      // `onclose` always follows; the retry is scheduled there.
    };

    socket.onclose = (closeEvent) => {
      socketRef.current = null;
      // 1000 = clean close after run_finished, 4404 = unknown run.
      if (closedForGoodRef.current || closeEvent.code === 1000 || closeEvent.code === 4404) {
        setConnection('closed');
        return;
      }
      const delay = BACKOFF_MS[Math.min(retryRef.current, BACKOFF_MS.length - 1)];
      retryRef.current += 1;
      setConnection('reconnecting');
      timerRef.current = window.setTimeout(connect, delay + Math.random() * 250);
    };
  }, [runId]);

  useEffect(() => {
    reset();
    if (!runId) {
      setConnection('idle');
      return;
    }
    connect();
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      timerRef.current = null;
      const socket = socketRef.current;
      socketRef.current = null;
      closedForGoodRef.current = true;
      socket?.close();
    };
  }, [runId, connect, reset]);

  const reconnect = useCallback(() => {
    closedForGoodRef.current = false;
    retryRef.current = 0;
    connect();
  }, [connect]);

  const events = useMemo(
    () => [...eventMap.values()].sort((a, b) => a.seq - b.seq),
    [eventMap],
  );

  const finished = useMemo(
    () => (events.findLast((e) => e.type === 'run_finished') as RunFinishedEvent | undefined) ?? null,
    [events],
  );

  const pendingApproval = useMemo(() => {
    const requests = events.filter((e) => e.type === 'approval_required') as ApprovalRequiredEvent[];
    const resolved = new Set(
      events.filter((e) => e.type === 'approval_resolved').map((e) => (e as { approval_id: string }).approval_id),
    );
    return requests.findLast((request) => !resolved.has(request.approval_id)) ?? null;
  }, [events]);

  const latestScreenshot = useMemo(
    () => (events.findLast((e) => e.type === 'screenshot') as ScreenshotEvent | undefined) ?? null,
    [events],
  );

  const status: RunStatus = useMemo(() => {
    if (finished) return finished.status;
    if (pendingApproval) return 'awaiting_approval';
    if (events.length > 0) return 'running';
    return initialStatus;
  }, [finished, pendingApproval, events.length, initialStatus]);

  return {
    events,
    connection,
    lastSeq: lastSeqRef.current,
    finished,
    status,
    pendingApproval,
    latestScreenshot,
    reconnect,
    reset,
  };
}

export function isTerminal(status: RunStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}
