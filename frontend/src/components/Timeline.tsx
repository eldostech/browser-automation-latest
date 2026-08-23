import { useEffect, useRef, useState } from 'react';
import type { AgentEvent } from '../lib/events';
import { formatDuration, formatTime, summariseArgs } from '../lib/format';

interface Props {
  events: AgentEvent[];
  autoScroll?: boolean;
}

/** Chronological view of everything the agent did, grouped by step. */
export function Timeline({ events, autoScroll = true }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [events.length, autoScroll]);

  if (events.length === 0) {
    return <div className="empty-state">Waiting for the first event...</div>;
  }

  const groups = groupByStep(events);

  return (
    <div className="timeline">
      {groups.map((group) => (
        <div className="step" key={group.key}>
          <div className="step-label">
            {group.step === null ? 'run' : group.step === 0 ? 'setup' : `step ${group.step}`}
          </div>
          {group.events.map((event) => (
            <TimelineEntry key={`${event.seq}-${event.type}`} event={event} />
          ))}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

interface StepGroup {
  key: string;
  step: number | null;
  events: AgentEvent[];
}

function stepOf(event: AgentEvent): number | null {
  return 'step' in event && typeof event.step === 'number' ? event.step : null;
}

function groupByStep(events: AgentEvent[]): StepGroup[] {
  const groups: StepGroup[] = [];
  for (const event of events) {
    const step = stepOf(event);
    const last = groups[groups.length - 1];
    if (last && last.step === step) {
      last.events.push(event);
    } else {
      groups.push({ key: `${step}-${event.seq}`, step, events: [event] });
    }
  }
  return groups;
}

function TimelineEntry({ event }: { event: AgentEvent }) {
  switch (event.type) {
    case 'run_started':
      return (
        <div className="entry">
          <div className="entry-head">
            <strong>Run started</strong>
            <span className="time">{formatTime(event.ts)}</span>
          </div>
          <div className="entry-args">
            {event.tools.length} browser tools discovered from the MCP server
            {event.start_url ? ` -- opening ${event.start_url}` : ''}
          </div>
        </div>
      );

    case 'thinking':
      return (
        <div className="entry thinking">
          {event.text}
          {!event.done && <span className="cursor" />}
        </div>
      );

    case 'tool_call':
      return (
        <div className={`entry tool ${event.sensitive ? 'sensitive' : ''}`}>
          <div className="entry-head">
            <span className="name">{event.name}</span>
            {event.sensitive && <span className="tag sensitive">sensitive</span>}
            <span className="time">{formatTime(event.ts)}</span>
          </div>
          {Object.keys(event.arguments ?? {}).length > 0 && (
            <div className="entry-args">{summariseArgs(event.arguments)}</div>
          )}
        </div>
      );

    case 'tool_result':
      return (
        <div className={`entry tool ${event.ok ? '' : 'failed'}`}>
          <div className="entry-head">
            <span className="name">{event.name}</span>
            <span className="tag">{event.ok ? 'ok' : 'error'}</span>
            {event.attempts > 1 && <span className="tag">retry x{event.attempts}</span>}
            <span className="time">{formatDuration(event.duration_ms)}</span>
          </div>
          {event.text && <Collapsible text={event.text} truncated={event.truncated} />}
        </div>
      );

    case 'screenshot':
      return (
        <div className="entry">
          <div className="entry-head">
            <span>Screenshot captured</span>
            <span className="time">{formatTime(event.ts)}</span>
          </div>
          {event.page_url && <div className="entry-args">{event.page_url}</div>}
        </div>
      );

    case 'approval_required':
      return (
        <div className="entry approval">
          <div className="entry-head">
            <strong>Approval requested</strong>
            <span className="name">{event.name}</span>
            <span className="time">{formatTime(event.ts)}</span>
          </div>
          <div className="entry-args">{event.reason}</div>
        </div>
      );

    case 'approval_resolved':
      return (
        <div className="entry approval">
          <div className="entry-head">
            <strong>
              Approval {event.decision === 'timeout' ? 'timed out' : event.decision}
            </strong>
            <span className="time">{formatTime(event.ts)}</span>
          </div>
          {event.note && <div className="entry-args">{event.note}</div>}
        </div>
      );

    case 'error':
      return (
        <div className="entry error">
          <div className="entry-head">
            <strong>{event.recoverable ? 'Recoverable error' : 'Error'}</strong>
            <span className="tag">{event.kind}</span>
            <span className="time">{formatTime(event.ts)}</span>
          </div>
          <div className="entry-args">{event.message}</div>
        </div>
      );

    case 'run_finished':
      return (
        <div className="entry finished">
          <div className="entry-head">
            <strong>Run {event.status}</strong>
            <span className="time">
              {event.steps} steps / {formatDuration(event.duration_ms)}
            </span>
          </div>
          {event.error && <div className="entry-args">{event.error}</div>}
        </div>
      );

    default:
      return null;
  }
}

/** Tool output is long; show a preview and expand on demand. */
function Collapsible({ text, truncated }: { text: string; truncated: boolean }) {
  const [open, setOpen] = useState(false);
  const isLong = text.length > 280;
  const shown = open || !isLong ? text : `${text.slice(0, 280)}...`;

  return (
    <>
      <pre>{shown}</pre>
      {isLong && (
        <button type="button" className="disclosure" onClick={() => setOpen(!open)}>
          {open ? 'show less' : `show all ${text.length} characters`}
        </button>
      )}
      {truncated && (
        <div className="entry-args">
          Output was truncated before being sent to the model to protect the context window.
        </div>
      )}
    </>
  );
}
