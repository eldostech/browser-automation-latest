/**
 * The fork: two ways to record, converging on one artifact.
 *
 * The line under the cards is the important part. Nobody should think they are
 * choosing a product -- both paths land in the same review screen and produce
 * the same document, which replays for nothing either way. What differs is who
 * does the work and what it costs the first time.
 *
 * The agent card is honest when the deployment cannot run it, rather than
 * offering a button that fails: `/api/config` reports the agent the same way it
 * reports the recorder, and for the same reason.
 */

import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { AgentSessionView } from './AgentSession';
import { RecordWorkflow } from './RecordWorkflow';

type Props = {
  onSaved: (usecaseId: string) => void;
};

type Choice = 'fork' | 'record' | 'describe';

export function NewUseCase({ onSaved }: Props) {
  const [choice, setChoice] = useState<Choice>('fork');
  const [agent, setAgent] = useState<{ enabled: boolean; reason: string | null } | null>(
    null,
  );

  useEffect(() => {
    api
      .getConfig()
      .then((body) => setAgent(body.agent ?? { enabled: false, reason: null }))
      .catch(() => setAgent({ enabled: false, reason: null }));
  }, []);

  if (choice === 'record') {
    return <RecordWorkflow onSaved={onSaved} onBack={() => setChoice('fork')} />;
  }
  if (choice === 'describe') {
    return <AgentSessionView onSaved={onSaved} onCancel={() => setChoice('fork')} />;
  }

  return (
    <div className="card">
      <h2>New use case</h2>
      <div className="fork">
        <button type="button" className="fork-card" onClick={() => setChoice('record')}>
          <span className="fork-icon">&#9679; RECORD</span>
          <strong>Do it myself</strong>
          <span className="hint">
            A browser opens. You do the task once by hand and close the window. Point at a
            field with <em>Assert value</em> to make it a column.
          </span>
          <span className="price free">no model &middot; free</span>
        </button>

        <button
          type="button"
          className="fork-card agentic"
          disabled={!agent?.enabled}
          onClick={() => setChoice('describe')}
        >
          <span className="fork-icon">&#10033; DESCRIBE</span>
          <strong>Describe it</strong>
          <span className="hint">
            Say what to do. An agent works it out in a browser you can watch, marking what
            varies per row and what to read out.
          </span>
          <span className="price paid">
            {agent?.enabled ? 'tokens once, then free forever' : 'not available here'}
          </span>
        </button>
      </div>

      {agent && !agent.enabled && agent.reason && (
        <p className="hint" style={{ marginTop: 12 }}>
          {agent.reason}
        </p>
      )}

      <p className="converge">
        Either way you get the same reviewable use case, and either way it replays for
        free afterwards.
      </p>
    </div>
  );
}
