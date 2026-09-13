/**
 * How this is built, for the people who have to build the next part of it.
 *
 * Mirrors the shape of `README.md` and `docs/design/*.md` rather than
 * restating them line for line — this is the page a technical reader opens
 * first, not the last word; it says where to go for the rest. The one thing
 * worth defending in detail here is the invariant the rest of the backend is
 * built around: replay never spends a token, and the code enforces that by
 * construction rather than by convention.
 *
 * `Help.tsx` builds its table of contents from `TECHNICAL_TOC` below; every
 * id referenced there must exist as a heading id here.
 */

export const TECHNICAL_TOC = [
  {
    label: 'Architecture',
    items: [
      { id: 'system-overview', title: 'System overview' },
      { id: 'two-lane-design', title: 'The two-lane design' },
    ],
  },
  {
    label: 'How it stays reliable',
    items: [
      { id: 'deterministic-replay', title: 'Replay with no LLM, ever' },
      { id: 'guardrails', title: 'Guardrails and the safety invariants' },
    ],
  },
  {
    label: 'The AI agent',
    items: [
      { id: 'agent-path', title: 'How the agent authors and repairs a use case' },
      { id: 'autonomy-technical', title: 'Autonomy levels, mapped to code' },
    ],
  },
  {
    label: 'Data and operations',
    items: [
      { id: 'data-model', title: 'Data model and tenancy' },
      { id: 'event-streaming', title: 'The job queue and the event bus' },
    ],
  },
  {
    label: 'Reference',
    items: [{ id: 'further-reading', title: 'Where to go deeper' }],
  },
];

type DiagramBox = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  subtitle: string;
  tone: 'neutral' | 'accent' | 'warn' | 'ok';
  dashed?: boolean;
};

type DiagramEdge = {
  from: [number, number];
  to: [number, number];
  label: string;
  tone: 'accent' | 'warn' | 'ok';
  dashed?: boolean;
  /** Where along the line the label sits (0 = from, 1 = to). Default 0.5 —
   * override when the midpoint lands inside a box or on top of another
   * edge's label. */
  labelT?: number;
  /** Vertical offset for the label, in SVG units. Default -6. */
  labelDy?: number;
};

const TONE_VAR: Record<DiagramBox['tone'], string> = {
  neutral: 'var(--chrome)',
  accent: 'var(--accent-ink)',
  warn: 'var(--warn)',
  ok: 'var(--ok)',
};
const TONE_BG: Record<DiagramBox['tone'], string> = {
  neutral: 'var(--bg-raised)',
  accent: 'var(--accent-dim)',
  warn: 'var(--warn-bg)',
  ok: 'var(--ok-bg)',
};

const BOXES: DiagramBox[] = [
  { id: 'you', x: 16, y: 24, w: 150, h: 56, title: 'You', subtitle: 'operator, in a browser', tone: 'neutral' },
  { id: 'webapp', x: 206, y: 24, w: 170, h: 56, title: 'Web App', subtitle: 'React + Vite + TS', tone: 'neutral' },
  {
    id: 'api',
    x: 416,
    y: 8,
    w: 486,
    h: 152,
    title: '',
    subtitle: '',
    tone: 'neutral',
    dashed: true,
  },
  {
    id: 'engine',
    x: 440,
    y: 48,
    w: 200,
    h: 92,
    title: 'Deterministic Engine',
    subtitle: 'engine.py — imports no llm',
    tone: 'accent',
  },
  {
    id: 'agent',
    x: 672,
    y: 48,
    w: 200,
    h: 92,
    title: 'AI Agent',
    subtitle: 'create_agent + LangGraph, via MCP',
    tone: 'warn',
  },
  { id: 'pg', x: 206, y: 282, w: 246, h: 74, title: 'PostgreSQL + pgvector', subtitle: 'workspace-scoped by construction', tone: 'ok' },
  { id: 'target', x: 472, y: 282, w: 196, h: 74, title: 'Target Website', subtitle: 'the site being automated', tone: 'neutral' },
  { id: 'bedrock', x: 688, y: 282, w: 214, h: 74, title: 'AWS Bedrock', subtitle: 'Claude, via the Converse API', tone: 'warn' },
];

const EDGES: DiagramEdge[] = [
  { from: [166, 52], to: [206, 52], label: '', tone: 'accent' },
  { from: [376, 52], to: [416, 52], label: 'REST + WSS', tone: 'accent', labelDy: -20 },
  { from: [540, 140], to: [560, 282], label: 'CDP, direct', tone: 'accent' },
  { from: [772, 140], to: [640, 282], label: 'MCP → CDP', tone: 'warn' },
  { from: [455, 140], to: [340, 282], label: 'SQL', tone: 'ok' },
  { from: [692, 140], to: [420, 282], label: 'SQL', tone: 'ok', dashed: true, labelT: 0.78 },
  { from: [860, 140], to: [790, 282], label: 'HTTPS + SigV4', tone: 'warn' },
];

function ArchitectureDiagram() {
  return (
    <figure className="help-diagram-figure">
      <svg viewBox="0 0 920 372" role="img" aria-label="You go through the Web App to the API, which has two lanes: a deterministic engine that talks to the target website over CDP and to PostgreSQL, and an AI agent that reaches the target website and AWS Bedrock through MCP, and also writes to PostgreSQL.">
        <defs>
          <marker id="help-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 Z" fill="context-stroke" />
          </marker>
        </defs>

        {EDGES.map((e, i) => (
          <path
            key={i}
            d={`M${e.from[0]},${e.from[1]} L${e.to[0]},${e.to[1]}`}
            fill="none"
            style={{ stroke: TONE_VAR[e.tone] }}
            strokeWidth={1.6}
            strokeDasharray={e.dashed ? '5 4' : undefined}
            markerEnd="url(#help-arrow)"
          />
        ))}
        {EDGES.map((e, i) => {
          if (!e.label) return null;
          const t = e.labelT ?? 0.5;
          const x = e.from[0] + (e.to[0] - e.from[0]) * t;
          const y = e.from[1] + (e.to[1] - e.from[1]) * t + (e.labelDy ?? -6);
          return (
            <text
              key={i}
              x={x}
              y={y}
              textAnchor="middle"
              style={{ fill: TONE_VAR[e.tone], fontFamily: 'var(--mono)' }}
              fontSize={11}
            >
              {e.label}
            </text>
          );
        })}

        {BOXES.map((b) => (
          <g key={b.id}>
            <rect
              x={b.x}
              y={b.y}
              width={b.w}
              height={b.h}
              rx={8}
              style={{
                fill: b.dashed ? 'none' : TONE_BG[b.tone],
                stroke: TONE_VAR[b.tone],
              }}
              strokeWidth={b.dashed ? 1.4 : 1.3}
              strokeDasharray={b.dashed ? '6 5' : undefined}
            />
            {b.title && (
              <text
                x={b.x + b.w / 2}
                y={b.y + b.h / 2 - (b.subtitle ? 6 : -4)}
                textAnchor="middle"
                style={{ fill: 'var(--text)', fontFamily: 'var(--sans)' }}
                fontSize={13.5}
                fontWeight={700}
              >
                {b.title}
              </text>
            )}
            {b.subtitle && (
              <text
                x={b.x + b.w / 2}
                y={b.y + b.h / 2 + 12}
                textAnchor="middle"
                style={{ fill: 'var(--text-dim)', fontFamily: 'var(--mono)' }}
                fontSize={10.5}
              >
                {b.subtitle}
              </text>
            )}
          </g>
        ))}

        <text x={460} y={22} style={{ fill: 'var(--text-faint)', fontFamily: 'var(--mono)' }} fontSize={11}>
          API — FastAPI :8000, one process, two lanes
        </text>
      </svg>
      <figcaption>
        One API, two ways of driving a browser. The deterministic lane never imports{' '}
        <code>llm</code>; the agent lane never touches a browser except through MCP.
      </figcaption>
    </figure>
  );
}

export function HelpTechnical() {
  return (
    <>
      <p>
        FastAPI + SQLAlchemy 2.0 async + PostgreSQL/pgvector on the backend; React 18 +
        Vite + TypeScript on the frontend. This section is a map, not the territory —
        it points at the modules and invariants that matter, and at the design docs
        that go deeper than a help screen should.
      </p>

      <h2 id="system-overview">System overview</h2>
      <p>
        The web app talks to one FastAPI process over REST for requests and a
        WebSocket for live run events. Inside that process, <code>main.py</code> wires
        application state in <code>lifespan</code> and does nothing else;
        <code> routers/</code> (one module per resource) call into{' '}
        <code>services.py</code>, which holds the logic shared by the single-row and
        batch paths, which in turn calls <code>store.py</code>. <code>deps.py</code>{' '}
        supplies everything a route handler needs, so a test swaps in a fake database,
        a scripted model, or a fixed principal by overriding one dependency rather than
        patching modules.
      </p>

      <h2 id="two-lane-design">The two-lane design</h2>
      <p>
        The central fact about this codebase is that there are two independent ways to
        drive a browser, and the API is the only thing that sits above both of them.
      </p>
      <ArchitectureDiagram />
      <p>
        <strong>Deterministic replay</strong> (<code>recorder.py</code> →{' '}
        <code>codegen.py</code> → <code>engine.py</code>) drives Chromium directly over
        CDP. <code>engine.py</code> has no parameter that could accept a model client —
        it is structurally incapable of calling one, and{' '}
        <code>tests/test_e2e_engine.py</code> asserts that the string{' '}
        <code>import llm</code> does not appear anywhere in its source. A step is a
        recorded action plus a ranked ladder of locators (accessible role and name
        first, the recorded selector as a fallback), never a live decision.
      </p>
      <p>
        <strong>The AI agent</strong> (<code>agent/graph.py</code>, built on{' '}
        <code>langchain.agents.create_agent</code>) never touches a browser directly
        either — every action goes through the Model Context Protocol, either to a
        local <code>npx @playwright/mcp</code> subprocess or to a workspace-registered
        MCP server. That is a second structural guarantee, not a convention: there is
        no code path from the agent to a page that does not pass through{' '}
        <code>AgentToolSession</code>&rsquo;s guard.
      </p>

      <h2 id="deterministic-replay">Replay with no LLM, ever</h2>
      <p>
        Codegen output is data, never code: <code>codegen.py</code> parses it with{' '}
        <code>ast</code>, and never <code>exec</code>s or imports it. A line it does
        not recognise becomes an <code>Unsupported</code> entry shown to the user —
        never a guessed-at step. <code>{'{{input.x}}'}</code> and{' '}
        <code>{'{{secret.x}}'}</code> templating is confined to value-bearing fields;
        the validator refuses it inside a locator, because a locator that can vary at
        runtime is a locator the model would effectively be writing.
      </p>
      <p>
        A batch runs every row of a spreadsheet against one browser session on a
        durable Postgres-backed queue (<code>jobs.py</code>, <code>SELECT ... FOR
        UPDATE SKIP LOCKED</code>, leases rather than flags), so a restart never loses
        or double-runs a row.
      </p>

      <h2 id="guardrails">Guardrails and the safety invariants</h2>
      <ul>
        <li>
          <strong>The model can never invent a locator.</strong> Healing and the agent
          are shown a numbered list of controls actually present on the page and hand
          back an index — never free-text they composed themselves.
        </li>
        <li>
          <strong>Tenancy is enforced by construction.</strong> Scoped operations live
          on <code>WorkspaceStore</code> (<code>store.workspace(id)</code>), never on{' '}
          <code>Store</code>, so a query that forgets the tenant filter fails to
          compile rather than leaking across workspaces.
        </li>
        <li>
          <strong>Authorization is a dependency, not an <code>if</code>.</strong> Every
          route declares <code>require(Permission.X)</code>, so the check is visible in
          the route definition and in the generated OpenAPI, not buried in a handler.
        </li>
        <li>
          <strong>Secrets are registered with the redactor before any event is
          emitted</strong>, and a batch takes a stored credential id, never an inline
          value — a credential cannot leak into a log because it is never in a form
          that could be logged.
        </li>
      </ul>

      <h2 id="agent-path">How the agent authors and repairs a use case</h2>
      <p>
        The agent path exists for two moments, and only two: <em>authoring</em> a use
        case from a plain-English task, and <em>healing</em> a step that broke after a
        site changed. It is built as an explicit tool/guardrail/provider layout under{' '}
        <code>agent/</code> — one file per tool in <code>agent/tools/</code>, the
        classification tables and the guard in <code>agent/guardrails/</code>, and the
        MCP/browser wiring in <code>agent/providers/</code> — so adding a capability
        means adding a file, not learning where in a monolith to splice it in.
      </p>
      <p>
        <code>agent/graph.py</code> wires <code>create_agent</code> with three
        middlewares: <code>BudgetMiddleware</code> (stops the run cleanly when a step,
        token, time, or dollar budget is exhausted), <code>FinishMiddleware</code>{' '}
        (refuses to end the session while a row is left unfinished), and{' '}
        <code>HumanInTheLoopMiddleware</code> (suspends the graph for an irreversible
        action and waits — checkpointed in Postgres, so the wait survives a restart).
        Once a session ends, <code>distil.py</code> turns its trajectory into the same
        kind of use case a human recording would have produced, and{' '}
        <code>verify.py</code> replays it once, cold, before anyone sees a draft.
      </p>

      <p>
        Two single calls bracket that session, in <code>agent/brief.py</code>. Before
        it: the request restated as a goal, the values expected to vary per row, and
        what proves a row worked — and deliberately no clicks, because nothing has seen
        the site yet and a plan made of invented buttons sends the recorder hunting for
        a control that does not exist. After it: the distilled steps described in plain
        language onto <code>UseCase.instructions</code>, with a purpose per step onto{' '}
        <code>Step.intent</code>. That second field is what healing and repair read: a
        step&apos;s <code>description</code> renders its own locator, so the question
        used to be &ldquo;which of these controls resembles a link named
        Billing&rdquo; rather than &ldquo;which of these opens the customer&apos;s
        billing tab&rdquo;. Neither pass can add, remove or alter a step, and{' '}
        <code>POST /usecases/:id/describe</code> runs the second one on demand for a
        codegen recording, which has no model in it and so no account of itself.
      </p>

      <h2 id="borrowed">Three checks borrowed, and one way out</h2>
      <p>
        A locator that says only <em>where</em> to look can match one element and
        still be the wrong one, which is worse than failing because it records as a
        success. Each step therefore carries <code>Step.expect_text</code>, the words
        its element had when recorded, and <code>engine.py</code> compares before
        acting — but only when the winning rung does not itself match on text
        (<code>Locator.matches_on_text</code>), since a role-and-name rung has already
        proved the wording. <code>Step.when</code> is the other half of a gap:{' '}
        <code>optional</code> says a failure is survivable, which never said &ldquo;this
        step is not always needed&rdquo;. It is an <code>Assertion</code>, evaluated once
        with no retry, so an absent cookie banner costs nothing per row. And{' '}
        <code>attribute_contains</code> lets a check read an href rather than the words
        on screen.
      </p>
      <p>
        <code>export.py</code> renders a use case back into a Playwright Python script —
        the inverse of <code>codegen.py</code>, and the same rule in both directions:
        generated Python is data, and nothing here runs it. The export carries the
        leading rung of each step with the rest as comments, because a script that fell
        through a ladder would be the engine reimplemented in generated code.{' '}
        <code>GET /usecases/:id/export/python</code> writes no version and changes
        nothing.
      </p>

      <h2 id="autonomy-technical">Autonomy levels, mapped to code</h2>
      <p>
        The three levels a use case can choose are not a UI-only idea — each one names
        a different set of modules that are allowed to run:
      </p>
      <ul>
        <li>
          <strong>Strict</strong> — <code>engine.py</code> only. There is no code path
          from here to <code>llm.py</code>.
        </li>
        <li>
          <strong>Guided</strong> — <code>engine.py</code>, and when a locator ladder
          fails to resolve, exactly one budgeted call into <code>healing.py</code>,
          which recalls similar past fixes from <code>memory.py</code> (pgvector) before
          asking Bedrock anything.
        </li>
        <li>
          <strong>Explore</strong> — the full agent graph, on every row: no recorded
          steps exist yet to fall back to.
        </li>
      </ul>

      <h2 id="data-model">Data model and tenancy</h2>
      <p>
        Four groups of tables, in <code>docs/design/data-model.md</code> in full:{' '}
        <strong>tenancy and identity</strong> (<code>workspaces</code>,{' '}
        <code>users</code>, <code>user_sessions</code>, <code>audit_log</code>);{' '}
        <strong>authoring</strong> (<code>usecases</code>,{' '}
        <code>usecase_versions</code>, <code>targets</code>, <code>credentials</code>);{' '}
        <strong>execution</strong> (<code>jobs</code>, <code>batches</code>,{' '}
        <code>executions</code>, <code>runs</code>, <code>events</code>,{' '}
        <code>run_steps</code>, <code>artifacts</code>); and{' '}
        <strong>learning</strong> (<code>healing_memory</code>, the pgvector table
        Guided mode recalls from). A <code>usecase_version</code> is immutable once
        published — repairing a step creates a new version rather than mutating one a
        batch might already be mid-run against.
      </p>

      <h2 id="event-streaming">The job queue and the event bus</h2>
      <p>
        <code>bus.py</code> fans events out across processes with Postgres{' '}
        <code>LISTEN</code>/<code>NOTIFY</code>, carrying only{' '}
        <code>run_id:seq:origin</code> — never the payload, which the WebSocket handler
        reloads from the database. Every event has a sequence number, and that number
        is the client&rsquo;s resume token: a dropped connection reconnects and asks for
        everything after the last <code>seq</code> it saw, rather than replaying a run
        from the start or silently missing a gap.
      </p>

      <h2 id="further-reading">Where to go deeper</h2>
      <p>This page is a map. For the territory:</p>
      <ul>
        <li>
          <strong>README.md</strong> — setup, the full workflow end to end, the API,
          deployment, troubleshooting.
        </li>
        <li>
          <strong>docs/design/data-model.md</strong> — every table, and the rules that
          apply everywhere.
        </li>
        <li>
          <strong>docs/design/agent-and-deterministic.md</strong> — the authoring and
          operate graphs, the tool surface, and why the agent and the engine can share
          a codebase without sharing a failure mode.
        </li>
        <li>
          <strong>docs/design/repeatable-usecases.md</strong> — why distillation exists
          and what it refused to do along the way.
        </li>
        <li>
          <strong>docs/design/architecture.drawio</strong> — this same system, as an
          importable diagram with a leadership view and a fuller technical view.
        </li>
      </ul>
    </>
  );
}
