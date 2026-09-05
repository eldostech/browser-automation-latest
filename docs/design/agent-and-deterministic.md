# Both capabilities: an agent that explores, an engine that repeats

## The ask

Users want to drive the browser with an LLM agent — for authoring a workflow and
for running one — while keeping the deterministic record-and-replay this
platform was rebuilt around. Not one or the other. Both, chosen per workflow and
per run, because some work is worth a model's time and money and some is not.

This document is the design. No code has been written for it.

---

## 1. The shape of the answer

The instinct is to treat this as "agent mode vs recorded mode" and build a
second product beside the first. That is the wrong cut, and it would double the
surface area of everything — runs, batches, audit, healing, promotion.

There are **two independent axes**, and every useful combination is a pairing of
one from each.

|  | **Authored by a person** | **Authored by an agent** |
|---|---|---|
| **Run deterministically** | Today. Record with codegen, replay free, forever. | **The valuable new one.** Agent explores once; its trajectory becomes a `UseCase`; 4,000 rows replay at zero token cost. |
| **Run by an agent** | Rare but real: a recorded flow whose site changes constantly, run adaptively. | One-off or highly dynamic work. Expensive per row, and honest about it. |

Authoring and execution are separate choices. That is the whole design in one
sentence, and everything below follows from it.

### The three execution modes

Execution is a spectrum of how much a model is allowed to do, not a switch:

| Mode | Model runs | Cost per row | Use it when |
|---|---|---|---|
| **Strict** | Never | Zero | The site is stable. The default, and what a migration should end on. |
| **Assisted** | Only when a step fails | Zero on a good row | The site drifts. **This already exists** — it is `REPLAY_HEALING_ENABLED` plus healing memory. |
| **Agent** | Every row | Full, every row | The page differs per record, or the task cannot be expressed as fixed steps. |

Assisted is the mode most people actually want and it is already built. Part of
this work is making it *visible* — today it is an environment variable nobody
sees.

### The economic argument, made in the product

The reason to build this is cost and time, so the product has to make cost a
first-class thing rather than a surprise on a bill:

- An agent authoring session is a **one-time** cost per workflow.
- An agent *run* is a cost **per row**, and 4,000 rows is 4,000 times it.
- The distillation step — agent trajectory becomes a `UseCase` — is the lever
  that turns the second into the first.

So the product should push, at every opportunity: **explore with the agent,
then promote to Strict.** Not by refusing agent runs, but by showing what a
batch will cost before it starts and offering the distilled alternative.

---

## 2. What already exists

Most of this is not new work. The redesign kept everything an agent needs
except the agent.

**Reusable unchanged:** `UseCase` schema, targets and promotion, credentials,
datasets, the job queue and batches, `run_steps` and the visual trail, events
and the live stream, artifacts and downloads, healing memory, the audit log,
RBAC (`RUN_APPROVE` is still defined and unused), `llm.py` and `chat.py`.

**One vestigial column becomes meaningful again.** `runs.task` holds a
natural-language instruction and was left behind when the agent went; it is
written today as a synthetic label (`"Replay: <name>"`). For an agent run it is
the actual instruction, which is what it was for.

**What was deleted, and would not come back as it was:** `agent.py` (858
lines), `distill.py` (1,274), `mcp_client.py` (467), `graph.py` (112),
`checkpoints.py` (117), `stash.py` (114) — 2,942 lines. The new agent is
smaller because it does less: no LangGraph, no MCP hop for the browser, and a
distiller that reads structured tool calls rather than inferring steps from
free-form transcripts.

---

## 3. The invariant this must not break

> **`engine.py` must never import `llm`.** Two tests assert the absence of that
> import in the source.

That is not ceremony. It is the reason a Strict batch cannot silently cost
money, and it is the only mechanical guarantee in the system. An agent mode
implemented as a branch *inside* the engine would destroy it.

So: **the agent executor is a separate module** (`pilot.py`) that drives the
same `PlaywrightSession` and writes the same events, steps and artifacts. The
runner picks one or the other. The engine never learns that an agent exists.

```
                    ┌───────────────────────┐
                    │      runner.py        │  picks a driver per run
                    └───────┬───────────────┘
                 mode=strict│           │mode=agent
                 mode=assisted          │
                    ┌───────▼──────┐  ┌─▼──────────────┐
                    │  engine.py   │  │   pilot.py     │
                    │  no llm, by  │  │  the agent     │
                    │  construction│  │  loop + tools  │
                    └───────┬──────┘  └─┬──────────────┘
                            │           │
                    ┌───────▼───────────▼───────┐
                    │   PlaywrightSession       │  one browser layer
                    └───────────┬───────────────┘
                    ┌───────────▼───────────────┐
                    │ events · run_steps ·      │  one audit trail
                    │ artifacts · executions    │
                    └───────────────────────────┘
```

Both drivers emit the *same* event and step records. A person looking at a run
sees the same trail whichever produced it — that is what keeps the audit story
whole.

---

## 4. The tool surface

The agent needs tools. Three candidate sources, and the recommendation is to
mix them deliberately.

**Browser tools: implemented directly on `PlaywrightSession`, not over MCP.**
This platform removed the MCP hop on purpose — a JSON-RPC round trip to a Node
process that then calls Playwright is a translation layer between us and the
library we want, and it cost auto-waiting and real locators. Re-introducing it
for the agent would re-introduce that. The tools are thin functions over the
session we already have:

`navigate` · `click` · `fill` · `select` · `press` · `extract` ·
`extract_rows` · `download` · `snapshot` · `wait_for` · `go_back` · `finish`

Note these are **the same verbs as `Step.action`**. That is deliberate and it
is what makes distillation nearly free: a trajectory of tool calls is already a
list of steps.

**MCP as an extension point, not the browser path.** A tool registry that can
mount MCP servers gives users the third-party tools they will ask for —
a ticketing system, a file store, an internal API — without putting the browser
behind one. `browser-use` sits here too if a user wants it: as an alternative
*driver* behind a feature flag, not as the default.

**No `script` tool.** The agent must not be able to evaluate arbitrary
JavaScript. `allow_scripts` is a privileged grant to a *reviewed* recording; an
agent that can write JS at run time makes that gate meaningless.

---

## 5. Distillation: the bridge that pays for everything

An agent session produces a **trajectory**: an ordered list of tool calls, each
with its arguments, the locator it resolved to, the page it was on, and whether
it worked. Distillation turns that into a draft `UseCase`.

Because the tool names are the step actions, most of this is mechanical:

| Trajectory | Becomes |
|---|---|
| `navigate(url)` | a `navigate` step, origin bound to `{{env.base_url}}` |
| `fill(locator, value)` | a `fill` step; the value offered as a declarable input |
| `extract_rows(...)` | an `extract_rows` step with its columns |
| `snapshot`, `wait_for`, retries, dead ends | dropped — they are how it *found* the way, not the way |

The genuinely hard part is **the loop**: an agent given "download every
statement" produces 400 near-identical sub-trajectories. Distillation has to
recognise the repeated shape and emit *one* row-step sequence plus the varying
value as an input. That is the piece worth designing carefully and the reason
the old `distill.py` was 1,274 lines.

A first version can refuse to guess: distil a **single-record** agent session
into row steps, and require the person to have run it against one record. That
is honest, much smaller, and matches the two-pass migration flow already built.

---

## 6. UI design

### 6.1 Creating a use case — the fork moves to the front

Today "Record" opens codegen. It becomes a choice:

```
┌─ New use case ────────────────────────────────────────────────┐
│                                                               │
│  ┌───────────────────────┐   ┌───────────────────────────┐    │
│  │  ⏺  Record it myself  │   │  ✨ Describe it            │    │
│  │                       │   │                           │    │
│  │  A browser opens. Do  │   │  Say what to do. An agent │    │
│  │  the task once. Free  │   │  works it out in a live   │    │
│  │  — no model involved. │   │  browser you can watch.   │    │
│  │                       │   │  Costs tokens once.       │    │
│  │  Best when you know   │   │  Best when the site is    │    │
│  │  the steps.           │   │  unfamiliar or fiddly.    │    │
│  └───────────────────────┘   └───────────────────────────┘    │
│                                                               │
│  Either way you get a use case you can review, edit and       │
│  replay for free.                                             │
└───────────────────────────────────────────────────────────────┘
```

The last line is the important one: **both paths converge on the same artifact.**
Nobody should think they are choosing a product.

### 6.2 Agent authoring session

```
┌─ Describe it ─────────────────────────────────────────────────┐
│ Target   [ schemora-local  ▾ ]   Sign in as [ vendor login ▾ ] │
│                                                               │
│ What should it do?                                            │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ Open the accounts list, find account A-1001, and download  │ │
│ │ its latest statement.                                      │ │
│ └───────────────────────────────────────────────────────────┘ │
│                                                               │
│ Budget  [ 40 ] steps   [ 60k ] tokens   [ 5 ] minutes         │
│         Stops and asks when it reaches any of these.          │
│                                                               │
│ ☑ Ask me before anything irreversible (submit, delete, pay)   │
│ ☐ Let it write as well as read                                │
│                                    [ Cancel ]  [ Start ]      │
└───────────────────────────────────────────────────────────────┘
```

While it runs — two panes, because watching is how trust is built:

```
┌─ Working ──────────────────── 12 steps · 18.4k tokens · $0.09 ─┐
│ ┌──────────────────────────┐ ┌───────────────────────────────┐ │
│ │  live browser            │ │ ▸ navigate  /accounts         │ │
│ │  (the same view the      │ │ ▸ snapshot                    │ │
│ │   "watch it run" option  │ │ ▸ click   link "A-1001"       │ │
│ │   already provides)      │ │ ▸ extract balance → "1,240.55"│ │
│ │                          │ │ ▸ download  statement.csv     │ │
│ │                          │ │ ⏸ Asks: submit the form?      │ │
│ │                          │ │      [ Allow ]  [ Refuse ]    │ │
│ └──────────────────────────┘ └───────────────────────────────┘ │
│                        [ Stop ]  [ Save as a use case → ]      │
└────────────────────────────────────────────────────────────────┘
```

**"Save as a use case"** is the distillation, and it lands in the existing
review screen — the same one a codegen recording lands in, with the same
"name what you typed" step. One review flow, two producers.

### 6.3 The use case screen: how it runs

A new card beside **Where it runs** and **Pace**:

```
┌─ How it runs ─────────────────────────────────────────────────┐
│  ● Strict          Follows the recorded steps. No model, no    │
│                    token cost, identical every time.           │
│  ○ Assisted        As Strict, but asks a model to re-find a    │
│                    control when a step stops matching.         │
│                    ~$0.02 per repair · nothing on a good row.  │
│  ○ Agent           Works out each row from the task text.      │
│                    ~$0.08 per row · 4,000 rows ≈ $320.         │
│                                                                │
│  ⓘ This use case has run 1,240 rows in Strict without a        │
│    failure. Agent mode would have cost about $99.              │
└────────────────────────────────────────────────────────────────┘
```

That last line is the product arguing for itself with the user's own data. It
is also trivially derivable from `executions`.

Agent mode reveals a task field (seeded from the authoring prompt) and the
budget controls, because in Agent mode the *instruction* is the definition —
the steps become a hint rather than a script.

### 6.4 Starting a batch: cost before commitment

```
┌─ Run a file ──────────────────────────────────────────────────┐
│  vendor-accounts.csv · 4,000 rows                              │
│                                                                │
│  Mode  ● Strict            estimated cost   $0.00              │
│        ○ Assisted          estimated cost   $0 – $12           │
│        ○ Agent             estimated cost   ~$320   ⚠          │
│                                                                │
│  ⚠ Agent mode on 4,000 rows. Cap this batch at [ $50 ] and     │
│    stop when it is reached.                                    │
│                                                                │
│  Consider: run 20 rows in Agent, save what it learns as a use  │
│  case, then run the rest in Strict.        [ Try that ]        │
└────────────────────────────────────────────────────────────────┘
```

A hard cap is not optional. An agent loop over a spreadsheet is the single
easiest way to spend a lot of money in this product.

### 6.5 Watching an agent run

The existing run view already has a step trail, screenshots and a timeline. An
agent run adds a **reasoning column** beside the steps (what it was trying,
which tool, why), and a running cost. The failure and repair panels work
unchanged.

### 6.6 Where things sit

| Screen | Change |
|---|---|
| Use cases (list) | A mode chip per row: `Strict` / `Assisted` / `Agent` |
| New use case | The two-card fork above |
| Agent authoring | New screen — the two-pane session |
| Use case detail | New "How it runs" card; task + budget when Agent |
| Run a file | Mode selector, estimate, hard cap |
| Run view | Reasoning column and live cost for agent runs |
| Runs (history) | Mode and cost columns |
| Admin | Per-workspace monthly token ceiling |

---

## 7. Backend design

### 7.1 New modules

| Module | What it is |
|---|---|
| `pilot.py` | The agent loop: perceive (snapshot) → decide (model) → act (tool) → record. Budgeted, cancellable, emits the same events as the engine. |
| `tools.py` | The tool registry. Browser tools over `PlaywrightSession`; MCP-provided tools mounted alongside. One schema, one dispatcher, one audit point. |
| `trajectory.py` | What an agent session did, as data: ordered tool calls with arguments, resolved locators, page URLs, outcomes. Persisted. |
| `distil.py` | Trajectory → draft `UseCase`. Mechanical for the straight-line case; loop detection is the hard part and is phased. |
| `budget.py` | Steps, tokens, wall clock and money. Checked before every model call and every tool call; the thing that makes agent mode safe to offer. |

### 7.2 Changes to what exists

- `runner.py` — picks driver by mode; passes budgets; writes mode and cost.
- `usecase.py` — `mode: strict | assisted | agent`, `task: str`,
  `budget: {steps, tokens, seconds, usd}`. All optional, defaulting to today's
  behaviour so every existing definition keeps working.
- `routers/` — a recordings-shaped surface for agent sessions
  (`POST /api/agent-sessions`, stream, approve, `POST .../distil`).
- `rbac.py` — `AGENT_RUN` and `AGENT_AUTHOR`; `RUN_APPROVE` finally used.

### 7.3 Data model additions

| Table | Change |
|---|---|
| `usecases` | `mode` column, mirrored from the definition like `target` |
| `executions` | `mode`, `cost_usd`; `llm_calls`/`llm_tokens` already exist |
| `batches` | `mode`, `cost_cap_usd`, `cost_usd` |
| `agent_sessions` | New: an authoring session — task, target, status, budget, cost |
| `trajectory_steps` | New: one row per tool call, the input to distillation |
| `workspaces` | `monthly_token_ceiling` |

`runs.task` stops being vestigial.

### 7.4 Safety

Agent mode is materially more dangerous than replay, and the design has to
answer that rather than assume goodwill:

- **The allowlist still binds.** `check_navigation` is enforced in the tool
  layer, not left to the model. A model asking to leave the allowed domains is
  a refused tool call and an audit entry.
- **Read-only by default.** Write tools (`click` on a submit, `fill`, `select`)
  are gated by an explicit "let it write" choice on the session.
- **Approval for irreversible actions.** `RUN_APPROVE` exists; the rendezvous
  needs rebuilding. Submitting, deleting and paying should stop and ask.
- **Credentials are never in the context window.** The model sees
  `{{secret.password}}`; the tool layer substitutes. This is how replay already
  works and must not regress.
- **Every tool call is audited**, with arguments redacted through the existing
  `Redactor`.
- **Budgets are enforced server-side**, never in the UI alone.

### 7.5 What I would argue against

**Do not put the browser behind MCP again.** It was removed for reasons that
have not changed. MCP belongs at the edge, for third-party tools.

**Do not let Agent mode become the default.** It should be reachable, honest
about cost, and constantly offering the distilled alternative. A platform whose
selling point is deterministic replay should not quietly become a per-row LLM
bill.

**Do not skip distillation to ship agent mode sooner.** Without it this is a
browser-using chatbot with an audit log, and the compounding value — explore
once, run free forever — never arrives.

---

## 8. Phasing and size

Sizes are relative (S ≈ a day or two, M ≈ a week, L ≈ two or three, XL ≈ more),
assuming one person who knows this codebase.

### A. Make the mode visible — **S**
Surface what exists. `mode` on the use case (strict/assisted), the "How it runs"
card, mode chips in the list, healing turned on per use case rather than per
deployment. **Ships value with no agent at all** and is the right first step:
Assisted is already built and currently invisible.

### B. The tool layer — **M**
`tools.py` over `PlaywrightSession`, schema per tool, allowlist and redaction
enforced in the dispatcher, audit per call. No agent yet. Testable on its own.

### C. The agent loop — **L**
`pilot.py`, `budget.py`, `trajectory.py`. Perceive/decide/act, budgets, cancel,
same events as the engine. The approval rendezvous comes back here.

### D. Agent authoring UI — **M**
The fork screen, the two-pane session, the live transcript over the existing
event stream, save-as-use-case wired to (E).

### E. Distillation, straight-line — **M**
Trajectory → draft `UseCase` for a single-record session, landing in the
existing review screen. Deliberately refuses to guess at loops.

### F. Agent execution mode — **M**
Run a use case per row through `pilot.py`. Cost estimates, per-batch caps,
mode and cost on runs and batches.

### G. Distillation with loop detection — **L**
Recognise the repeated shape in a multi-record session and emit one row-step
sequence plus the varying input. The hard, valuable one.

### H. MCP tool providers — **M**
Mount MCP servers as extra tools, per workspace, with their own allowlist.

### I. Spend governance — **S/M**
Workspace ceilings, alerting, a cost report per use case.

**Rough total: 10–14 weeks** for one experienced person, with A–F (~7–9 weeks)
being a coherent, shippable product on its own. G is the piece that makes agent
authoring pay for itself at scale; H and I can follow demand.

### The order I would actually ship in

1. **A** alone, first, and let people use Assisted. It is free value and it will
   tell you how much of the demand for "agent" is really demand for "cope with
   a site that changes".
2. **B → C → D → E** as one arc, ending at: describe a task, watch it work,
   save it as a use case, replay it free.
3. **F** only after E, so agent execution is offered next to a cheaper
   alternative rather than as the only way to use the agent.

---

## 9. The risk worth naming

The reason this codebase deleted its agent was that a per-row model call made
every run cost money, made no two runs identical, and made a thousand-row batch
a thousand chances to improvise. None of that has become untrue.

What makes it safe to bring back is that it is no longer the only way to run
something: the agent's job is to *find* the way, and the engine's job is to
repeat it. Kept in that order — with distillation as the bridge and cost
visible at every step — both capabilities make each other more valuable.

Inverted, with agent execution as the default and distillation an afterthought,
this becomes the product the redesign was written to replace.
