/**
 * How to actually use this thing.
 *
 * Written because the product is not self-evident: recording a workflow is
 * discoverable, and getting data *out* of a website is not. The content is
 * organised by what someone is trying to do rather than by what the software
 * is made of, and it names what still has no screen — telling somebody to
 * click a button that does not exist is worse than telling them there is none.
 *
 * `Help.tsx` builds its table of contents from `USER_GUIDE_TOC` below; every
 * id referenced there must exist as a heading id here, or a sidebar link
 * scrolls to nothing.
 */

export const USER_GUIDE_TOC = [
  {
    label: 'Getting started',
    items: [
      { id: 'two-ways', title: 'Two ways to record, one result' },
      { id: 'filling-forms', title: 'Filling a website in from a spreadsheet' },
    ],
  },
  {
    label: 'Working with data',
    items: [
      { id: 'getting-data-out', title: 'Getting data out, into a spreadsheet' },
      { id: 'spreadsheet-source', title: 'Where the spreadsheet comes from' },
      { id: 'finding-records', title: 'Finding what to run against in the first place' },
    ],
  },
  {
    label: 'Running it',
    items: [
      { id: 'autonomy-levels', title: 'How much a model is allowed to do' },
      { id: 'environments', title: 'Running the same workflow in dev, UAT and production' },
      { id: 'ambiguous-locators', title: 'Why a draft warns "matched N elements"' },
    ],
  },
  {
    label: 'When things change',
    items: [
      { id: 'self-healing', title: 'When a step stops working' },
      { id: 'pacing', title: 'Being polite to the site' },
    ],
  },
];

export function HelpUserGuide() {
  return (
    <>
      <p>
        You do a task in a browser once. TRACE records it, and then repeats it —
        once per row of a spreadsheet, with no model involved and no cost per run.
      </p>

      <h2 id="getting-started">Getting started</h2>

      <h3 id="two-ways">Two ways to record, one result</h3>
      <p>
        <strong>Do it myself</strong> opens a browser and records what you do. Free,
        and the right choice when you know the steps.
      </p>
      <p>
        <strong>Describe it</strong> gives the task to an agent, which works it out in
        a browser you can watch. It costs tokens once. It marks what varies per row and
        what to read out as it goes, and before you see anything it replays what it
        recorded to check that it works &mdash; you are told either way.
      </p>
      <p>
        Both land in the same review screen and produce the same use case, which
        replays for nothing afterwards. Neither publishes anything: a person always
        reviews before a thousand rows run.
      </p>

      <h3 id="filling-forms">Filling a website in from a spreadsheet</h3>
      <ol>
        <li>
          <strong>Record.</strong> Give it a starting address. A browser window opens;
          do the task by hand, then close the window. Nothing is recorded after that.
        </li>
        <li>
          <strong>Name what you typed.</strong> Every value you entered is listed. Name
          the ones that change per row — those become spreadsheet columns. Tick the
          sign-in as a <em>credential</em>: those are entered once per run, stored
          encrypted, and never written into the workflow.
        </li>
        <li>
          <strong>Publish it.</strong> A recording is a draft until someone reviews it.
          Publishing re-checks it and is required in each environment separately.
        </li>
        <li>
          <strong>Run a file.</strong> Upload your spreadsheet, confirm which column
          feeds which field, and start. Rows run in sequence on one browser session, so
          it signs in once rather than once per row.
        </li>
      </ol>

      <h2 id="working-with-data">Working with data</h2>

      <h3 id="getting-data-out">Getting data out, into a spreadsheet</h3>
      <p>
        While recording, use the recorder&rsquo;s own toolbar:
        <strong> Assert text</strong> for something shown on the page, or{' '}
        <strong>Assert value</strong> for something typed into a field. Click the
        button, then click the thing you want. Do that for each value.
      </p>
      <p>
        When you close the window you are asked <em>What should it read?</em> — name
        each one, and it becomes a column in the results file. Leave a name blank and
        it stays a check that the page still says what it said.
      </p>
      <p>
        Values are read on the page you pointed at them on, in the order you did it, so
        reading something on one screen and then moving to the next works as you would
        expect.
      </p>
      <div className="help-warning">
        <strong>Still needing the API:</strong> reading a whole table in one step
        (<code>extract_rows</code>), keeping a file (<code>download</code>), and reading
        an attribute such as a link&rsquo;s address. Those exist in the engine and have
        no screen yet.
      </div>

      <h3 id="spreadsheet-source">Where the spreadsheet comes from</h3>
      <p>
        Results are produced <strong>per batch</strong>, not per single run. Use
        &ldquo;Run a file&rdquo; even for one row, then use <strong>Results</strong> on
        that batch — under &ldquo;Earlier runs&rdquo; if you have closed the tab since.
        You get a CSV with one row out for every row in: your inputs, whether it worked,
        and every value it read. It opens directly in Excel.
      </p>
      <p>
        Files a <code>download</code> step kept are listed separately on the run, and
        live wherever this deployment puts artifacts.
      </p>

      <h3 id="finding-records">Finding what to run against in the first place</h3>
      <p>
        If you do not already have the list of records — account numbers, references —
        an <code>extract_rows</code> step can read the site&rsquo;s own list page into
        rows. When a run produces rows, the screen offers{' '}
        <strong>Save as a dataset</strong>, and a second use case runs once per row to
        pull the detail. One pass to find what exists, one to fetch it.
      </p>

      <h2 id="running-it">Running it</h2>

      <h3 id="autonomy-levels">How much a model is allowed to do</h3>
      <p>
        Each use case chooses, under <em>How it runs</em>, and the choice travels
        with it when it is promoted.
      </p>
      <ul>
        <li>
          <strong>Strict</strong> follows the recorded steps. No model can run &mdash;
          the replay engine cannot reach one &mdash; so this costs nothing, ever.
        </li>
        <li>
          <strong>Guided</strong> is the same until a step stops matching. Then one
          budgeted call re-finds the control and the run carries on. A row where
          nothing breaks costs nothing, which on a site that has not changed is all
          of them.
        </li>
        <li>
          <strong>Explore</strong> has no plan at all: it works each row out from the
          page and the task. That costs on <em>every</em> row, so four thousand rows
          is four thousand times. It is for work that genuinely cannot be recorded.
        </li>
      </ul>
      <p>
        Before a batch starts you are shown what it will cost. If a workspace has a
        monthly limit (under <strong>Targets</strong>), a run stops when it is
        reached rather than going past it.
      </p>

      <h3 id="environments">Running the same workflow in dev, UAT and production</h3>
      <p>
        A use case names a <strong>target</strong>; each deployment says what address
        that target has, under <strong>Targets</strong>. The workflow itself carries no
        address, so the same one runs everywhere unchanged. Pick its target on the use
        case screen under <em>Where it runs</em>.
      </p>

      <h3 id="ambiguous-locators">Why a draft warns &ldquo;matched N elements&rdquo;</h3>
      <p>
        Recording points at one specific thing &mdash; a click always lands on the
        control you meant, whichever way you recorded it. What gets <em>saved</em> is a
        description of that control &mdash; its role and name, such as &ldquo;button
        &lsquo;Chat&rsquo;&rdquo; &mdash; because that is what survives the page changing
        in small ways later. On a screen with several identical controls (a
        &ldquo;Chat&rdquo; button on every row of a project list, say), that description
        can match more than one of them.
      </p>
      <p>
        When the draft says a step matched more than one element when it was recorded,
        replay will refuse to guess which one was meant &mdash; the same way it refuses
        an invented selector. Silently clicking the first match is how a batch acts on
        the wrong row; refusing and telling you is the safer failure. Re-record that
        step pointing at something that names the right one &mdash; the row or card it
        sits inside, not just the button &mdash; rather than publishing past the warning.
      </p>

      <h2 id="when-things-change">When things change</h2>

      <h3 id="self-healing">When a step stops working</h3>
      <p>
        Sites get redesigned and a recorded control moves. The run stops at that step
        and shows the page as it was, with <strong>Fix it with AI</strong>: it reads
        that page, proposes a new locator, and saves it as a new draft version for you
        to approve. What it learns is remembered, so the same change on the same site
        costs one model call rather than one per run.
      </p>

      <h3 id="pacing">Being polite to the site</h3>
      <p>
        <em>Pace</em> on the use case screen sets the wait between rows. A long
        extraction that reads as an attack gets the account blocked — and automating a
        site you do not own can breach its terms even when the data is yours. Worth
        checking before a few thousand rows.
      </p>
    </>
  );
}
