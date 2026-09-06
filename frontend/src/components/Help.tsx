/**
 * How to actually use this thing.
 *
 * Written because the product is not self-evident: recording a workflow is
 * discoverable, and getting data *out* of a website is not. The panel is
 * organised by what someone is trying to do rather than by what the software
 * is made of, and it is honest about the one path that has no UI yet — telling
 * somebody to click a button that does not exist is worse than telling them
 * there is no button.
 */

import { useEffect } from 'react';

type Props = {
  onClose: () => void;
};

export function Help({ onClose }: Props) {
  // Escape closes it. A panel that traps you is worse than no panel.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <>
      <div className="help-scrim" onClick={onClose} />
      <aside className="help-panel" role="dialog" aria-label="How to use Understudy">
        <header className="help-head">
          <h2>How this works</h2>
          <button type="button" className="linkish" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="help-body">
          <p>
            You do a task in a browser once. Understudy records it, and then repeats it —
            once per row of a spreadsheet, with no model involved and no cost per run.
          </p>

          <h3>Filling a website in from a spreadsheet</h3>
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

          <h3>Getting data out, into a spreadsheet</h3>
          <div className="help-warning">
            <strong>There is no UI for this yet.</strong> Recording captures what you
            <em> do</em> to a page — clicks, typing, choosing. Reading a value is not
            something you do, so the recorder never sees it. The engine fully supports
            extraction; the screen to set it up has not been built. Until it is, the steps
            are added over the API, below.
          </div>
          <p>
            Three kinds of reading step exist. Add them to a use case&rsquo;s{' '}
            <code>row_steps</code>, and list their <code>output</code> names in the use
            case&rsquo;s <code>outputs</code> — that is what puts them in the results file.
          </p>
          <ul>
            <li>
              <code>extract</code> — one value from one element. Add{' '}
              <code>"attribute": "href"</code> to read a link&rsquo;s address instead of
              its text.
            </li>
            <li>
              <code>extract_rows</code> — a whole table or list in one step. Give it a
              locator matching the rows and a column per field.
            </li>
            <li>
              <code>download</code> — clicks something that yields a file and keeps it,
              under the name the site gave it.
            </li>
          </ul>
          <pre>{`# fetch it
curl -H "Authorization: Bearer $TOKEN" \\
  $API/api/usecases/$ID | jq .definition > uc.json

# add to row_steps, and add the name to "outputs":
{ "id": "x1", "action": "extract", "output": "balance",
  "locators": [{"strategy": "css", "selector": ".balance"}] }

# put it back — it saves a new version and re-validates
curl -H "Authorization: Bearer $TOKEN" -X PUT \\
  -H "Content-Type: application/json" \\
  -d @uc.json $API/api/usecases/$ID`}</pre>
          <p className="hint">
            Ask for the step editor if this is in your way — it is the missing half of the
            extraction work, not a deliberate omission.
          </p>

          <h3>Where the spreadsheet comes from</h3>
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

          <h3>Finding what to run against in the first place</h3>
          <p>
            If you do not already have the list of records — account numbers, references —
            an <code>extract_rows</code> step can read the site&rsquo;s own list page into
            rows. When a run produces rows, the screen offers{' '}
            <strong>Save as a dataset</strong>, and a second use case runs once per row to
            pull the detail. One pass to find what exists, one to fetch it.
          </p>

          <h3>Running the same workflow in dev, UAT and production</h3>
          <p>
            A use case names a <strong>target</strong>; each deployment says what address
            that target has, under <strong>Targets</strong>. The workflow itself carries no
            address, so the same one runs everywhere unchanged. Pick its target on the use
            case screen under <em>Where it runs</em>.
          </p>

          <h3>When a step stops working</h3>
          <p>
            Sites get redesigned and a recorded control moves. The run stops at that step
            and shows the page as it was, with <strong>Fix it with AI</strong>: it reads
            that page, proposes a new locator, and saves it as a new draft version for you
            to approve. What it learns is remembered, so the same change on the same site
            costs one model call rather than one per run.
          </p>

          <h3>Being polite to the site</h3>
          <p>
            <em>Pace</em> on the use case screen sets the wait between rows. A long
            extraction that reads as an attack gets the account blocked — and automating a
            site you do not own can breach its terms even when the data is yours. Worth
            checking before a few thousand rows.
          </p>
        </div>
      </aside>
    </>
  );
}
