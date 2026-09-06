/**
 * How to actually use this thing.
 *
 * Written because the product is not self-evident: recording a workflow is
 * discoverable, and getting data *out* of a website is not. The panel is
 * organised by what someone is trying to do rather than by what the software
 * is made of, and it names what still has no screen — telling somebody to
 * click a button that does not exist is worse than telling them there is none.
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
