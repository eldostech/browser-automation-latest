import { artifactUrl } from '../lib/api';
import type { ScreenshotEvent } from '../lib/events';
import { formatTime } from '../lib/format';

interface Props {
  screenshot: ScreenshotEvent | null;
  /** All screenshots in the run, for stepping back through them. */
  history: ScreenshotEvent[];
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
}

/**
 * The current page as the browser sees it.
 *
 * Screenshots are a human-facing signal only -- the model observes the
 * accessibility snapshot instead. The frame keeps a fixed aspect ratio so
 * swapping images never reflows the column.
 */
export function ScreenshotPane({ screenshot, history, selectedIndex, onSelect }: Props) {
  const isLive = selectedIndex === null;
  const index = isLive ? history.length - 1 : selectedIndex;
  const current = isLive ? screenshot : (history[selectedIndex] ?? screenshot);

  return (
    <div className="panel">
      <header>
        <span>Browser</span>
        {current?.page_url && (
          <span style={{ textTransform: 'none', letterSpacing: 0, color: 'var(--text-faint)' }}>
            {current.page_url}
          </span>
        )}
        <span style={{ marginLeft: 'auto' }}>
          {current ? `step ${current.step} - ${formatTime(current.ts)}` : ''}
        </span>
      </header>

      <div className="shot">
        {current ? (
          <img
            src={artifactUrl(current.url)}
            alt={current.caption ?? `Page at step ${current.step}`}
          />
        ) : (
          <span className="empty">No screenshot yet</span>
        )}
      </div>

      {history.length > 1 && (
        <div className="body row" style={{ borderTop: '1px solid var(--border)' }}>
          <button
            type="button"
            disabled={index <= 0}
            onClick={() => onSelect(Math.max(0, index - 1))}
          >
            Previous
          </button>
          <span style={{ fontSize: 12, color: 'var(--text-faint)' }}>
            {index + 1} / {history.length}
          </span>
          <button
            type="button"
            disabled={index >= history.length - 1}
            onClick={() => onSelect(Math.min(history.length - 1, index + 1))}
          >
            Next
          </button>
          {!isLive && (
            <button type="button" onClick={() => onSelect(null)} style={{ marginLeft: 'auto' }}>
              Follow live
            </button>
          )}
        </div>
      )}
    </div>
  );
}
