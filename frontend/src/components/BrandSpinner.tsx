/**
 * The one busy indicator for the whole app, so a person never has to guess
 * whether something is happening from a button label that merely swapped its
 * verb tense ("Save" -> "Saving...").
 *
 * The mark is the favicon's own dial-and-check redrawn to animate: a dashed
 * ring in motion is "working"; the same ring with the check being traced
 * on a loop is "checking something specific" (verifying a replay, diagnosing
 * a failure) rather than merely busy; a dimmed, still ring is "waiting on a
 * person", not the system; and it settles into a solid ring with the check
 * (or, on failure, a cross) drawn once and left there -- the resting mark
 * *is* the favicon, so success reads as "this became the logo again".
 */

import type { ReactNode } from 'react';

export type SpinnerState = 'working' | 'validating' | 'waiting' | 'success' | 'error';
export type SpinnerLayout = 'inline' | 'block' | 'overlay' | 'blocking';

const DEFAULT_LABEL: Record<SpinnerState, string> = {
  working: 'Working…',
  validating: 'Checking…',
  waiting: 'Waiting…',
  success: 'Done',
  error: 'Failed',
};

const CHECK_PATH = 'M9.5 17 L14 21.5 L22.5 10.5';
const CROSS_PATH = 'M11 11 L21 21 M21 11 L11 21';

interface MarkProps {
  state: SpinnerState;
  size: number;
}

function Mark({ state, size }: MarkProps) {
  const dashed = state === 'working' || state === 'validating';
  return (
    <svg
      className={`brand-mark ${state}`}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      aria-hidden="true"
      focusable="false"
    >
      <circle
        className="ring"
        cx="16"
        cy="16"
        r="11"
        fill="none"
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeDasharray={dashed ? '10 8' : undefined}
      />
      <circle className="hub" cx="16" cy="16" r="2.4" />
      {(state === 'success' || state === 'error') && (
        <path
          className="mark-path"
          d={state === 'success' ? CHECK_PATH : CROSS_PATH}
          fill="none"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      {state === 'validating' && (
        <path
          className="scan-path"
          d={CHECK_PATH}
          fill="none"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
    </svg>
  );
}

interface BrandSpinnerProps {
  state?: SpinnerState;
  /** What's actually happening, in plain words -- this is the point. */
  label?: ReactNode;
  /** A second line: why it takes a moment, or what to expect next. */
  detail?: ReactNode;
  size?: number;
  layout?: SpinnerLayout;
  className?: string;
}

/**
 * `inline`: sits mid-sentence, next to a button label or a heading.
 * `block`: replaces a panel's content while there is nothing else to show.
 * `overlay`: dims one panel, from inside it. The parent needs
 * `position: relative` (add the `.spinner-host` class), and the panel needs to
 * be roughly a screenful -- an overlay centres itself in its host, so on a
 * host taller than the window the spinner lands below the fold.
 * `blocking`: dims the *window* and centres itself in the viewport. For an
 * action the whole screen has to wait for. This is the one to reach for on a
 * long page: `overlay` on a full page view put the spinner at the centre of
 * the document, which meant clicking Run at the top of a use case dimmed the
 * screen and showed nothing until you scrolled half a page down to find it.
 */
export function BrandSpinner({
  state = 'working',
  label,
  detail,
  size,
  layout = 'inline',
  className,
}: BrandSpinnerProps) {
  const resolvedSize = size ?? (layout === 'inline' ? 16 : 40);
  // A blocking wait speaks over whatever else is announcing itself: the
  // screen is unusable until it clears, which is not polite news.
  const urgent = layout === 'blocking';
  const announced = typeof label === 'string' ? label : DEFAULT_LABEL[state];

  return (
    <span
      className={`brand-spinner ${layout} ${state}${className ? ` ${className}` : ''}`}
      role={urgent ? 'alert' : 'status'}
      aria-live={urgent ? 'assertive' : 'polite'}
      aria-busy={state === 'working' || state === 'validating'}
    >
      <Mark state={state} size={resolvedSize} />
      {label !== undefined && <span className="spinner-label">{label}</span>}
      {label === undefined && <span className="sr-only">{announced}</span>}
      {detail !== undefined && <span className="spinner-detail">{detail}</span>}
    </span>
  );
}
