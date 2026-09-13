import type { ModelChoice } from './events';

/**
 * Which model this browser is currently asking for.
 *
 * Kept here, in the browser, and sent on every request that can spend a token.
 * Not stored on the server, and deliberately: comparing two models means
 * running two at once, so a "current model" setting in the workspace would
 * serialise the exact thing it exists to support -- and two people trying
 * different models would silently overwrite each other's choice.
 *
 * `null` means "whatever the deployment is configured for". That is different
 * from having picked the default: a request carrying no choice at all lets the
 * server decide, which is what should happen when nobody has expressed a
 * preference.
 */

const KEY = 'trace.model';

/** Everything that reads the choice, so a change repaints the chrome too. */
const listeners = new Set<(choice: ModelChoice | null) => void>();

export function readChoice(): ModelChoice | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ModelChoice;
    // Both halves required. A half-written choice is how a request ends up
    // naming a provider with no model, which the server then refuses.
    return parsed?.provider && parsed?.model ? parsed : null;
  } catch {
    // A private window, cleared site data, or storage the browser refuses.
    // Falling back to the deployment default is always safe.
    return null;
  }
}

export function writeChoice(choice: ModelChoice | null): void {
  try {
    if (choice === null) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, JSON.stringify(choice));
  } catch {
    // Not fatal: the choice applies to this page's requests either way, it
    // just will not survive a reload.
  }
  listeners.forEach((listener) => listener(choice));
}

export function onChoiceChanged(listener: (choice: ModelChoice | null) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * The choice as request fields, spread into a body.
 *
 * An empty object when nothing is chosen, so the request looks exactly as it
 * did before this existed rather than carrying explicit nulls the server would
 * have to tell apart from absence.
 */
export function modelFields(): { provider?: string; model?: string } {
  const choice = readChoice();
  return choice === null ? {} : { provider: choice.provider, model: choice.model };
}
