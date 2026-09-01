/**
 * The signed-in session.
 *
 * One module owns the token, so nothing else has to think about where it is
 * kept or remember to attach it. `api.ts` reads it on every request; the
 * WebSocket reads it too, because a browser cannot set a header on a
 * handshake.
 *
 * **Where it lives.** `localStorage`, so a refresh does not sign you out. That
 * is a deliberate trade: a token in `localStorage` is readable by any script
 * that gets injected into this origin, whereas an httpOnly cookie is not.
 * Cookies would bring CSRF back and require the backend to sit on the same
 * site, and the session here is short-lived and individually revocable
 * server-side -- so a leaked token is bounded in both time and blast radius.
 * If this ever holds anything more valuable, revisit that.
 */

const TOKEN_KEY = 'understudy.token';
const USER_KEY = 'understudy.user';

export interface CurrentUser {
  id: string;
  workspace_id: string;
  email: string;
  display_name: string;
  role: string;
  /** Resolved server-side. Used to decide what to render, never to permit. */
  permissions: string[];
}

type Listener = (user: CurrentUser | null) => void;

const listeners = new Set<Listener>();

/** Reads never throw: private-browsing modes can make storage unavailable. */
function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: the session lasts as long as the tab does */
  }
}

let token: string | null = read(TOKEN_KEY);
let user: CurrentUser | null = parseUser(read(USER_KEY));

function parseUser(raw: string | null): CurrentUser | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CurrentUser;
  } catch {
    return null;
  }
}

function notify(): void {
  for (const listener of listeners) listener(user);
}

export const session = {
  get token(): string | null {
    return token;
  },

  get user(): CurrentUser | null {
    return user;
  },

  get isSignedIn(): boolean {
    return token !== null;
  },

  /**
   * Whether the UI should offer an action.
   *
   * A convenience, not a security boundary: every one of these is checked
   * again on the server. Hiding a button the server would refuse is better
   * manners than showing it and returning 403.
   */
  can(permission: string): boolean {
    return user?.permissions?.includes(permission) ?? false;
  },

  start(newToken: string, newUser: CurrentUser): void {
    token = newToken;
    user = newUser;
    write(TOKEN_KEY, newToken);
    write(USER_KEY, JSON.stringify(newUser));
    notify();
  },

  /** Forget the session locally. Revoking it server-side is `api.logout`. */
  clear(): void {
    token = null;
    user = null;
    write(TOKEN_KEY, null);
    write(USER_KEY, null);
    notify();
  },

  /** Subscribe to sign-in and sign-out. Returns an unsubscribe function. */
  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  /** Authorization header for a fetch, or nothing when signed out. */
  headers(): Record<string, string> {
    return token ? { Authorization: `Bearer ${token}` } : {};
  },

  /**
   * Append the token to a WebSocket URL.
   *
   * A query parameter because the WebSocket handshake cannot carry headers.
   * The backend accepts it there for the same reason.
   */
  withToken(url: string): string {
    if (!token) return url;
    return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
  },
};
