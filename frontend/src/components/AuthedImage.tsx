/**
 * An `<img>` for a URL that requires the session token.
 *
 * A plain `<img src="/api/artifacts/…">` cannot work once the API is
 * authenticated: the browser issues that request itself, and there is no way
 * to attach an `Authorization` header to it. Every screenshot came back 401
 * and rendered as a broken image — the bug this component exists to fix.
 *
 * So the bytes are fetched with `fetch` (which does carry the header), turned
 * into an object URL, and handed to a normal `<img>`. The alternative — putting
 * the token in the query string, as the WebSocket has to — was rejected here
 * because an image URL ends up in far more places: proxy logs, referrer
 * headers, the browser's own history and cache index.
 *
 * Object URLs are revoked on unmount and whenever the source changes. Without
 * that, a long session paging through screenshots leaks every one of them for
 * as long as the tab stays open.
 */

import { useEffect, useState } from 'react';

import { session } from '../lib/session';

interface Props {
  src: string;
  alt: string;
  className?: string;
}

type State =
  | { status: 'loading' }
  | { status: 'ready'; href: string }
  | { status: 'error'; message: string };

export function AuthedImage({ src, alt, className }: Props) {
  const [state, setState] = useState<State>({ status: 'loading' });

  useEffect(() => {
    let objectUrl: string | null = null;
    // Guards against a slow response for an image the user has already paged
    // past: without it, an earlier fetch can overwrite a later one's result.
    let current = true;
    const controller = new AbortController();

    setState({ status: 'loading' });

    fetch(src, { headers: session.headers(), signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(
            response.status === 404
              ? 'This screenshot is recorded but its file is missing from storage.'
              : `Could not load the screenshot (${response.status}).`,
          );
        }
        return response.blob();
      })
      .then((blob) => {
        if (!current) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ status: 'ready', href: objectUrl });
      })
      .catch((error: Error) => {
        if (!current || error.name === 'AbortError') return;
        setState({ status: 'error', message: error.message });
      });

    return () => {
      current = false;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  if (state.status === 'loading') {
    return <span className="empty">Loading screenshot…</span>;
  }
  if (state.status === 'error') {
    return <span className="empty">{state.message}</span>;
  }
  return <img className={className} src={state.href} alt={alt} />;
}
