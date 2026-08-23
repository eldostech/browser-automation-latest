/** Small presentation helpers shared by the views. */

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '--';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

export function formatTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function formatRelative(iso: string | null): string {
  if (!iso) return '--';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const deltaSeconds = Math.round((Date.now() - then) / 1000);
  if (deltaSeconds < 60) return 'just now';
  if (deltaSeconds < 3600) return `${Math.floor(deltaSeconds / 60)}m ago`;
  if (deltaSeconds < 86400) return `${Math.floor(deltaSeconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export function truncate(text: string, max = 400): string {
  return text.length <= max ? text : `${text.slice(0, max)}...`;
}

export function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** Compact one-line rendering of tool arguments for the timeline header. */
export function summariseArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args ?? {});
  if (entries.length === 0) return '';
  return entries
    .map(([key, value]) => {
      const rendered =
        typeof value === 'string' ? value : JSON.stringify(value ?? null) ?? 'null';
      return `${key}=${truncate(rendered, 60)}`;
    })
    .join('  ');
}
