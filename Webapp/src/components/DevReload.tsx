import { useEffect, useState } from 'react';

/** Opt-in native preview refresh; Vite uses its own React Fast Refresh. */
export function DevReload() {
  const enabled = new URLSearchParams(location.search).get('dev') === '1';
  const [status, setStatus] = useState('Checking UI revision…');
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let revision: string | null = null;
    let timer: ReturnType<typeof setTimeout>;
    async function check() {
      try {
        const response = await fetch(location.pathname, { method: 'HEAD', cache: 'no-store', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(3000)]) });
        if (controller.signal.aborted) return;
        const next = response.ok ? response.headers.get('ETag') : null;
        if (!next) setStatus('UI auto-refresh unavailable');
        else if (revision && revision !== next) {
          if (document.activeElement?.matches('input,textarea') || document.querySelector('dialog[open]')) setStatus('UI update ready · finish editing to reload');
          else { location.reload(); return; }
        } else { revision = next; setStatus('UI auto-refresh on'); }
      } catch { if (!controller.signal.aborted) setStatus('UI auto-refresh reconnecting'); }
      if (!controller.signal.aborted) timer = setTimeout(check, 2000);
    }
    void check();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [enabled]);
  return enabled ? <span id="dev-state" role="status">{status}</span> : null;
}
