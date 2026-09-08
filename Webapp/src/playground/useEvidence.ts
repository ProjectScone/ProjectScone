import { useEffect, useState } from 'react';
import type { ApiClient } from '../api';
import { EMPTY_GRAPH, type EvidenceGraph } from '../types';

export function useEvidence(api: ApiClient, enabled: boolean, paused: boolean, revision: number, sessionId?: string) {
  const query = new URLSearchParams({ limit: '200' });
  if (sessionId) query.set('session_id', sessionId);
  const path = `/v1/graph?${query}`;
  const [snapshot, setSnapshot] = useState({ api, path, graph: EMPTY_GRAPH, error: '', updated: '' });
  useEffect(() => {
    if (!enabled) { setSnapshot({ api, path, graph: EMPTY_GRAPH, error: '', updated: '' }); return; }
    if (paused) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      try {
        const next = await api.request<EvidenceGraph>(path, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(8000)]),
        });
        if (!Array.isArray(next.nodes) || !Array.isArray(next.edges)) throw Error('Invalid evidence response');
        if (active) setSnapshot({ api, path, graph: next, error: '', updated: new Date().toLocaleTimeString() });
      } catch (cause) {
        if (active) setSnapshot(previous => ({ api, path,
          graph: previous.api === api && previous.path === path ? previous.graph : EMPTY_GRAPH,
          updated: previous.api === api && previous.path === path ? previous.updated : '',
          error: cause instanceof Error ? cause.message : 'Evidence unavailable',
        }));
      } finally { if (active) timer = setTimeout(poll, 1000); }
    }
    void poll();
    return () => { active = false; controller.abort(); clearTimeout(timer); };
  }, [api, enabled, paused, revision, path]);
  if (!enabled || snapshot.api !== api || snapshot.path !== path) return { graph: EMPTY_GRAPH, error: '', updated: '' };
  return snapshot;
}
