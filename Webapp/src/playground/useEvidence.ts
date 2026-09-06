import { useEffect, useState } from 'react';
import type { ApiClient } from '../api';
import { EMPTY_GRAPH, type EvidenceGraph } from '../types';

export function useEvidence(api: ApiClient, enabled: boolean, paused: boolean, revision: number) {
  const [graph, setGraph] = useState<EvidenceGraph>(EMPTY_GRAPH);
  const [error, setError] = useState('');
  const [updated, setUpdated] = useState('');
  useEffect(() => {
    if (!enabled) { setGraph(EMPTY_GRAPH); setError('Connection needs authentication'); return; }
    if (paused) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      try {
        const next = await api.request<EvidenceGraph>('/v1/graph?limit=200', {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(8000)]),
        });
        if (!Array.isArray(next.nodes) || !Array.isArray(next.edges)) throw Error('Invalid evidence response');
        if (active) { setGraph(next); setError(''); setUpdated(new Date().toLocaleTimeString()); }
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : 'Evidence unavailable');
      } finally { if (active) timer = setTimeout(poll, 1000); }
    }
    void poll();
    return () => { active = false; controller.abort(); clearTimeout(timer); };
  }, [api, enabled, paused, revision]);
  return { graph, error, updated };
}
