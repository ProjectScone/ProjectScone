import {useEffect, useState} from 'react';
import type {ApiClient, RecallResponse} from './types';

interface SearchReceipt {
  api: ApiClient;
  query: string;
  attempt: number;
  data?: RecallResponse;
  error?: string;
}

/** A response belongs to one submitted query, credential client and attempt.
 * Do not reuse the ledger's stale-while-refreshing behavior for search evidence. */
export function useSearchRecall(api: ApiClient, query: string, enabled: boolean) {
  const [attempt, setAttempt] = useState(0);
  const [receipt, setReceipt] = useState<SearchReceipt | null>(null);
  useEffect(() => {
    if (!enabled) { setReceipt(null); return; }
    const controller = new AbortController();
    setReceipt(null);
    api.request<RecallResponse>('/v1/recall?' + query, {
      signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]),
    }).then(
      data => { if (!controller.signal.aborted) setReceipt({api, query, attempt, data}); },
      error => { if (!controller.signal.aborted) setReceipt({api, query, attempt,
        error: error instanceof Error ? error.message : 'Search unavailable. Try again.'}); },
    );
    return () => controller.abort();
  }, [api, query, enabled, attempt]);

  // Gate during render, before effect cleanup, so old images unmount immediately.
  const current = enabled && receipt?.api === api && receipt.query === query
    && receipt.attempt === attempt ? receipt : null;
  return {data: current?.data, error: current?.error, retry: () => setAttempt(n => n + 1)};
}
