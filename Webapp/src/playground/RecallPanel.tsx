import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { ApiClient } from '../api';
import type { RecallResult } from '../types';
import { SourceImages } from '../components/SourceImages';
import { MarkdownText, SourceContent } from '../components/SourceContent';

export function RecallPanel({ api, enabled, onRecalled }: {api:ApiClient;enabled:boolean;onRecalled:()=>void}) {
  const [query,setQuery] = useState('');
  const [submitted,setSubmitted] = useState('');
  const [result,setResult] = useState<RecallResult | null>(null);
  const [message,setMessage] = useState('');
  const [pending,setPending] = useState(false);
  const request = useRef<AbortController | null>(null);
  useEffect(() => {
    request.current?.abort();setResult(null);setSubmitted('');setMessage('');setPending(false);
    return () => request.current?.abort();
  }, [api,enabled]);
  function change(value:string) {
    request.current?.abort();setQuery(value);setSubmitted('');setResult(null);setMessage('');setPending(false);
  }
  async function recall(event:FormEvent) {
    event.preventDefault(); const q = query.trim(); if (!q || !enabled) return;
    request.current?.abort(); const controller = new AbortController();request.current=controller;
    setSubmitted(q);setPending(true);setResult(null);setMessage('');
    try {
      const data = await api.request<RecallResult>('/v1/recall?q='+encodeURIComponent(q)+'&limit=5', {signal:AbortSignal.any([controller.signal,AbortSignal.timeout(8000)])});
      if (!controller.signal.aborted) {setResult(data);onRecalled();}
    } catch (error) {if(!controller.signal.aborted)setMessage(error instanceof Error ? error.message : 'Recall unavailable. Try again.');}
    finally {if(!controller.signal.aborted)setPending(false);}
  }
  return <section className="recall-panel" aria-label="Memory recall">
    <div className="section-title"><div><span className="eyebrow">Retrieval workspace</span><h2>Test what comes back</h2></div><span className="recall-badge">Actual engine recall</span></div>
    <form className="recall-form" onSubmit={recall}><input aria-label="Test memory recall" placeholder="Ask about a decision, preference or project…" maxLength={1000} required value={query} onChange={e=>change(e.target.value)} /><button className="primary" disabled={!enabled || pending || !query.trim()}>{pending ? 'Searching…' : 'Recall'}</button></form>
    <p className="recall-hint">Searches this memory space. Returned candidates may be unrelated; ranking is not confidence or proof of correctness.</p>
    <div id="recall-results" aria-live="polite" aria-busy={pending}>
      {submitted && <div className="recall-query"><span>Results for</span><strong>{submitted}</strong>{result && <small>{result.items.length} candidate{result.items.length===1?'':'s'}</small>}</div>}
      {pending && <p className="recall-state">Retrieving stored evidence…</p>}
      {message && <p className="recall-state" role="alert">{message}</p>}
      {result && !result.items.length && <p className="recall-state">No evidence returned for this query.</p>}
      {!submitted && <div className="recall-empty"><span aria-hidden>⌕</span><p>Follow a question back to its source.</p><small>Submit a query to inspect the passages the engine retrieves.</small></div>}
      {result?.items.map((item,i)=><article className="recall-card" key={`${submitted}:${item.episode_id}:${i}`}>
        <div className="recall-card-heading"><span className="recall-rank">{String(i+1).padStart(2,'0')}</span><h3>Episode {item.episode_id}</h3><span>Retrieved passage</span></div>
        {item.text.length > 420 ? <details className="recall-passage"><summary><span><MarkdownText text={item.text.slice(0,420)} inline/>…</span><b>Read full passage</b></summary><div className="recall-full"><SourceContent text={item.text}/></div></details> : <div className="recall-full"><SourceContent text={item.text}/></div>}
        <SourceImages episodeId={item.episode_id} api={api}/>
        <footer><code>episode:{item.episode_id}</code><span>Ranking score {Number.isFinite(item.score) ? Number(item.score).toFixed(3) : 'not reported'} · not confidence</span></footer>
      </article>)}
    </div>
  </section>;
}
