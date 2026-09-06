import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError } from '../api';
import { Modal } from '../components/Modal';
import type { ApiClient, Episode, Fact } from './types';
import './review.css';

const PAGE_SIZE = 25;
type Sort = 'oldest' | 'newest' | 'subject';
type Outcome = { id: number; text: string; kind: 'approved' | 'duplicate' | 'declined' | 'uncertain' };
type Batch = { ids: number[]; outcomes: Outcome[]; stopped: boolean };
const subjectOf = (fact: Fact) => fact.subject.trim() || 'Unspecified subject';
const messageOf = (error: unknown) => error instanceof Error ? error.message : String(error);

/** A source error does not mean the source was deleted. Fetch only on inspection. */
function SourceExcerpt({ api, id }: { api: ApiClient; id?: number | null }) {
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), [api, id]);
  const load = async () => {
    if (id == null || loading) return;
    const request = new AbortController();
    controller.current = request;
    setLoading(true); setError('');
    try {
      const source = await api.request<Episode>(`/v1/episodes/${id}`, {
        signal: AbortSignal.any([request.signal, AbortSignal.timeout(10000)]),
      });
      if (!request.signal.aborted) setEpisode(source);
    } catch (e) {
      if (!request.signal.aborted) setError(e instanceof ApiError && e.status === 404
        ? 'This source is not available in the current space.' : messageOf(e));
    } finally { if (!request.signal.aborted) setLoading(false); }
  };
  return <div className="excerpt">
    {id == null ? <span>No source episode recorded</span> : <details onToggle={event => {
      if (event.currentTarget.open && !episode && !error && !loading) void load();
    }}>
      <summary>Read full source <span>Episode #{id}</span></summary>
      {loading && <p role="status">Loading source…</p>}
      {error && <div><p role="alert">{error}</p><button className="btn small quiet" onClick={() => void load()}>Retry source</button></div>}
      {episode && <><div className="source-meta">{episode.source || 'Recorded episode'} · {episode.created_at?.slice(0, 10)}</div><div className="review-source">{episode.content}</div></>}
    </details>}
  </div>;
}

function confirmedOutcome(id: number, response: Fact, action: 'approve' | 'decline'): Outcome {
  if (!Number.isSafeInteger(response?.fact_id) || response.fact_id <= 0) throw new Error('Server returned an unrecognized decision receipt');
  if (action === 'decline' && response.fact_id === id && response.status === 'declined')
    return { id, kind: 'declined', text: `Proposal #${id} declined.` };
  if (action === 'approve' && (response.status === 'active' || response.status === 'closed')) {
    if (response.fact_id !== id) return { id, kind: 'duplicate', text: `Proposal #${id} resolved as duplicate of #${response.fact_id}.` };
    return { id, kind: 'approved', text: `Proposal #${id} approved${response.status === 'closed' ? ' as historical memory' : ''}.` };
  }
  throw new Error('Server returned an unrecognized decision receipt');
}

export function ReviewView({ api, onChanged }: { api: ApiClient; onChanged: () => void }) {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [loaded, setLoaded] = useState(false), [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [search, setSearch] = useState(''), [subject, setSubject] = useState<string | null>(null);
  const [origin, setOrigin] = useState('all'), [evidence, setEvidence] = useState('all');
  const [sort, setSort] = useState<Sort>('oldest'), [page, setPage] = useState(0);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [declining, setDeclining] = useState<number | null>(null), [reason, setReason] = useState('');
  const [saving, setSaving] = useState<number | null>(null);
  const [confirmation, setConfirmation] = useState<Fact[] | null>(null);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [notice, setNotice] = useState('');
  const [failure, setFailure] = useState('');
  const [needsRefresh, setNeedsRefresh] = useState(false);
  const busy = useRef(false), interacting = useRef(false), stop = useRef(false);
  const lifetime = useRef<AbortController | null>(null), queueRequest = useRef<AbortController | null>(null);
  interacting.current = confirmation !== null || declining !== null || needsRefresh;

  const refresh = useCallback(async () => {
    if (busy.current || lifetime.current?.signal.aborted) return;
    queueRequest.current?.abort();
    const controller = new AbortController(); queueRequest.current = controller;
    setLoading(true);
    try {
      const response = await api.request<{ facts: Fact[] }>('/v1/facts?status=proposed', {
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]),
      });
      if (controller.signal.aborted) return;
      setFacts(response.facts.filter(f => f.status === 'proposed'));
      setLoaded(true); setLoadError(''); setNeedsRefresh(false);
      setFailure('');
    } catch (error) {
      if (!controller.signal.aborted) setLoadError(messageOf(error));
    } finally { if (!controller.signal.aborted) setLoading(false); }
  }, [api]);

  useEffect(() => {
    lifetime.current = new AbortController();
    void refresh();
    const timer = window.setInterval(() => {
      if (!interacting.current && !document.hidden) void refresh();
    }, 15000);
    return () => { window.clearInterval(timer); lifetime.current?.abort(); queueRequest.current?.abort(); stop.current = true; };
  }, [refresh]);

  const subjects = useMemo(() => {
    const groups = new Map<string, number>();
    for (const f of facts) groups.set(subjectOf(f), (groups.get(subjectOf(f)) || 0) + 1);
    return [...groups].sort(([a], [b]) => a.localeCompare(b));
  }, [facts]);
  const matching = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return facts.filter(f => (subject === null || subjectOf(f) === subject)
      && (origin === 'all' || (f.origin || 'unknown') === origin)
      && (evidence === 'all' || (evidence === 'quoted' ? f.grounded === true : f.grounded !== true))
      && (!query || [f.subject, f.predicate.replaceAll('_', ' '), f.object, f.quote, String(f.fact_id), String(f.source_episode_id ?? '')].join(' ').toLocaleLowerCase().includes(query)))
      .sort((a, b) => sort === 'newest' ? b.fact_id - a.fact_id : sort === 'subject'
        ? subjectOf(a).localeCompare(subjectOf(b)) || a.fact_id - b.fact_id : a.fact_id - b.fact_id);
  }, [facts, subject, origin, evidence, search, sort]);
  useEffect(() => setPage(0), [subject, origin, evidence, search, sort]);
  const currentPage = Math.min(page, Math.max(0, Math.ceil(matching.length / PAGE_SIZE) - 1));
  const visible = matching.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);
  const declineVisible = visible.some(f => f.fact_id === declining && !collapsed.has(subjectOf(f)));
  useEffect(() => {
    if (declining !== null && !declineVisible && saving === null) {
      setDeclining(null); setReason('');
    }
  }, [declining, declineVisible, saving]);
  const groups = new Map<string, Fact[]>();
  for (const fact of visible) groups.set(subjectOf(fact), [...(groups.get(subjectOf(fact)) || []), fact]);
  const locked = saving !== null || confirmation !== null;
  const cannotDecide = locked || loading || needsRefresh || Boolean(loadError);

  const decide = async (snapshot: Fact[], action: 'approve' | 'decline') => {
    if (busy.current || !snapshot.length || needsRefresh || (action === 'decline' && !reason.trim())) return;
    busy.current = true; stop.current = false;
    queueRequest.current?.abort(); setLoading(false);
    setConfirmation(null); setNotice(''); setFailure('');
    // Placement mutates the temporal ledger; screen sorting must not change it.
    const ordered = [...snapshot].sort((a, b) => a.valid_from.localeCompare(b.valid_from) || a.fact_id - b.fact_id);
    const result: Batch = { ids: ordered.map(f => f.fact_id), outcomes: [], stopped: false };
    const bulk = snapshot.length > 1;
    setBatch(bulk ? { ...result } : null);
    for (const fact of ordered) {
      if (stop.current || lifetime.current?.signal.aborted) { result.stopped = true; break; }
      const id = fact.fact_id; setSaving(id);
      try {
        const response = await api.request<Fact>(`/v1/facts/${id}/${action}`, {
          method: 'POST', headers: { 'X-Scone-Actor': 'Scone Review' },
          body: action === 'decline' ? JSON.stringify({ reason: reason.trim() }) : undefined,
          signal: AbortSignal.any([lifetime.current!.signal, AbortSignal.timeout(10000)]),
        });
        if (lifetime.current?.signal.aborted) break;
        const outcome = confirmedOutcome(id, response, action);
        result.outcomes.push(outcome);
        setFacts(previous => previous.filter(f => f.fact_id !== id));
        if (!bulk) setNotice(outcome.text);
      } catch (error) {
        if (lifetime.current?.signal.aborted) break;
        const text = `Proposal #${id}: ${messageOf(error)}. No decision was confirmed. Refresh the queue to check its status before trying again.`;
        result.outcomes.push({ id, kind: 'uncertain', text });
        setFailure(text); setNeedsRefresh(true); result.stopped = true;
        break;
      } finally { if (bulk && !lifetime.current?.signal.aborted) setBatch({ ...result, outcomes: [...result.outcomes] }); }
    }
    busy.current = false;
    if (lifetime.current?.signal.aborted) return;
    setSaving(null); setDeclining(null); setReason('');
    if (bulk) setBatch({ ...result, outcomes: [...result.outcomes] });
    onChanged();
    // Uncertain writes are never automatically retried or reconciled away.
    if (!result.outcomes.some(o => o.kind === 'uncertain')) void refresh();
  };

  const clearFilters = () => { setSearch(''); setSubject(null); setOrigin('all'); setEvidence('all'); };
  return <section className="review-workspace" aria-label="Review inbox">
    <div className="review-overview">
      <div><span className="review-kicker">Your review inbox</span><strong>{loaded ? facts.length.toLocaleString() : '—'}<span>awaiting review</span></strong></div>
      <p>Organized by subject.<br />You decide what becomes memory.</p>
      <button className="btn small quiet" disabled={locked || loading} onClick={() => { void refresh(); onChanged(); }}>Refresh queue</button>
    </div>
    {loadError && <div className="review-error" role="alert">Could not refresh the queue: {loadError}. Existing records may be stale. Use Refresh queue to reconnect.</div>}
    <div className="review-inbox">
      <aside className="review-categories" aria-label="Subject categories">
        <div className="review-kicker">Subjects <span>{subjects.length}</span></div>
        <button aria-pressed={subject === null} disabled={locked} onClick={() => setSubject(null)}><span>All subjects</span><b>{facts.length}</b></button>
        <div className="review-category-list">{subjects.map(([name, count]) => <button key={name} disabled={locked} aria-pressed={subject === name} aria-label={`${name}, ${count} proposals`} onClick={() => setSubject(name)}><span>{name}</span><b>{count}</b></button>)}</div>
        <p>Groups use recorded subjects, not inferred topics. New proposals stay pending until you approve them.</p>
      </aside>
      <div className="review-content">
        <div className="review-controls">
          <label className="review-search">Search proposals<input type="search" value={search} disabled={locked} placeholder="Subject, claim, quote or record ID" onChange={event => setSearch(event.target.value)} /></label>
          <div className="review-filter-row">
            <label>Origin<select value={origin} disabled={locked} onChange={event => setOrigin(event.target.value)}><option value="all">All origins</option><option value="extracted">Extracted</option><option value="stated">Stated</option><option value="inferred">Inferred</option><option value="unknown">Not recorded</option></select></label>
            <label>Evidence<select value={evidence} disabled={locked} onChange={event => setEvidence(event.target.value)}><option value="all">All evidence</option><option value="quoted">Source quote checked</option><option value="unchecked">No checked quote</option></select></label>
            <label>Sort queue<select value={sort} disabled={locked} onChange={event => setSort(event.target.value as Sort)}><option value="oldest">Oldest record first</option><option value="newest">Newest record first</option><option value="subject">Subject A–Z</option></select></label>
          </div>
        </div>
        <div className="review-toolbar"><span>{matching.length} matching · {facts.length} total</span><button className="btn small" disabled={cannotDecide || !matching.length} onClick={() => setConfirmation([...matching])}>Approve all matching ({matching.length})</button></div>
        {(notice || !loaded) && <p className="review-notice" role="status">{notice || 'Loading review queue…'}</p>}
        {batch && <div className="review-result"><div role="status"><strong>{batch.outcomes.filter(o => o.kind === 'approved').length} approved</strong> · {batch.outcomes.filter(o => o.kind === 'duplicate').length} duplicates resolved · {batch.outcomes.filter(o => o.kind === 'uncertain').length} unconfirmed · {batch.ids.length - batch.outcomes.length} {saving !== null ? 'remaining' : 'not attempted'}</div>
          {saving !== null && <button className="btn small quiet" onClick={() => { stop.current = true; }}>Stop after current</button>}
          <details><summary>Decision receipts</summary><ul>{batch.outcomes.map(o => <li key={o.id}>{o.text}</li>)}</ul></details></div>}
        {failure && <p className="review-error" role="alert">{failure}</p>}
        {needsRefresh && <p className="review-reconcile">Decisions are paused until you refresh. A timed-out request may still have reached the server.</p>}
        {loaded && !matching.length && <div className="review-empty"><span className="review-empty-symbol" aria-hidden="true">✓</span><h2>{facts.length ? 'No matching proposals' : 'Nothing awaits review.'}</h2><p>{facts.length ? 'Try another subject or clear your filters.' : 'New claims will appear here for your review. Nothing is approved automatically.'}</p>{facts.length > 0 && <button className="btn small quiet" onClick={clearFilters}>Clear filters</button>}</div>}
        {[...groups].map(([name, rows]) => <section className="review-group" key={name}>
          <button className="review-group-heading" aria-expanded={!collapsed.has(name)} onClick={() => setCollapsed(previous => { const next = new Set(previous); if (next.has(name)) next.delete(name); else next.add(name); return next; })}><span aria-hidden="true">{collapsed.has(name) ? '›' : '⌄'}</span><h2>{name}</h2><span>{rows.length} on this page</span></button>
          {!collapsed.has(name) && rows.map(f => <article className="proposal" data-fact-id={f.fact_id} key={f.fact_id} aria-busy={saving === f.fact_id}>
            <div className="review-claim">
              <div className="review-record"><span>#{f.fact_id} · {f.origin || 'Origin not recorded'}</span><span className={f.grounded === true ? 'review-grounded' : 'review-unchecked'}>{f.grounded === true ? 'Source quote checked' : 'No checked quote'}</span></div>
              <div className="triple"><b>{f.subject}</b> {f.predicate.replaceAll('_', ' ')} <span>{f.object}</span></div>
              <div className="review-claim-meta">Effective {f.valid_from.slice(0, 10)} · Model confidence {Number.isFinite(f.confidence) ? f.confidence.toFixed(2) : 'not recorded'} <span title="Model-provided confidence is not calibrated probability">(uncalibrated)</span></div>
              {f.quote && <blockquote className="quote">“{f.quote}”</blockquote>}
            </div>
            <div className="decide"><button className="btn small" disabled={cannotDecide} onClick={() => void decide([f], 'approve')}>{saving === f.fact_id ? 'Saving…' : 'Approve'}</button><button className="btn small danger" disabled={cannotDecide} onClick={() => { setDeclining(f.fact_id); setReason(''); }}>Decline</button></div>
            {declining === f.fact_id && <form className="review-decline" onSubmit={event => { event.preventDefault(); void decide([f], 'decline'); }}><label htmlFor={`decline-${f.fact_id}`}>Reason for declining</label><p>The proposal stays in history with this reason. Its source is not deleted.</p><textarea id={`decline-${f.fact_id}`} autoFocus required maxLength={2000} value={reason} disabled={locked} onChange={event => setReason(event.target.value)} /><div className="decide"><button type="submit" className="btn small danger" disabled={cannotDecide || !reason.trim()}>Confirm decline</button><button className="btn small quiet" type="button" disabled={locked} onClick={() => setDeclining(null)}>Cancel</button></div></form>}
            <SourceExcerpt api={api} id={f.source_episode_id} />
          </article>)}
        </section>)}
        {matching.length > PAGE_SIZE && <nav className="review-pagination" aria-label="Review pages"><button className="btn small quiet" disabled={locked || currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button><span>{currentPage * PAGE_SIZE + 1}–{Math.min((currentPage + 1) * PAGE_SIZE, matching.length)} of {matching.length}</span><button className="btn small quiet" disabled={locked || (currentPage + 1) * PAGE_SIZE >= matching.length} onClick={() => setPage(currentPage + 1)}>Next</button></nav>}
        <p className="review-footnote">Pages use the selected record order, then group cards by subject. Within each group, order uses record IDs, not belief dates. Refreshes every 15 seconds while idle. Bulk approval includes every matching page, not just the cards shown.</p>
      </div>
    </div>
    {confirmation && <Modal title={`Approve ${confirmation.length} proposals?`} onClose={() => setConfirmation(null)}>
      <p className="setup-intro">Approve this exact set of {confirmation.length} matching proposals across all pages. New arrivals are not included. Approval can replace existing beliefs or add historical memory.</p>
      <p className="setup-intro">{confirmation.filter(f => f.grounded !== true).length} have no checked source quote. Model confidence is not a guarantee of correctness.</p>
      <details className="review-confirm-list"><summary>Inspect the selected claims</summary><ul>{confirmation.map(f => <li key={f.fact_id}><b>#{f.fact_id} {f.subject}</b> {f.predicate.replaceAll('_', ' ')} {f.object}</li>)}</ul></details>
      <p className="setup-note">Decisions run one at a time, oldest effective date first (record ID breaks ties), regardless of display sorting. This is not an atomic transaction: confirmed decisions remain saved if a later request fails. There is no bulk undo.</p>
      <div className="review-confirm-actions"><button onClick={() => setConfirmation(null)}>Cancel</button><button className="primary" onClick={() => void decide(confirmation, 'approve')}>Confirm approval</button></div>
    </Modal>}
  </section>;
}
