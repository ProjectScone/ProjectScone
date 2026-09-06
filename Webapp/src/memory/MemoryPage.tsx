// The memory console as a React page. Same semantics as the packaged
// console.html it replaces: every figure carries its n, retrieved text is
// an excerpt, claims say where they came from, and nothing is shown that
// the API did not return. Capabilities the host lacks (lanes, events,
// review) simply do not appear; there are no disabled stand-ins.

import React, { useCallback, useEffect, useRef, useState } from "react";
import type {
  ApiClient,
  Episode,
  EventRecord,
  EventsResponse,
  Fact,
  Metric,
  MetricsReport,
  RecallItem,
  RecallResponse,
  Scopes,
  SearchState,
  Status,
  View,
} from "./types";
import "./memory.css";
import { ReviewView } from "./ReviewView";
import { SourceImages } from '../components/SourceImages';
import { useSearchRecall } from './useSearchRecall';
import { SourceComposer } from './SourceComposer';
import { parseCapabilities, type Capabilities, type Feature } from '../capabilities';
import {claimGroups,filterClaimGroups,type ClaimFilter,type DisplayClaimGroup} from './claim-groups';

const GROUPS: Array<{ label: string; views: Array<[View, string]> }> = [
  { label: "Memory", views: [["search", "Search"], ["beliefs", "Beliefs"], ["review", "Review"]] },
  { label: "Activity", views: [["live", "Live"], ["analytics", "Analytics"]] },
  { label: "System", views: [["scopes", "Scopes"], ["status", "Status"]] },
];
const INTRO: Record<View, [string, string]> = {
  search: ["Search", "Ask in plain words. Every result is an excerpt of something stored, and says where it came from."],
  beliefs: ["Memory claims", "Inspect what Scone has recorded about a subject, check its source, and control whether it can appear in recall."],
  review: ["Review", "Claims a model read out of your memory. Nothing here counts until you approve it."],
  live: ["Live", "What the engine is doing, as it happens."],
  analytics: ["Analytics", "How memory is used, from recorded activity. Each figure shows how much evidence it rests on."],
  scopes: ["Scopes", "Who has memory here: episodes per user, agent or session."],
  status: ["Status", "What this server runs on and how much it holds."],
};

const day = (s?: string | null) => (s ?? "").slice(0, 10);
const clock = (s?: string | null) => (s ?? "").slice(11, 19);
const safeHref = (url: string) => {
  try {
    // A source basename or native path is provenance, not a relative web link.
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
};

function useAsync<T>(load: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  const gen = useRef(0);
  const run = useCallback(() => {
    const g = ++gen.current;
    setState((s) => ({ ...s, loading: true }));
    return load().then(
      (data) => { if (g !== gen.current) return false; setState({ data, loading: false }); return true; },
      (e: unknown) => { if (g === gen.current) setState(s => ({ ...s, error: e instanceof Error ? e.message : String(e), loading: false })); return false; },
    );
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { void run(); return () => { gen.current++; }; }, [run]);
  return { ...state, reload: run };
}

const VIEW_FEATURES: Record<View, Feature> = {
  search: 'recall', beliefs: 'facts.read', review: 'facts.review',
  live: 'events.read', analytics: 'metrics.read', scopes: 'scopes.read', status: 'status.read',
};

function useCapabilities(api: ApiClient) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{api: ApiClient; data?: Capabilities; error?: string} | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setState({api});
    api.request<unknown>('/v1/capabilities', {signal:AbortSignal.any([controller.signal, AbortSignal.timeout(10000)])})
      .then(parseCapabilities).then(data => { if (!controller.signal.aborted) setState({api, data}); })
      .catch(error => { if (!controller.signal.aborted) setState({api, error:error instanceof Error ? error.message : String(error)}); });
    return () => controller.abort();
  }, [api, attempt]);
  return {data:state?.api === api ? state.data : undefined, error:state?.api === api ? state.error : undefined,
    retry:() => {setState({api});setAttempt(value => value + 1);}};
}

export function MemoryPage({ api }: { api: ApiClient }) {
  const discovery = useCapabilities(api);
  const caps = discovery.data?.features;
  const [view, setView] = useState<View>(() => (window.location.hash.slice(1) as View) || "search");
  const [pending, setPending] = useState<number | null>(null);
  const [search, setSearch] = useState<SearchState>(() => {
    const p = new URLSearchParams(window.location.search);
    const where: Record<string, string> = {};
    for (const pair of (p.get("where") ?? "").split(",")) {
      const [k, v] = pair.split(":");
      if (k && v) where[k] = v;
    }
    return { q: p.get("q") ?? "", where, tags: [], asOf: "" };
  });

  useEffect(() => {
    if (!INTRO[view]) { setView("search"); return; }
    const hash = "#" + view;
    if (window.location.hash !== hash) window.history.replaceState(null, "", window.location.pathname + window.location.search + hash);
  }, [view, caps]);

  const refreshPending = useCallback(() => {
    if (!caps?.['facts.review']) return;
    api.request<{ facts: Fact[] }>("/v1/facts?status=proposed").then((r) => setPending(r.facts.length)).catch(() => setPending(null));
  }, [api, caps]);
  useEffect(refreshPending, [refreshPending]);

  const searchInScope = (key: string, value: string) => {
    setSearch((s) => ({ ...s, where: { ...s.where, [key]: value } }));
    setView("search");
  };

  const [title, intro] = INTRO[view] ?? INTRO.search;
  return (
    <div className="memory-page">
      <nav className="sections" aria-label="Sections">
        {GROUPS.filter((g) => caps && g.views.some(([id]) => caps[VIEW_FEATURES[id]])).map((g) => (
          <span key={g.label} className="grpwrap">
            <span className="grp">{g.label}</span>
            {g.views.filter(([id]) => caps?.[VIEW_FEATURES[id]]).map(([id, label]) => (
              <button key={id} className="nav" aria-current={view === id} onClick={() => setView(id)}>
                {label}
                {id === "review" && pending ? <span className="count">{pending}</span> : null}
              </button>
            ))}
          </span>
        ))}
      </nav>
      <main>
        <div className="page-head">
          <div className="eyebrow">{GROUPS.find((g) => g.views.some(([id]) => id === view))?.label}</div>
          <h1>{title}</h1>
          <p>{intro}</p>
        </div>
        {!caps ? discovery.error ? <section role="alert"><h2>Server capabilities unavailable</h2><p>We could not verify which operations this server supports. Your selected page is preserved; no workflow requests were sent.</p><ErrorLine message={discovery.error} /><button className="btn quiet" onClick={discovery.retry}>Retry capabilities</button></section> : <p role="status">Checking server capabilities…</p>
          : !caps[VIEW_FEATURES[view]] ? <Empty>This server does not support this page. Choose an available section above.</Empty>
          : <>
            {view === "search" && <SearchView api={api} state={search} setState={setSearch} onScope={searchInScope} canAddSources={caps['episodes.attachments']} />}
            {view === "beliefs" && <BeliefsView api={api} onChanged={refreshPending} features={caps} />}
            {view === "review" && <ReviewView api={api} onChanged={refreshPending} />}
            {view === "live" && <LiveView api={api} />}
            {view === "analytics" && <AnalyticsView api={api} />}
            {view === "scopes" && <ScopesView api={api} onScope={searchInScope} />}
            {view === "status" && <StatusView api={api} />}
          </>}
      </main>
    </div>
  );
}

/* ---------- shared bits ---------- */

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="empty">{children}</p>;
}
function ErrorLine({ message }: { message: string }) {
  return <p className="err">{message}</p>;
}
function SourceLink({ source }: { source?: string | null }) {
  if (!source) return null;
  const href = safeHref(source);
  return href ? <a href={href} target="_blank" rel="noreferrer noopener">{source}</a> : <span title="source">{source}</span>;
}
function Origin({ origin }: { origin?: string }) {
  const o = origin ?? "stated";
  return <span className={"origin " + o}>{o}</span>;
}
/** Says whether a claim's words were checked against its source. Only for
 *  claims that have a source; a stated claim has nothing to check. */
function Grounding({ f }: { f: Fact }) {
  if (f.grounded === true) return <span className="ground yes" title={f.quote ?? ""}>quoted from source</span>;
  if (f.grounded === false) return <span className="ground" title="A missing quotation does not establish whether the claim is correct.">No stored quotation</span>;
  return null;
}

/* ---------- search ---------- */

function SearchView({ api, state, setState, onScope, canAddSources }: {
  canAddSources: boolean;
  api: ApiClient; state: SearchState; setState: (f: (s: SearchState) => SearchState) => void; onScope: (k: string, v: string) => void;
}) {
  const [draft, setDraft] = useState(state.q);
  const [adding, setAdding] = useState(false);
  const [kind, setKind] = useState<"where" | "tag" | "asof">("where");
  const [fkey, setFkey] = useState("");
  const [fval, setFval] = useState("");
  const [feedback, setFeedback] = useState<Record<string, boolean>>({});
  const active = Boolean(state.q || Object.keys(state.where).length || state.tags.length || state.asOf);

  const params = new URLSearchParams({ q: state.q || "*", limit: "25" });
  const where = Object.entries(state.where).map(([k, v]) => `${k}:${v}`).join(",");
  if (where) params.set("where", where);
  if (state.tags.length) params.set("tags", state.tags.join(","));
  if (state.asOf) params.set("as_of", state.asOf);
  const query = params.toString();

  const results = useSearchRecall(api, query, active);
  const start = useAsync(
    () => Promise.all([
      api.request<Status>("/v1/status"),
      api.request<RecallResponse>("/v1/recall?q=*&limit=6"),
      api.request<Scopes>("/v1/scopes").catch(() => ({ scopes: {} } as Scopes)),
    ]),
    [api, active],
  );

  const submit = () => setState((s) => ({ ...s, q: draft.trim() }));
  const addFilter = () => {
    setState((s) => {
      if (kind === "where" && fkey.trim() && fval.trim()) return { ...s, where: { ...s.where, [fkey.trim()]: fval.trim() } };
      if (kind === "tag" && fval.trim() && !s.tags.includes(fval.trim())) return { ...s, tags: [...s.tags, fval.trim()] };
      if (kind === "asof" && fval) return { ...s, asOf: fval };
      return s;
    });
    setFkey(""); setFval(""); setAdding(false);
  };
  const sendFeedback = async (eventId: number, chunkId: number, useful: boolean) => {
    try {
      await api.request("/v1/feedback", { method: "POST", body: JSON.stringify({ recall_event_id: eventId, chunk_id: chunkId, useful }) });
      setFeedback((f) => ({ ...f, [`${eventId}:${chunkId}`]: useful }));
    } catch { /* the row keeps its state; the API said no */ }
  };

  return (
    <>
      {canAddSources&&<SourceComposer api={api} onSaved={()=>{void start.reload();results.retry();}}/>}
      <div className="search">
        <input type="search" value={draft} placeholder="What have we decided about this project?" autoComplete="off"
          onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") submit(); }} />
        <button className="btn" onClick={submit}>Search</button>
      </div>
      <div className="chips">
        {Object.entries(state.where).map(([k, v]) => (
          <span key={k} className="chip"><b>{k}</b> {v} <button aria-label="remove" onClick={() => setState((s) => { const w = { ...s.where }; delete w[k]; return { ...s, where: w }; })}>×</button></span>
        ))}
        {state.tags.map((t) => (
          <span key={t} className="chip"><b>tag</b> {t} <button aria-label="remove" onClick={() => setState((s) => ({ ...s, tags: s.tags.filter((x) => x !== t) }))}>×</button></span>
        ))}
        {state.asOf && <span className="chip"><b>as of</b> {state.asOf} <button aria-label="remove" onClick={() => setState((s) => ({ ...s, asOf: "" }))}>×</button></span>}
        <button className="chip add" onClick={() => setAdding((a) => !a)}>+ filter</button>
      </div>
      {adding && (
        <div className="addrow">
          <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
            <option value="where">scope key = value</option><option value="tag">tag</option><option value="asof">as of date</option>
          </select>
          {kind === "where" && <input value={fkey} placeholder="user_id" onChange={(e) => setFkey(e.target.value)} style={{ width: 130 }} />}
          <input type={kind === "asof" ? "date" : "text"} value={fval} placeholder={kind === "tag" ? "work" : "alice"} onChange={(e) => setFval(e.target.value)} style={{ width: 160 }} />
          <button className="btn small quiet" onClick={addFilter}>Add</button>
        </div>
      )}

      {!active ? (
        start.error ? <ErrorLine message={start.error} /> : !start.data ? <Empty>Loading…</Empty> : (() => {
          const [status, recent, scopes] = start.data;
          if (!status.episodes) return <Empty>Nothing is stored yet. Store a note with <code>scone-memory remember</code>, or connect an agent from the Playground, and it appears here.</Empty>;
          const chips = Object.entries(scopes.scopes ?? {}).flatMap(([k, vals]) => Object.entries(vals).slice(0, 4).map(([v, n]) => (
            <button key={k + v} className="chip scope" title={`${n} episode(s)`} onClick={() => onScope(k, v)}><b>{k.replace(/_/g, " ")}</b> {v}</button>
          ))).slice(0, 8);
          return (
            <>
              <div className="start">
                <div className="start-row">
                  <span className="stat-n">{status.episodes}</span><span className="stat-l">episodes</span>
                  <span className="stat-n">{status.chunks}</span><span className="stat-l">excerpts</span>
                  {status.pending_review != null && <><span className="stat-n">{status.pending_review}</span><span className="stat-l">awaiting review</span></>}
                </div>
                {chips.length > 0 && <div className="chips" style={{ marginTop: 14 }}><span className="k">Search within</span>{chips}</div>}
              </div>
              <p className="summary-line">Most recent</p>
              <div className="rows">{recent.items.map((i) => <ResultRow key={i.chunk_id} item={i} api={api} onScope={onScope} onTag={(t) => setState((s) => s.tags.includes(t) ? s : { ...s, tags: [...s.tags, t] })} />)}</div>
            </>
          );
        })()
      ) : results.error ? <div role="alert"><ErrorLine message={results.error} /><button className="btn quiet" onClick={results.retry}>Retry search</button></div> : !results.data ? <Empty>Searching…</Empty> : (
        <ResultList r={results.data} api={api} asOf={state.asOf} onScope={onScope} feedback={feedback} sendFeedback={sendFeedback}
          onTag={(t) => setState((s) => s.tags.includes(t) ? s : { ...s, tags: [...s.tags, t] })} />
      )}
    </>
  );
}

function ResultList({ r, api, asOf, onScope, onTag, feedback, sendFeedback }: {
  api: ApiClient;
  r: RecallResponse; asOf: string; onScope: (k: string, v: string) => void; onTag: (t: string) => void;
  feedback: Record<string, boolean>; sendFeedback: (e: number, c: number, u: boolean) => void;
}) {
  if (!r.items.length && !r.facts.length) return <Empty>Nothing matched. Store something with <code>scone-memory remember</code> and ask again.</Empty>;
  const reduction = r.context_reduction != null && r.space_bytes ? `, ${Math.round(r.context_reduction * 100)}% of stored bytes left behind` : "";
  return (
    <>
      <p className="summary-line">
        {r.items.length} excerpt{r.items.length === 1 ? "" : "s"}{reduction}
        {r.degraded.length > 0 && <span className="err"> · one lane failed: {r.degraded.join("; ")}</span>}
      </p>
      {r.facts.length > 0 && (
        <div className="facts-inline">
          <h3>Recorded claims that held{asOf ? ` on ${asOf}` : ""}</h3>
          {r.facts.map((f) => (
            <div key={f.fact_id} className="fact">
              <span className="s">{f.subject}</span><span className="p">{f.predicate.replace(/_/g, " ")}</span><span>{f.object}</span>
              <span className="since">since {day(f.valid_from)}</span><Origin origin={f.origin} /><Grounding f={f} />
            </div>
          ))}
        </div>
      )}
      <div className="rows">
        {r.items.map((i) => (
          <ResultRow key={i.chunk_id} item={i} api={api} onScope={onScope} onTag={onTag}
            why={
              <details className="why"><summary>why this</summary>
                <div className="why-body">
                  <span className="k">rank in this query</span>
                  <span><span className="rank"><i style={{ width: `${Math.round((i.score || 0) * 100)}%` }} /></span> <span className="sub">{(i.score || 0).toFixed(2)}, top is 1.00</span></span>
                  {i.lanes && (
                    <><span className="k">found by</span>
                      <span>{[i.lanes.vector && `vector lane, rank ${i.lanes.vector}`, i.lanes.text && `text lane, rank ${i.lanes.text}`].filter(Boolean).join(" and ") || "unknown"}</span></>
                  )}
                  {i.similarity != null && <><span className="k">cosine similarity</span><span>{i.similarity.toFixed(3)} <span className="sub">uncalibrated</span></span></>}
                  <span className="k">episode</span><span>#{i.episode_id}, chunk #{i.chunk_id}</span>
                  {r.event_id != null && (
                    <><span className="k">useful?</span>
                      <span className="fb">
                        <button className={feedback[`${r.event_id}:${i.chunk_id}`] === true ? "set" : ""} onClick={() => sendFeedback(r.event_id!, i.chunk_id, true)}>yes</button>
                        <button className={feedback[`${r.event_id}:${i.chunk_id}`] === false ? "set" : ""} onClick={() => sendFeedback(r.event_id!, i.chunk_id, false)}>no</button>
                      </span></>
                  )}
                </div>
              </details>
            } />
        ))}
      </div>
    </>
  );
}

function ResultRow({ item, api, onScope, onTag, why }: { item: RecallItem; api: ApiClient; onScope: (k: string, v: string) => void; onTag: (t: string) => void; why?: React.ReactNode }) {
  return (
    <div className="row">
      <div className="meta">
        <span>{day(item.created_at)}</span><SourceLink source={item.source} />
        {Object.entries(item.metadata ?? {}).map(([k, v]) => <button key={k} className="linkish" title={`Search within ${k} ${v}`} onClick={() => onScope(k, v)}>{k.replace(/_/g, " ")} {v}</button>)}
        {(item.tags ?? []).map((t) => <button key={t} className="linkish" title={`Add tag filter ${t}`} onClick={() => onTag(t)}>#{t}</button>)}
      </div>
      <div className="text"><span className="label">excerpt</span>{item.text}</div>
      <SourceImages episodeId={item.episode_id} api={api} />
      {why}
    </div>
  );
}

/* ---------- beliefs ---------- */

function ClaimSource({ fact, api }: { fact: Fact; api: ApiClient }) {
  const [open, setOpen] = useState(false);
  return <div className="claim-source">
    <span className="label">Source evidence</span>
    {fact.quote ? <blockquote className="quote">{fact.quote}</blockquote> : <p>{fact.source_episode_id != null ? 'No stored quotation. Read the original to assess this claim; missing evidence does not mean it is false.' : 'No source episode is attached to this claim. Its source support cannot be checked here.'}</p>}
    {fact.source_episode_id != null && <button className="btn small quiet" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Hide source' : `Read source episode #${fact.source_episode_id}`}</button>}
    {open && fact.source_episode_id != null && <ClaimEpisode key={fact.source_episode_id} id={fact.source_episode_id} api={api} />}
  </div>;
}

function ClaimEpisode({ id, api }: { id: number; api: ApiClient }) {
  const source = useAsync(() => api.request<Episode>(`/v1/episodes/${id}`, {signal: AbortSignal.timeout(10000)}), [api, id]);
  return <div className="claim-episode">{source.error ? <><ErrorLine message={`Source unavailable: ${source.error}`} /><button className="btn small quiet" onClick={() => void source.reload()}>Retry source</button></> : !source.data ? <p role="status">Loading source…</p> : <><p className="sub">Episode #{id} · {day(source.data.created_at)} · {source.data.kind}</p><div>{source.data.content}</div></>}</div>;
}

function BeliefsView({ api, onChanged, features }: { api: ApiClient; onChanged: () => void; features: Capabilities['features'] }) {
  const facts = useAsync(() => api.request<{ facts: Fact[] }>("/v1/facts?all=true&excluded=true", {signal: AbortSignal.timeout(10000)}), [api]);
  const [query,setQuery]=useState(''),[filter,setFilter]=useState<ClaimFilter>('all'),[page,setPage]=useState(0);
  const [action, setAction] = useState<{ fact: Fact; path: 'close' | 'exclude' | 'include' } | null>(null);
  const [reason, setReason] = useState(''), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [saving, setSaving] = useState(false), [reconcile, setReconcile] = useState(false);
  const request = useRef<AbortController | null>(null), busy = useRef(false);
  useEffect(() => () => { request.current?.abort(); }, [api]);
  useEffect(() => {
    if (!action || !facts.data || saving) return;
    const current = facts.data.facts.find(f => f.fact_id === action.fact.fact_id);
    if (!current || current.status !== action.fact.status || current.excluded_reason !== action.fact.excluded_reason
      || current.valid_until !== action.fact.valid_until || current.object !== action.fact.object) {
      setAction(null); setReason('');
      setNotice('The selected belief changed. Inspect its refreshed state before choosing another action.');
    }
  }, [facts.data, action, saving]);
  const locked = saving || facts.loading || reconcile || Boolean(facts.error);
  const selectAction = (fact: Fact, path: 'close' | 'exclude' | 'include') => {
    if (locked || !features[`facts.${path}`]) return;
    setAction({fact, path}); setReason(''); setError(''); setNotice('');
  };
  const refresh = async () => {
    if (busy.current) return;
    if (await facts.reload()) { setReconcile(false); setError(''); }
  };
  const act = async () => {
    if (!action || !features[`facts.${action.path}`] || locked || busy.current || (action.path !== 'include' && !reason.trim())) return;
    const {fact, path} = action, id = fact.fact_id;
    busy.current = true; setSaving(true); setError(''); setNotice('');
    const controller = new AbortController(); request.current = controller;
    try {
      const receipt = await api.request<Record<string, unknown>>(`/v1/facts/${id}/${path}`, {
        method: 'POST', headers: {'X-Scone-Actor': 'Scone Beliefs'},
        body: path === 'include' ? undefined : JSON.stringify({reason: reason.trim()}),
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]),
      });
      if (controller.signal.aborted) return;
      const confirmed = path === 'close' ? receipt?.closed === id
        : receipt?.fact_id === id && (receipt.status === 'active' || receipt.status === 'closed')
          && (path === 'exclude' ? typeof receipt.excluded_reason === 'string' && Boolean(receipt.excluded_reason.trim()) : receipt.excluded_reason === null);
      if (!confirmed) throw new Error('Server returned an unrecognized decision receipt');
      setNotice(`Belief #${id} ${path === 'close' ? 'closed' : path === 'exclude' ? 'excluded from recall' : 'included in recall'}. History is retained.`);
      setAction(null); setReason(''); void facts.reload(); onChanged();
    } catch (e) {
      if (!controller.signal.aborted) { setError(e instanceof Error ? e.message : String(e)); setReconcile(true); }
    } finally { busy.current = false; if (!controller.signal.aborted) setSaving(false); }
  };
  if (facts.error && !facts.data) return <><ErrorLine message={facts.error} /><button className="btn quiet" onClick={() => void refresh()}>Refresh beliefs</button></>;
  if (!facts.data) return <Empty>Loading…</Empty>;
  const ledger = facts.data.facts.filter((f) => f.status === "active" || f.status === "closed");
  if (!ledger.length) return <Empty>No beliefs yet. Assert one with <code>scone-memory assert</code>, or approve a proposal under Review.</Empty>;
  const groups=claimGroups(ledger,Date.now()),matching=filterClaimGroups(groups,query,filter);
  const currentPage=Math.min(page,Math.max(0,Math.ceil(matching.length/25)-1));
  const bySubject = new Map<string, DisplayClaimGroup[]>();
  for (const group of matching.slice(currentPage*25,(currentPage+1)*25)) {
    bySubject.set(group.subject,[...(bySubject.get(group.subject)||[]),group]);
  }
  const abandonAction=()=>{setAction(null);setReason('');};
  const clearFilters=()=>{setQuery('');setFilter('all');setPage(0);abandonAction();};
  return (
    <>
      <section className="belief-purpose" aria-label="How memory claims work">
        <div><span className="label">For you</span><h3>A record you can inspect and correct</h3><p>These are stored claims, not established facts or a view into an LLM’s thoughts. Check the source, end an outdated claim, or exclude one from recall.</p></div>
        <div><span className="label">For your agent</span><h3>Context it may retrieve—not training</h3><p>When an agent calls Scone, matching claims can be returned for the requested point in time. Storage does not prove an agent retrieved, believed, or used them.</p></div>
        <p className="belief-purpose-foot">Review is for proposed claims. This ledger contains active and closed records; “active” alone does not prove human approval. Excluding a claim does not remove its original source text from search.</p>
      </section>
      <section className="claim-browser" aria-label="Browse memory claims">
        <div className="claim-browser-top"><div><span className="label">Your claim ledger</span><h2>{groups.length} claim histories <small>· {ledger.length} stored versions</small></h2></div><button className="btn small quiet" disabled={saving || facts.loading} onClick={() => void refresh()}>Refresh beliefs</button></div>
        <label className="claim-search">Search claim histories<input type="search" value={query} disabled={saving} placeholder="Subject, claim, quotation or record ID" onChange={event=>{setQuery(event.target.value);setPage(0);abandonAction();}} /></label>
        <div className="claim-filters" role="group" aria-label="Filter claim histories">{([['all','All records'],['current','Current records'],['excluded','Excluded records'],['no-quote','No stored quotation']] as const).map(([value,label])=><button key={value} disabled={saving} aria-pressed={filter===value} onClick={()=>{setFilter(value);setPage(0);abandonAction();}}>{label}</button>)}</div>
        <p className="sub">{matching.length} matching histories · Each card shows a matching version. Open its history for other versions. Current means eligible by recorded dates and exclusion state, not verified true or human-approved.</p>
      </section>
      {notice && <p className="belief-notice" role="status">{notice}</p>}
      {facts.error && <p className="err" role="alert">Could not refresh beliefs: {facts.error}. Displayed records may be stale.</p>}
      {error && <p className="err" role="alert">{error}. No change was confirmed. Refresh beliefs to check the current state before retrying.</p>}
      {reconcile && <p className="sub">Actions are paused. A timed-out request may still have reached the server.</p>}
      {!matching.length&&<div className="claim-no-results"><p>No claim histories match these filters.</p><button className="btn quiet" disabled={saving} onClick={clearFilters}>Clear claim filters</button></div>}
      {[...bySubject.entries()].map(([subject, preds]) => (
        <div key={subject}>
          <div className="subject">{subject}</div>
          {preds.map(({predicate:pred,versions:sorted,current,displayed}) => {
            const displayedCurrent=current?.fact_id===displayed.fact_id;
            return (
              <div key={pred} className="belief">
                <div className="head">
                  <div className="claim-heading"><span className="label">Recorded claim</span><h3>{subject} {pred.replace(/_/g, " ")} {displayed.object}</h3><p className="sub">#{displayed.fact_id} · {displayed.origin === 'extracted' ? 'Model-extracted' : displayed.origin === 'inferred' ? 'Model-inferred' : displayed.origin === 'stated' ? 'Directly stated' : 'Origin not reported'} · {displayedCurrent ? 'Current record · may be returned by recall' : displayed.excluded_reason ? 'Excluded from claim recall' : displayed.status==='closed' ? 'Historical version · not current' : 'Outside its current effective dates'}</p>{current&&!displayedCurrent&&<p className="sub">Current version: #{current.fact_id} · {current.object}</p>}</div>
                  <span className="spacer" />
                  <details className="actions" onClick={event => { if (event.target instanceof Element && event.target.closest('button')) event.currentTarget.open = false; }}><summary>Manage claim</summary>
                    <div className="menu">
                      {features['facts.close'] && sorted.filter((f) => f.status === "active" && !f.excluded_reason).map((f) => <button disabled={locked} key={"c" + f.fact_id} onClick={() => selectAction(f, 'close')}>Close “{f.object}” with a reason</button>)}
                      {features['facts.exclude'] && sorted.filter((f) => !f.excluded_reason).map((f) => <button disabled={locked} key={"e" + f.fact_id} onClick={() => selectAction(f, 'exclude')}>Exclude “{f.object}” from recall</button>)}
                      {features['facts.include'] && sorted.filter((f) => f.excluded_reason).map((f) => <button disabled={locked} key={"i" + f.fact_id} onClick={() => selectAction(f, 'include')}>Include “{f.object}” again</button>)}
                    </div>
                  </details>
                </div>
                {action && action.fact.subject === subject && action.fact.predicate === pred && <form className="belief-action" onSubmit={event => {event.preventDefault(); void act();}} aria-busy={saving}>
                  <h3>{action.path === 'close' ? 'Close this belief' : action.path === 'exclude' ? 'Exclude from recall' : 'Include in recall again'}</h3>
                  <p><b>#{action.fact.fact_id}</b> {action.fact.subject} {action.fact.predicate.replaceAll('_', ' ')} {action.fact.object}</p>
                  <p className="sub">{action.path === 'close' ? 'Ends this belief at the server’s current time. Earlier history is retained; this is not deletion or an undoable review decision.' : action.path === 'exclude' ? 'Hides this belief from recall without deleting its history or source. You can include it again.' : 'Restores recall eligibility. It does not reopen a closed belief or change its historical dates.'}</p>
                  {action.path !== 'include' && <label>Reason for this change<textarea autoFocus required maxLength={500} value={reason} disabled={locked} onChange={event => setReason(event.target.value)} /></label>}
                  <div className="belief-action-buttons"><button className="btn small" disabled={locked || (action.path !== 'include' && !reason.trim())}>{saving ? 'Saving…' : `Confirm ${action.path}`}</button><button type="button" className="btn small quiet" disabled={saving} onClick={() => {setAction(null); setReason('');}}>Cancel</button></div>
                </form>}
                <ClaimSource key={displayed.fact_id} fact={displayed} api={api} />
                {displayed.excluded_reason && <p className="note">{displayed.object}: excluded from recall, {displayed.excluded_reason}</p>}
                <details className="claim-history"><summary>Record history · {sorted.length} {sorted.length === 1 ? 'version' : 'versions'}</summary>
                  <ol>{sorted.map(f => <li key={f.fact_id}><b>#{f.fact_id} · {f.object}</b><p>{day(f.valid_from)} → {f.valid_until ? day(f.valid_until) : 'No end date recorded'} · {f.status}{f.excluded_reason ? ' · Excluded' : ''}</p>{f.closed_reason && <p>Closure: {f.closed_reason}</p>}{f.excluded_reason && <p>Exclusion: {f.excluded_reason}</p>}{f.fact_id !== displayed.fact_id && <ClaimSource fact={f} api={api} />}</li>)}</ol>
                </details>
              </div>
            );
          })}
        </div>
      ))}
      {matching.length>25&&<nav className="claim-pagination" aria-label="Claim pages"><button className="btn quiet" disabled={saving||currentPage===0} onClick={()=>{setPage(currentPage-1);abandonAction();}}>Previous</button><span>{currentPage*25+1}–{Math.min((currentPage+1)*25,matching.length)} of {matching.length} histories</span><button className="btn quiet" disabled={saving||(currentPage+1)*25>=matching.length} onClick={()=>{setPage(currentPage+1);abandonAction();}}>Next</button></nav>}
    </>
  );
}


/* ---------- live ---------- */

const KINDS = ["recall", "remember", "fact_assert", "fact_review", "fact_exclude", "fact_close", "forget", "feedback", "distill", "job", "agent"];

function summarise(e: EventRecord): string {
  const p = e.payload as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  switch (e.kind) {
    case "recall": { const top = (p.items ?? [])[0]; const lanes = top?.lanes ? Object.keys(top.lanes).join(" and ") : "no";
      return `recall ${p.query_hashed ? "(query hashed)" : p.query}: ${(p.items ?? []).length} excerpt(s), top via ${lanes} lane${p.latency_ms?.total != null ? `, ${Number(p.latency_ms.total).toFixed(1)} ms` : ""}${p.degraded?.length ? `, degraded: ${p.degraded.join("; ")}` : ""}`; }
    case "remember": return `stored ${p.fresh ?? "?"} new, ${p.deduplicated ?? 0} already known, ${p.chunks ?? 0} chunk(s), ${p.bytes ?? 0} B${p.latency_ms != null ? `, ${Number(p.latency_ms).toFixed(1)} ms` : ""}`;
    case "fact_assert": return `belief ${p.subject} ${String(p.predicate ?? "").replace(/_/g, " ")}: ${p.outcome}${p.origin && p.origin !== "stated" ? ` (${p.origin})` : ""}${p.superseded?.length ? `, superseded #${p.superseded.join(", #")}` : ""}`;
    case "fact_close": return `belief #${p.fact_id} closed by a person`;
    case "fact_review": return `proposal #${p.fact_id} ${p.decision}${p.of ? ` of #${p.of}` : ""}`;
    case "fact_exclude": return `belief #${p.fact_id} ${p.action === "exclude" ? "excluded from recall" : "included again"}`;
    case "forget": return `episode #${p.episode_id} forgotten, ${p.chunks_removed ?? 0} chunk(s) removed`;
    case "feedback": return `recall #${p.recall_event_id}, chunk #${p.chunk_id} marked ${p.useful ? "useful" : "not useful"}`;
    case "distill": return `consolidation pass: ${p.episodes ?? 0} read, ${p.proposed ?? 0} proposed, ${p.accepted ?? 0} accepted, ${p.parked ?? 0} parked${p.error ? `, ${p.error}` : ""}`;
    case "agent": return `${p.agent} ${String(p.event).replace(/_/g, " ")}${p.tool_name ? ` ${p.tool_name}` : ""}${p.text ? `: ${String(p.text).slice(0, 140)}` : ""}${p.text_truncated ? " (truncated)" : ""}`;
    default: return JSON.stringify(p).slice(0, 160);
  }
}
const stateOf = (e: EventRecord): ["ok" | "bad" | "run", string] => {
  const p = e.payload as Record<string, unknown>;
  if (e.kind === "job") return p.status === "running" ? ["run", "running"] : p.status === "failed" ? ["bad", "failed"] : ["ok", "completed"];
  return p.error ? ["bad", "failed"] : ["ok", "completed"];
};

function LiveView({ api }: { api: ApiClient }) {
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [meta, setMeta] = useState<{ evidence?: string; queries?: string; error?: string; updated?: number }>({});
  const [paused, setPaused] = useState(false);
  const [kind, setKind] = useState("");
  const cursor = useRef<number | null>(null);
  const inFlight = useRef(false);
  const [, tick] = useState(0);

  const poll = useCallback(async () => {
    if (paused || inFlight.current) return;
    inFlight.current = true;
    try {
      // First read: newest 80. Afterwards: a forward cursor, so a burst is never dropped.
      const q = cursor.current == null ? `limit=80${kind ? `&kind=${kind}` : ""}` : `after_id=${cursor.current}&limit=200${kind ? `&kind=${kind}` : ""}`;
      const r = await api.request<EventsResponse>(`/v1/events?${q}`);
      setEvents((prev) => {
        const incoming = cursor.current == null ? r.events : [...r.events].reverse();
        const merged = cursor.current == null ? incoming : [...incoming, ...prev];
        return merged.slice(0, 300);
      });
      const newest = r.events.reduce((m, e) => Math.max(m, e.event_id), cursor.current ?? 0);
      cursor.current = r.next_after_id ?? newest;
      setMeta({ evidence: r.evidence, queries: r.queries_recorded, updated: Date.now() });
    } catch (e) {
      setMeta((m) => ({ ...m, error: e instanceof Error ? e.message : String(e) }));
    } finally {
      inFlight.current = false;
    }
  }, [api, paused, kind]);

  useEffect(() => { cursor.current = null; setEvents([]); poll(); const id = window.setInterval(poll, 3000); const t = window.setInterval(() => tick((n) => n + 1), 1000); return () => { window.clearInterval(id); window.clearInterval(t); }; }, [poll]);

  const jobs = new Map<string, EventRecord>();
  const rows: EventRecord[] = [];
  for (const e of events) {
    if (e.kind === "job") { const id = String((e.payload as Record<string, unknown>).job_id); if (!jobs.has(id)) jobs.set(id, e); continue; }
    rows.push(e);
  }
  let minute = "";
  return (
    <>
      <div className="statusline">
        <span><span className={"dot " + (meta.error ? "off" : "")} />{meta.error ? `disconnected, ${meta.error}` : "connected"}</span>
        <span>polling every 3 s{meta.updated ? `, updated ${Math.round((Date.now() - meta.updated) / 1000)} s ago` : ""}</span>
        <button className="btn small quiet" onClick={() => setPaused((p) => !p)}>{paused ? "Resume" : "Pause"}</button>
        <select value={kind} onChange={(e) => setKind(e.target.value)}><option value="">all kinds</option>{KINDS.map((k) => <option key={k} value={k}>{k}</option>)}</select>
        {meta.evidence && <span>evidence in {meta.evidence}, queries {meta.queries === "hash" ? "hashed" : "recorded"}</span>}
      </div>
      {!events.length && !meta.error && <Empty>No activity recorded yet. Recall or store something and it appears here within a few seconds.</Empty>}
      {jobs.size > 0 && (
        <div className="jobs"><h3>Jobs</h3>
          {[...jobs.values()].map((e) => { const p = e.payload as Record<string, any>; const [cls, label] = stateOf(e); // eslint-disable-line @typescript-eslint/no-explicit-any
            return (
              <div key={e.event_id} className="job">
                <span className="name">{p.name} <span className={"state " + cls}>{label}</span></span>
                <span className="prog">{p.progress ? `${p.progress.done} / ${p.progress.total}` : ""}</span>
                <span className="sub">{p.job_id}{p.adapter ? `, ${p.adapter}` : ""}{p.product ? `, ${p.product}` : ""}, last report {clock(e.ts)}{p.detail ? ` · ${p.detail}` : ""}{p.error ? <> · <span className="err">{p.error}</span></> : null}</span>
              </div>
            ); })}
        </div>
      )}
      {rows.map((e) => {
        const m = e.ts.slice(0, 16);
        const header = m !== minute ? (minute = m, <div key={"m" + e.event_id} className="minute">{day(e.ts)} {e.ts.slice(11, 16)}</div>) : null;
        const [cls, label] = stateOf(e);
        return (
          <div key={e.event_id}>
            {header}
            <div className="ev"><span className="t">{clock(e.ts)}</span><span className="k">{e.kind.replace(/_/g, " ")}</span>
              <span className="d">{summarise(e)}{cls === "bad" && <span className="state bad">{label}{(e.payload as Record<string, unknown>).error ? `: ${String((e.payload as Record<string, unknown>).error)}` : ""}</span>}</span></div>
          </div>
        );
      })}
    </>
  );
}

/* ---------- analytics ---------- */

const fmt = (m: Metric) => {
  if (m.value == null) return <span className="none">no evidence yet</span>;
  if (typeof m.value === "number") return m.unit === "share" || m.unit === "share of bytes" ? `${(m.value * 100).toFixed(1)}%` : m.unit === "ms" ? `${m.value.toFixed(1)} ms` : String(m.value);
  return Object.entries(m.value).map(([k, v]) => `${k} ${typeof v === "number" ? v.toFixed(3) : v}`).join(" · ");
};
const label = (name: string) => name.split(".").slice(1).join(" ").replace(/_/g, " ").replace("latency ms ", "latency ").replace("top item ", "top excerpt via ").replace(" share", "").replace("top similarity ", "top similarity, ");

function Fig({ m }: { m?: Metric }) {
  if (!m) return null;
  return (
    <div className="fig" title={m.definition + (m.caveat ? "\n" + m.caveat : "")}>
      <span className="l">{label(m.name)}</span><span className="v">{fmt(m)}</span><span className="n">n = {m.n} {m.denominator}</span>
    </div>
  );
}

function AnalyticsView({ api }: { api: ApiClient }) {
  const r = useAsync(() => api.request<MetricsReport>("/v1/metrics"), [api]);
  if (r.error) return <ErrorLine message={r.error} />;
  if (!r.data) return <Empty>Loading…</Empty>;
  const cov = r.data.coverage;
  if (!cov) return <Empty>No event log is attached to this server, so nothing can be computed.</Empty>;
  if (!cov.events_considered) return <Empty>No activity recorded yet. Once memory is used, its figures appear here.</Empty>;
  const by = Object.fromEntries((r.data.metrics ?? []).map((m) => [m.name, m]));
  const panel = (title: string, sub: string, names: string[]) => (
    <div className="panel"><h3>{title}</h3><div className="sub">{sub}</div>{names.map((n) => <Fig key={n} m={by[n]} />)}</div>
  );
  const days = Object.entries(r.data.daily ?? {});
  const max = Math.max(1, ...days.map(([, k]) => Object.values(k).reduce((a, b) => a + b, 0)));
  return (
    <>
      <div className="grid">
        {panel("Recall", "use and timing", ["recall.count", "recall.latency_ms.p50", "recall.latency_ms.p95", "recall.degraded_count", "recall.byte_context_reduction.median"])}
        {panel("Ingest", "what was stored", ["ingest.calls", "ingest.fresh_episodes", "ingest.deduplicated", "ingest.dedup_fraction", "ingest.stored_bytes"])}
        {panel("Beliefs", "ledger events, not accuracy", ["ledger.asserted", "ledger.asserted_already_closed", "ledger.restated", "ledger.superseded", "ledger.manual_closures"])}
        {panel("Feedback", "what people judged", ["feedback.judged_items", "feedback.useful_share", "feedback.coverage"])}
        {panel("Lanes", "retrievers agreeing, not relevance", ["recall.top_item.both_lanes_share", "recall.top_item.vector_only_share", "recall.top_item.text_only_share", ...Object.keys(by).filter((k) => k.startsWith("recall.top_similarity."))])}
        {panel("Steps", "median latency by step", ["recall.latency_ms.embed.p50", "recall.latency_ms.vector.p50", "recall.latency_ms.text.p50"])}
      </div>
      <div className="foot">
        <span><b>{cov.events_considered}</b> events considered</span><span>from <b>{(cov.earliest_retained ?? "").slice(0, 16)}</b></span>
        <span>to <b>{(cov.latest ?? "").slice(0, 16)}</b></span><span>{cov.truncated ? <span className="err">read truncated, coverage partial</span> : "read complete"}</span>
        <span>failures: <b>{Object.entries(r.data.failures ?? {}).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}</b></span>
        <span>embedders: <b>{(r.data.embedders ?? []).join(", ") || "none"}</b></span>
      </div>
      {days.length > 0 && (
        <>
          <div className="spark">{days.map(([d, k]) => { const t = Object.values(k).reduce((a, b) => a + b, 0); return <i key={d} style={{ height: `${Math.round((t / max) * 100)}%` }} title={`${d}: ${t} event(s)`} />; })}</div>
          <div className="legend"><span>{days[0][0]}</span><span>events per day</span><span>{days[days.length - 1][0]}</span></div>
        </>
      )}
    </>
  );
}

/* ---------- scopes, status ---------- */

function ScopesView({ api, onScope }: { api: ApiClient; onScope: (k: string, v: string) => void }) {
  const r = useAsync(() => api.request<Scopes>("/v1/scopes"), [api]);
  if (r.error) return <ErrorLine message={r.error} />;
  if (!r.data) return <Empty>Loading…</Empty>;
  const keys = Object.entries(r.data.scopes ?? {});
  if (!keys.length) return <Empty>No scoped memory yet. Store episodes with metadata such as user_id or agent_id and they are counted here.</Empty>;
  return (
    <>
      {keys.map(([key, values]) => { const max = Math.max(...Object.values(values));
        return (
          <div key={key} className="panel" style={{ marginBottom: 16 }}>
            <h3>{key.replace(/_/g, " ")}</h3><div className="sub">{Object.keys(values).length} value(s), choose one to search within it</div>
            <div className="kv">
              {Object.entries(values).map(([v, n]) => (
                <React.Fragment key={v}>
                  <button className="linkish" style={{ textAlign: "left" }} onClick={() => onScope(key, v)}>{v}</button>
                  <div className="bar"><i style={{ width: `${Math.round((n / max) * 100)}%` }} /></div>
                  <span className="sub">{n}</span>
                </React.Fragment>
              ))}
            </div>
          </div>
        ); })}
    </>
  );
}

function StatusView({ api }: { api: ApiClient }) {
  const r = useAsync(() => api.request<Status>("/v1/status"), [api]);
  if (r.error) return <ErrorLine message={r.error} />;
  if (!r.data) return <Empty>Loading…</Empty>;
  const s = r.data;
  const rows: Array<[string, string | number | undefined]> = [
    ["Space", s.space], ["Episodes", s.episodes], ["Chunks", s.chunks], ["Stored bytes", s.bytes], ["Awaiting review", s.pending_review],
    ["Awaiting distillation", s.pending_distill], ["Revision", s.revision],
    ["Claims", s.semantic_lane === "manual" ? "asserted by people and agents; no distiller configured on this server" : s.semantic_lane === "active" ? "a consolidation worker proposes claims from new episodes" : s.semantic_lane],
  ];
  return (
    <>
      {s.embedder && (
        <div className="flow">
          <span className="node">episodes<small>{s.episodes} stored, {s.chunks} chunks</small></span><span className="arrow">→</span>
          <span className="node">{s.embedder}<small>embedder</small></span><span className="arrow">→</span>
          <span className="node">{s.vector_index}<small>vector index</small></span><span className="arrow">+</span>
          <span className="node">{s.document_store}<small>documents, facts, text lane</small></span>
        </div>
      )}
      <div className="panel">{rows.filter(([, v]) => v != null).map(([k, v]) => <div key={k} className="stat"><span>{k}</span><span>{v}</span></div>)}</div>
    </>
  );
}
