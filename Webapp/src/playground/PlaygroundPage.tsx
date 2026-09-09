import { useMemo, useState } from 'react';
import { WorkspaceHeading } from '../components/WorkspaceHeading';
import type { ApiClient } from '../api';
import type { EvidenceGraph } from '../types';
import { useEvidence } from './useEvidence';
import { GraphCanvas } from './GraphCanvas';
import type { GraphLayout } from './LayoutPicker';
import { SourceInspector } from './SourceInspector';
import { Modal } from '../components/Modal';
import { DevReload } from '../components/DevReload';
import { RecallPanel } from './RecallPanel';

function sessionNodes(graph: EvidenceGraph, session: string | null) {
  if (!session) return graph.nodes;
  const keep = new Set([session]);
  let changed = true;
  while (changed) { changed = false; for (const edge of graph.edges) {
    if (keep.has(edge.source) && !keep.has(edge.target)) { keep.add(edge.target); changed = true; }
    if (['returned', 'held', 'judged'].includes(edge.kind) && keep.has(edge.target) && !keep.has(edge.source)) { keep.add(edge.source); changed = true; }
  } }
  return graph.nodes.filter(node => keep.has(node.id));
}

export function PlaygroundPage({ api, enabled }: { api: ApiClient; enabled: boolean }) {
  const [paused, setPaused] = useState(false), [revision, setRevision] = useState(0);
  const overview = useEvidence(api, enabled, paused, revision);
  const [session, setSession] = useState<string | null>(null), [selected, setSelected] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>();
  const focused = useEvidence(api, enabled && Boolean(sessionId), paused, revision, sessionId);
  const { graph, error, updated } = sessionId ? focused : overview;
  const [depth, setDepth] = useState(false), [records, setRecords] = useState(false), [query, setQuery] = useState('');
  const [layout,setLayout]=useState<GraphLayout>('constellation');
  const [connections, setConnections] = useState(false);
  const visible = useMemo(() => sessionNodes(graph, session), [graph, session]);
  const ids = new Set(visible.map(node => node.id));
  const edges = graph.edges.filter(edge => ids.has(edge.source) && ids.has(edge.target));
  const sessions = overview.graph.nodes.filter(node => node.kind === 'session');
  const found = visible.filter(node => [node.id, node.label, node.kind, JSON.stringify(node.data || {})].join(' ').toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  const events = visible.filter(node => node.ts && ['turn', 'tool_call', 'recall', 'feedback'].includes(node.kind)).sort((a, b) => String(b.ts).localeCompare(String(a.ts))).slice(0, 50);
  const inspected = graph.nodes.find(node => node.id === selected);
  const omitted=typeof graph.provenance_omitted==='number'&&Number.isSafeInteger(graph.provenance_omitted)&&graph.provenance_omitted>0?graph.provenance_omitted:0;
  const coverage = graph.fact_read_status === 'failed' || graph.fact_read_status === 'unavailable'
    ? 'Partial snapshot · claims unavailable'
    : graph.facts_truncated ? 'Partial snapshot · some claims outside this view'
    : graph.truncated ? 'Partial snapshot · records omitted' : '';
  return <div className="workspace">
    <aside className="rail" aria-label="Sessions"><div className="rail-head"><h2>Session trace</h2><p className="muted">Interactions and the evidence they leave behind.</p></div><section><div className="rail-label"><span className="eyebrow">Recorded sessions</span><span>{sessions.length}</span></div><div id="sessions">
      <button className={`session ${!session ? 'selected' : ''}`} aria-pressed={!session} onClick={() => { setSession(null); setSessionId(undefined); setSelected(null); }}><b>All recorded activity</b><small>{overview.graph.nodes.length} records in this snapshot</small></button>
      {sessions.map(node => <button key={node.id} className={`session ${session === node.id ? 'selected' : ''}`} aria-pressed={session === node.id} onClick={() => { setSession(node.id); setSessionId(typeof node.data?.session_id === 'string' ? node.data.session_id : undefined); setSelected(null); }}><b>{node.data?.agent === 'claude-code' ? 'Claude Code' : node.data?.agent === 'codex' ? 'Codex' : node.label}</b><small>{String(node.data?.project || 'Session')} · {String(node.data?.session_id || node.id).slice(0,8)}</small></button>)}
    </div></section><div className="rail-footer"><span className="eyebrow">Evidence, not inference</span><p>A captured conversation is a source, not an approved belief. Connections require stored references.</p></div></aside>
    <main className="main" id="main"><WorkspaceHeading className="heading" eyebrow="Explore your knowledge" title="Playground" description="Trace an interaction. Inspect its memory. Test its recall." actions={<div className="heading-controls"><span className="pill"><span className={`dot ${updated && !error ? 'online' : ''}`} /><span id="connection" role="status">{!enabled ? 'Connection needs authentication' : error || (updated ? `API online · ${sessions.length ? 'activity recorded' : 'No agent events'}` : 'Connecting to Scone API…')}</span></span><button onClick={() => setConnections(true)}>Connect agents</button><button aria-label={paused ? 'Resume live updates' : 'Pause live updates'} onClick={() => setPaused(value => !value)}>{paused ? '▶ Resume' : 'Ⅱ Pause'}</button></div>}/>
      <div className="record-toolbar"><label><span className="eyebrow">Find in this snapshot</span><input type="search" aria-label="Find a record" placeholder="Search content, source or record ID" value={query} onChange={event => { setQuery(event.target.value); setRecords(true); }} /></label><div className="view-switch"><button aria-label="Graph view" aria-pressed={!records} onClick={() => setRecords(false)}>Graph</button><button aria-label="Records view" aria-pressed={records} onClick={() => setRecords(true)}>Records</button></div></div>
      <section className={`stage evidence-atlas ${inspected ? 'has-inspector' : 'map-only'}`} aria-label="Evidence explorer"><div className={`graph-panel ${layout}-theme ${records ? 'records-mode' : ''}`}>
        <div className="graph-top"><span className="graph-title">{visible.length} records · {edges.length} links</span><span id="coverage">{error && graph.nodes.length ? 'Stale snapshot · not live' : <>{coverage}{omitted>0&&<span className="provenance-count">{omitted} source episodes outside snapshot</span>}</>}</span>{layout!=='growth'&&<div className="view-switch"><button aria-label="2D view" aria-pressed={!depth} onClick={() => setDepth(false)}>2D</button><button aria-label="Depth view" aria-pressed={depth} onClick={() => setDepth(true)}>Depth</button></div>}</div>
        {depth && layout!=='growth'&&<div id="layout-note">3D perspective · spatial depth is not confidence</div>}
        <GraphCanvas key={session || 'all'} nodes={visible} edges={edges} selected={selected} select={setSelected} depth={depth} layout={layout} setLayout={setLayout} />
        {records && <div id="record-list" aria-label="Recorded evidence">{found.length ? found.map(node => <button className="record-row" data-record={node.id} key={node.id} onClick={() => setSelected(node.id)}><span className="record-kind">{node.kind.replaceAll('_', ' ')}</span><span><b>{node.label}</b><small>{String(node.data?.text ?? node.data?.content ?? node.id).slice(0, 180)}</small></span><span aria-hidden>↗</span></button>) : <div className="record-empty"><h2>No matching records</h2><p>Try another phrase or clear the search. Only this bounded snapshot is searched.</p></div>}</div>}
      </div>{inspected && <SourceInspector key={inspected.id} node={inspected} graph={graph} api={api} select={setSelected} close={() => setSelected(null)} />}</section>
      <section className="lower"><div className="timeline-panel"><div className="section-title"><h2>Activity timeline</h2><small>{paused ? 'Paused · snapshot retained' : updated ? 'Refresh every 1s · ' + updated : 'Waiting for evidence'}</small></div><div id="timeline">{events.length ? events.map(node => <button className="timeline-row" key={node.id} onClick={() => setSelected(node.id)}><time>{new Date(node.ts!).toLocaleTimeString([], { hour12: false })}</time><span className="event-kind">{String(node.data?.event || node.kind)}</span><span className="event-label">{String(node.data?.text || node.label)}</span></button>) : <p className="empty-row">No observed activity in this view.</p>}</div></div>
        <RecallPanel api={api} enabled={enabled && !error} onRecalled={() => setRevision(value=>value+1)} />
      </section><footer className="coverage-bar"><span>Source <b>authenticated Scone API</b></span><span>Capture <b>{sessions.length ? 'activity recorded; liveness unverified' : 'no agent events'}</b></span><DevReload /><span>Bounded snapshot · not a complete replay</span></footer>
    </main>
    {connections && <Modal title="Agent connections" onClose={() => setConnections(false)}><p className="setup-intro">Bring visible prompts, completed replies and supported tool events into this space. This page observes records; it does not control your agents.</p>{[['claude-code', 'Claude Code · Fable'], ['codex', 'Codex']].map(([agent, label]) => { const count = sessions.filter(node => node.data?.agent === agent).length; return <div className="agent-row" key={agent}><h3>{label}</h3><output>{count ? `${count} recorded sessions in this snapshot · live capture not verified` : 'No activity received'}</output><p>{agent === 'codex' ? 'Supported CLI hooks require a one-time trust review. Capture from an already-running Codex App session is unverified.' : 'Capture requires a scoped observer hook. The automatic prompt compiler is separate: compiling a prompt does not send it to this graph.'}</p></div>; })}<ol className="setup-steps"><li>Select the project and session allowed to send records. Use a key bound to this memory space.</li><li>Configure the Scone agent-hook observer with that allowlist. Metadata and text capture are separate choices.</li><li>Send a prompt in the host, then verify its receipt and source here. API health alone never proves that capture is working.</li></ol><p className="setup-note">No hidden reasoning is captured. Completed replies are not a token stream. Missing events remain missing.</p></Modal>}
  </div>;
}
