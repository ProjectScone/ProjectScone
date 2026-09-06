import type { EvidenceGraph, EvidenceNode } from '../types';
import type { ApiClient } from '../api';
import { SourceImages } from '../components/SourceImages';

export function SourceInspector({ node, graph, api, select, close }: { node?: EvidenceNode; graph: EvidenceGraph; api: ApiClient; select: (id: string) => void; close?: () => void }) {
  const data = node?.data || {};
  const content = data.content ?? data.text;
  const related = node ? graph.edges.filter(e => e.source === node.id || e.target === node.id) : [];
  // Episode node IDs are the graph's source identity. A chunk, turn or claim ID
  // must never be sent to the episode API as though it identified an original.
  const episodeId = node?.kind === 'episode' ? /^episode:(\d+)$/.exec(node.id)?.[1] : undefined;
  return <aside className="inspector" id="inspector" aria-label="Source inspector">
    {close && <button className="inspector-close" aria-label="Close source inspector" onClick={close}>×</button>}
    <div className="eyebrow">{node ? `${node.kind.replaceAll('_', ' ')} · recorded evidence` : 'Source inspector'}</div>
    <h2>{node?.label || 'Inspect the evidence'}</h2>
    {!node && <p className="intro">Select a record to see its source, contents and recorded connections. A relationship is never inferred from visual proximity.</p>}
    {node && <>
      {content != null ? <div className="source-content">{typeof content === 'string' ? content : JSON.stringify(content, null, 2)}</div> : <p className="intro">This record has no captured text.</p>}
      {data.text_truncated === true && <p className="note">The connector truncated this text. The full original is not present in this event.</p>}
      {episodeId && <SourceImages key={node.id} episodeId={episodeId} api={api} />}
      <div className="field"><span>Record ID</span><div className="mono">{node.id}</div></div>
      {node.ts && <div className="field"><span>Recorded time</span>{node.ts}</div>}
      {['origin', 'status', 'source', 'provenance', 'valid_from', 'valid_until'].filter(k => data[k] != null).map(k => <div className="field" key={k}><span>{k.replaceAll('_', ' ')}</span>{typeof data[k] === 'object' ? JSON.stringify(data[k]) : String(data[k])}</div>)}
      <div className="eyebrow relation-heading">Recorded connections · {related.length}</div>
      {related.map((edge, i) => { const id = edge.source === node.id ? edge.target : edge.source; return <button className="relation" key={`${id}:${i}`} onClick={() => select(id)}>{edge.kind.replaceAll('_', ' ')} <span className="muted">{id}</span></button>; })}
      <p className="note">{['session', 'turn', 'tool_call'].includes(node.kind) ? 'Host activity is connector-reported. Capture does not establish truth.' : node.kind === 'recall' ? 'Returned evidence does not establish answer correctness.' : 'Only stored relationships are shown. Missing provenance is not inferred.'}</p>
      <details><summary>Inspect recorded fields</summary><pre>{JSON.stringify(data, null, 2)}</pre></details>
    </>}
  </aside>;
}
