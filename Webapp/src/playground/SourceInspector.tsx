import type { EvidenceGraph, EvidenceNode } from '../types';
import type { ApiClient } from '../api';
import { SourceImages } from '../components/SourceImages';
import { SourceContent } from '../components/SourceContent';
import { owners } from './graph-layout';

export function SourceInspector({ node, graph, api, select, close }: { node?: EvidenceNode; graph: EvidenceGraph; api: ApiClient; select: (id: string) => void; close?: () => void }) {
  const data = node?.data || {};
  const content = data.content ?? data.text;
  const related = node ? graph.edges.filter(e => e.source === node.id || e.target === node.id) : [];
  // Episode node IDs are the graph's source identity. A chunk, turn or claim ID
  // must never be sent to the episode API as though it identified an original.
  const episodeId = node?.kind === 'episode' ? /^episode:(\d+)$/.exec(node.id)?.[1] : undefined;
  const sourceId=node?.kind==='claim'&&Number.isSafeInteger(data.source_episode_id)&&Number(data.source_episode_id)>0?Number(data.source_episode_id):null;
  const sourceOmitted=sourceId!==null&&!graph.nodes.some(n=>n.id===`episode:${sourceId}`);
  const sessionPath=node?owners(graph.nodes,graph.edges).has(node.id):false;
  return <aside className="inspector" id="inspector" aria-label="Source inspector">
    {close && <button className="inspector-close" aria-label="Close source inspector" onClick={close}>×</button>}
    <div className="eyebrow">{node ? `${node.kind.replaceAll('_', ' ')} · recorded evidence` : 'Source inspector'}</div>
    <h2>{node?.label || 'Inspect the evidence'}</h2>
    {!node && <p className="intro">Select a record to see its source, contents and recorded connections. A relationship is never inferred from visual proximity.</p>}
    {node && <>
      {content != null ? <div className="source-content">{typeof content === 'string' ? <SourceContent text={content}/> : <pre>{JSON.stringify(content, null, 2)}</pre>}</div> : <p className="intro">This record has no captured text.</p>}
      {data.text_truncated === true && <p className="note">The connector truncated this text. The full original is not present in this event.</p>}
      {episodeId && <SourceImages key={node.id} episodeId={episodeId} api={api} />}
      <div className="field"><span>Record ID</span><div className="mono">{node.id}</div></div>
      {node.ts && <div className="field"><span>Recorded time</span>{node.ts}</div>}
      {['origin', 'status', 'source', 'provenance', 'valid_from', 'valid_until'].filter(k => data[k] != null).map(k => <div className="field" key={k}><span>{k.replaceAll('_', ' ')}</span>{typeof data[k] === 'object' ? JSON.stringify(data[k]) : String(data[k])}</div>)}
      <div className="eyebrow relation-heading">Recorded connections · {related.length}</div>
      {sourceOmitted&&<p className="provenance-gap">Episode #{sourceId} is referenced by this claim but is not included in this snapshot.</p>}
      {!sessionPath&&<p className="note">No recorded session path is available in this snapshot. This does not mean the record has no source; its supporting records may be outside the snapshot or may lack captured session attribution.</p>}
      {related.map((edge, i) => { const outgoing=edge.source===node.id,id=outgoing?edge.target:edge.source; return <button className="relation" key={`${id}:${i}`} onClick={() => select(id)}><span>{outgoing?'This record →':'This record ←'} {id}</span><small className="muted">{edge.source} → {edge.kind.replaceAll('_', ' ')} → {edge.target}</small></button>; })}
      <p className="note">{['session', 'turn', 'tool_call'].includes(node.kind) ? 'Host activity is connector-reported. Capture does not establish truth.' : node.kind === 'recall' ? 'Returned evidence does not establish answer correctness.' : 'Only stored relationships are shown. Missing provenance is not inferred.'}</p>
      <details><summary>Inspect recorded fields</summary><pre>{JSON.stringify(data, null, 2)}</pre></details>
    </>}
  </aside>;
}
