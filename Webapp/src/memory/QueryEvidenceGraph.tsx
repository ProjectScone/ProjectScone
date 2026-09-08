import {useId,useMemo,useState} from 'react';
import type {ApiClient} from '../api';
import {EvidenceExplorer} from './EvidenceExplorer';
import {parseQueryEvidence,type QueryEvidence} from './query-evidence';
import './query-evidence.css';

export function QueryEvidenceGraph({value,api,title='Query evidence map',context='query'}:{value:unknown;api:ApiClient;title?:string;context?:'query'|'reply'}){
  const heading=useId(),parsed=useMemo(()=>parseQueryEvidence(value),[value]);
  const fingerprint=JSON.stringify(parsed);
  // Identical receipt polling preserves an open explorer; changed evidence revokes it.
  const graph=useMemo(()=>parsed,[fingerprint]);
  const [opened,setOpened]=useState<{api:ApiClient;graph:QueryEvidence}|null>(null);
  if(!graph)return <p className="qe-unavailable">{value==null?'This response did not include an evidence map.':'The evidence map could not be validated. Inspect the returned records directly.'}</p>;
  const concepts=graph.nodes.filter(node=>node.kind==='concept').length,passages=graph.nodes.filter(node=>node.kind==='chunk').length;
  const relations=graph.edges.filter(edge=>edge.kind==='relation'||['extends','derived_from','contradicts','supports','superseded_by'].includes(edge.kind)).length;
  return <section className="query-evidence-summary" aria-labelledby={heading} data-context={context}>
    <div className="qe-summary-title"><svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true"><path d="M8 10L25 7L27 26L10 28L8 10L27 26M25 7L10 28" fill="none" stroke="#a7b8d1"/><circle cx="8" cy="10" r="4" fill="#3979ca"/><circle cx="25" cy="7" r="3" fill="#8d65b7"/><circle cx="27" cy="26" r="4" fill="#26917d"/><circle cx="10" cy="28" r="3" fill="#c5772d"/></svg><div><span className="qe-eyebrow">Follow the evidence</span><h3 id={heading}>{title}</h3></div></div>
    <p className="qe-summary-counts">{concepts>0?`${concepts} concepts · `:''}{passages} passage{passages===1?'':'s'} · {relations} recorded relation{relations===1?'':'s'}{graph.truncated?' · partial view':''}</p>
    <p className="qe-summary-caption">{context==='reply'?'Explore the context supplied to this reply. Not a model reasoning trace.':'Explore the node-link graph, inspect communities, and trace records to their sources.'}</p>
    <button type="button" className="qe-open-explorer" onClick={()=>setOpened({api,graph})}>Open evidence explorer <span aria-hidden="true">↗</span></button>
    {opened?.api===api&&opened.graph===graph&&<EvidenceExplorer graph={graph} api={api} title={title} context={context} close={()=>setOpened(null)}/>}
  </section>;
}
