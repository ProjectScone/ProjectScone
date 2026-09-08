import {useLayoutEffect,useMemo,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import type {ApiClient} from '../api';
import {SourcePageLink} from './SourcePageLink';
import {EvidenceNetworkCanvas,GROUP_COLORS} from './EvidenceNetworkCanvas';
import {detectEvidenceGroups,projectEvidenceNetwork,type EvidenceView} from './evidence-network';
import {connectionsFor,edgeCategory,evidenceEdgeLabel,evidenceSourceId,type QueryEvidence,type QueryEvidenceEdge,type QueryEvidenceNode} from './query-evidence';

const text=(data:Record<string,unknown>,key:string)=>typeof data[key]==='string'?data[key]:null;
const metric=(data:Record<string,unknown>,key:string)=>typeof data[key]==='number'&&Number.isFinite(data[key])?data[key].toFixed(3):null;
const kindName={query:'Query',chunk:'Passage',episode:'Source',claim:'Claim',concept:'Concept'};

function SourceEvidence({item,api,fallback}:{item:QueryEvidenceNode|QueryEvidenceEdge;api:ApiClient;fallback?:QueryEvidenceNode}){
  const source=evidenceSourceId(item)??(fallback?evidenceSourceId(fallback):null),quote=text(item.data,'quote'),status=text(item.data,'provenance_status');
  return <div className="en-source-evidence">
    {quote&&<><span className="en-kicker">{item.kind==='mentions'?'Exact retained mention':'Retained quotation · excerpt'}</span><blockquote>{quote}</blockquote></>}
    {!quote&&('fact_id' in item.data||item.kind==='source_of')&&<p className="en-caption">{status==='missing'?'The source is no longer retained.':status==='out_of_scope'?'Source evidence is outside this query’s scope.':status==='omitted'?'Source evidence was omitted from this bounded view.':'No retained quotation was returned for this record.'}</p>}
    {source!==null&&<SourcePageLink api={api} episodeId={source}/>}
  </div>;
}
function NodeDetail({node,api}:{node:QueryEvidenceNode;api:ApiClient}){
  const passage=text(node.data,'text')??text(node.data,'preview'),source=text(node.data,'source'),ranking=metric(node.data,'score'),similarity=metric(node.data,'similarity');
  return <><span className="en-kicker">{kindName[node.kind]}</span><h3>{node.label}</h3><code className="en-record-id">{node.id}</code>
    {node.kind==='concept'&&<p className="en-caption">{node.data.basis==='literal_mention'?'This name-like text occurs literally in retained passages. It is not a verified entity or a factual relationship.':'A named subject or object from recorded claims. Follow its relationships and source evidence to assess what is supported.'}</p>}
    {node.kind==='claim'&&<div className="en-record-tags">{['origin','status'].map(key=>text(node.data,key)&&<span key={key}>{key}: {text(node.data,key)}</span>)}</div>}
    {passage&&<><span className="en-kicker">Retained excerpt</span><p className="en-passage">{passage}</p></>}
    {source&&<p className="en-caption">Source: {source}</p>}{node.ts&&<p className="en-caption">Recorded {node.ts}</p>}
    {(ranking||similarity)&&<p className="en-caption">{ranking&&`Ranking score ${ranking}`}{ranking&&similarity?' · ':''}{similarity&&`Similarity ${similarity}`}<br/>Retrieval signals, not confidence or factual support.</p>}
    <SourceEvidence item={node} api={api}/></>;
}
function EdgeDetail({edge,graph,api,selectNode}:{edge:QueryEvidenceEdge;graph:QueryEvidence;api:ApiClient;selectNode:(id:string)=>void}){
  const source=graph.nodes.find(node=>node.id===edge.source)!,target=graph.nodes.find(node=>node.id===edge.target)!;
  const fact=typeof edge.data.fact_id==='number'?graph.nodes.find(node=>node.kind==='claim'&&node.data.fact_id===edge.data.fact_id):undefined;
  return <><span className="en-kicker">{edgeCategory(edge)}</span><h3>{evidenceEdgeLabel(edge)}</h3>
    <div className="en-direction"><button type="button" onClick={()=>selectNode(source.id)}>{source.label}</button><span aria-label="points to">↓</span><button type="button" onClick={()=>selectNode(target.id)}>{target.label}</button></div>
    <p className="en-caption">{edge.kind==='returned'?'The engine returned this candidate for the query. This link does not establish relevance or correctness.':edge.kind==='chunked_into'?'This passage belongs to the source record. This is a membership link.':edge.kind==='mentions'?'The passage contains this exact name. A mention does not assert a relationship between concepts.':edge.kind==='asserts'?'This concept is the recorded subject or object of the claim. This is an assertion reference, not another factual relationship.':edge.kind==='source_of'?'This source is recorded as provenance for the claim. Check the retained words to assess its support.':'This is a stored relationship, not proof that either statement is correct. The label preserves the recorded predicate.'}</p>
    {fact&&<button type="button" className="en-text-button" onClick={()=>selectNode(fact.id)}>Inspect recorded claim #{String(fact.data.fact_id)}</button>}
    <SourceEvidence item={edge} api={api} fallback={['source_of','mentions','asserts','chunked_into'].includes(edge.kind)?source:undefined}/>
  </>;
}

export function EvidenceExplorer({graph,api,title,context,close}:{graph:QueryEvidence;api:ApiClient;title:string;context:'query'|'reply';close:()=>void}){
  const dialog=useRef<HTMLDialogElement>(null),inspector=useRef<HTMLElement>(null),closeRef=useRef(close);closeRef.current=close;
  const initialView:EvidenceView=graph.edges.some(edge=>edge.kind==='relation')?'concepts':'provenance';
  const [view,setView]=useState<EvidenceView>(initialView),[groupId,setGroupId]=useState<string|null>(null),[colors,setColors]=useState(true),[labels,setLabels]=useState(true);
  const [nodeId,setNodeId]=useState<string|null>(null),[edgeId,setEdgeId]=useState<string|null>(null),[find,setFind]=useState(''),[records,setRecords]=useState(false),[focus,setFocus]=useState(false);
  const groups=useMemo(()=>detectEvidenceGroups(graph),[graph]);
  const projected=useMemo(()=>projectEvidenceNetwork(graph,view,groups,groupId),[graph,view,groups,groupId]);
  const network=useMemo(()=>{
    if(!focus||!nodeId)return projected;
    const ids=new Set([nodeId,...projected.edges.filter(edge=>edge.source===nodeId||edge.target===nodeId).flatMap(edge=>[edge.source,edge.target])]);
    return {nodes:projected.nodes.filter(node=>ids.has(node.id)),edges:projected.edges.filter(edge=>ids.has(edge.source)&&ids.has(edge.target))};
  },[projected,focus,nodeId]);
  const node=graph.nodes.find(node=>node.id===nodeId),edge=graph.edges.find(edge=>edge.id===edgeId);
  const connections=node?connectionsFor(graph,node.id):[];
  const matches=(find?graph.nodes.filter(node=>node.label.toLocaleLowerCase().includes(find.toLocaleLowerCase())):network.nodes);
  const groupMethod=groups.method==='label_propagation'?'Detected communities':groups.method==='components'?'Connected groups':'Unlinked concepts';
  useLayoutEffect(()=>{
    const element=dialog.current;if(!element)return;
    const opener=document.activeElement instanceof HTMLElement?document.activeElement:null;
    const root=document.documentElement,overflow=root.style.overflow,scroll={x:window.scrollX,y:window.scrollY};
    element.showModal();root.style.overflow='hidden';
    return()=>{element.close();root.style.overflow=overflow;opener?.focus({preventScroll:true});window.scrollTo(scroll.x,scroll.y);};
  },[]);
  function selectNode(id:string){
    const next=graph.nodes.find(node=>node.id===id);if(!next)return;
    if(next.kind!=='concept'&&view==='concepts'){setView('provenance');setGroupId(null);}
    if(!projected.nodes.some(node=>node.id===id))setGroupId(null);
    setNodeId(id);setEdgeId(null);setRecords(false);setFind('');inspector.current?.scrollTo({top:0});
  }
  function selectEdge(id:string){const next=graph.edges.find(edge=>edge.id===id);if(!next)return;setEdgeId(id);setNodeId(next.source);setRecords(false);setFind('');inspector.current?.scrollTo({top:0});}
  function reset(){setView(initialView);setGroupId(null);setNodeId(null);setEdgeId(null);setFind('');setRecords(false);setFocus(false);setColors(true);setLabels(true);}
  return createPortal(<dialog ref={dialog} className="evidence-explorer" aria-label="Evidence explorer" onCancel={event=>{event.preventDefault();closeRef.current();}} onKeyDown={event=>{
    if(event.key!=='Tab')return;
    const focusable=[...event.currentTarget.querySelectorAll<HTMLElement>('button,a[href],input,select,textarea,[tabindex]')].filter(element=>element.tabIndex>=0&&!element.hasAttribute('disabled')&&element.getClientRects().length>0);
    if(!focusable.length)return;
    event.preventDefault();const current=focusable.findIndex(element=>element===document.activeElement),next=(current+(event.shiftKey?-1:1)+focusable.length)%focusable.length;
    focusable[next].focus();
  }}>
    <header className="en-header"><div><span className="en-kicker">Memory explorer</span><h2>{title}</h2></div><button type="button" className="en-close" onClick={close} aria-label="Close evidence explorer">Close <span aria-hidden="true">×</span></button></header>
    <div className="en-context">{context==='reply'?'Evidence supplied to this reply. This is not a model reasoning trace or proof the model used it.':'Explore concepts and follow recorded connections back to their retained sources.'}</div>
    <div className="en-toolbar">
      <div className="en-view-picker" aria-label="Graph view"><button type="button" aria-pressed={view==='concepts'} onClick={()=>{setView('concepts');setGroupId(null);setFocus(false);}}>Concepts</button><button type="button" aria-pressed={view==='provenance'} onClick={()=>{setView('provenance');setGroupId(null);setFocus(false);}}>Evidence</button></div>
      <label className="en-find"><input type="search" aria-label="Find a node" placeholder="Find a concept or source…" value={find} onChange={event=>{setFind(event.target.value);setRecords(true);}}/></label>
      {view==='concepts'&&<select aria-label="Community filter" value={groupId??''} onChange={event=>{setGroupId(event.target.value||null);setFocus(false);}}><option value="">All groups ({groups.groups.length})</option>{groups.groups.map(group=><option key={group.id} value={group.id}>{group.label} · {group.nodeIds.length}</option>)}</select>}
      <button type="button" className="en-records-toggle" aria-pressed={records} onClick={()=>{setRecords(!records);setFind('');}}>Records</button>
    </div>
    <div className="en-display-options"><label><input type="checkbox" checked={colors} onChange={event=>setColors(event.target.checked)}/>{view==='concepts'?'Community colors':'Node colors'}</label><label><input type="checkbox" checked={labels} onChange={event=>setLabels(event.target.checked)}/>Labels</label><label><input type="checkbox" checked={focus} onChange={event=>setFocus(event.target.checked)} disabled={!nodeId}/>Focus selection</label><span>{view==='concepts'?`${groupMethod} · colors show structure, not certainty`:'Dashed: retrieval, membership, mentions · solid: recorded relations'}</span></div>
    <div className="en-main">
      <section className="en-network-pane" aria-label="Evidence graph canvas">
        <EvidenceNetworkCanvas nodes={network.nodes} edges={network.edges} groups={view==='concepts'?groups:undefined} groupColors={colors} selectedNode={nodeId} selectedEdge={edgeId} selectNode={selectNode} selectEdge={selectEdge} reset={reset} labels={labels}/>
        {view==='concepts'&&groups.groups.length>0&&<div className="en-group-legend" aria-label="Detected groups">{groups.groups.slice(0,8).map((group,index)=><button key={group.id} type="button" aria-pressed={groupId===group.id} onClick={()=>{setGroupId(groupId===group.id?null:group.id);setFocus(false);}}><i style={{background:group.isolated?'#8995a7':GROUP_COLORS[index%GROUP_COLORS.length]}}/>{group.label}<span>{group.nodeIds.length}</span></button>)}</div>}
      </section>
      <aside ref={inspector} className="en-inspector" aria-label="Evidence inspector" onClick={event=>{if((event.target as Element).closest('a'))close();}}>
        <div className="en-inspector-heading"><strong>{records||find?'Records':'Inspector'}</strong><span>{records||find?`${matches.length} shown`:edge?'Relationship':node?kindName[node.kind]:'Select a node or link'}</span></div>
        {records||find?<div className="en-record-list">{matches.length?matches.map(record=><button type="button" key={record.id} onClick={()=>selectNode(record.id)}><small>{kindName[record.kind]}</small><strong>{record.label}</strong></button>):<p className="en-caption">No record names match this search.</p>}</div>:<>
          <div className="en-selected" aria-live="polite">{edge?<EdgeDetail edge={edge} graph={graph} api={api} selectNode={selectNode}/>:node?<NodeDetail node={node} api={api}/>:<div className="en-inspector-empty"><div className="en-explore-symbol" aria-hidden="true">⌘</div><h3>Follow a connection.</h3><p>Select a node to read its record, or select a link to inspect the exact relationship and source quotation.</p><p>{groups.method==='label_propagation'?'Communities were detected by deterministic label propagation using recorded concept relations.':groups.method==='components'?'These groups are connected components. The evidence is too sparse to assert a finer community structure.':'No connected concept relationships were returned. Named mentions remain unlinked.'}</p><button type="button" onClick={()=>setRecords(true)}>Browse all {network.nodes.length} records</button></div>}</div>
          {node&&<section className="en-connections"><h4>Connections <span>{connections.length}</span></h4><p className="en-caption">Arrows preserve the recorded direction.</p><ul>{connections.map(link=>{const from=graph.nodes.find(record=>record.id===link.source)!,to=graph.nodes.find(record=>record.id===link.target)!;return <li key={link.id}><button type="button" className="en-connection" aria-pressed={edgeId===link.id} onClick={()=>selectEdge(link.id)}><span className="en-connection-kind">{evidenceEdgeLabel(link)}<small>{edgeCategory(link)}</small></span><span className="en-connection-path"><span>{from.label}</span><b aria-label="points to">→</b><span>{to.label}</span></span></button></li>;})}</ul>{!connections.length&&<p className="en-caption">No recorded connections were returned for this node.</p>}</section>}
        </>}
        {(graph.truncated||graph.provenanceMissing>0||graph.provenanceOmitted>0||graph.notices.length>0)&&<details className="en-notices"><summary>Evidence availability{graph.truncated?' · partial view':''}</summary>{graph.truncated&&<p>Some records or connections are omitted from this bounded view.</p>}{graph.provenanceMissing>0&&<p>{graph.provenanceMissing} source records are missing.</p>}{graph.provenanceOmitted>0&&<p>{graph.provenanceOmitted} source records were omitted.</p>}{graph.discarded>0&&<p>{graph.discarded} invalid or unsupported graph records were excluded.</p>}{graph.notices.map((notice,index)=><p key={index}>{notice}</p>)}</details>}
      </aside>
    </div>
    <footer className="en-footer"><span>Read-only · {graph.nodes.length} records in this response</span><span>Group colors and distance are not confidence or proof.</span></footer>
  </dialog>,document.body);
}
