import {sourceKinds,type RecallScope,type ScopeDraft} from './recall-scope';

const labels:Record<string,string>={note:'Notes',file:'Files',conversation:'Conversations',observation:'Observations',connector:'Connectors'};

export function RecallScopeEditor({draft,onChange,disabled}:{draft:ScopeDraft;onChange:(value:ScopeDraft)=>void;disabled:boolean}){
  return <fieldset className="recall-scope-editor" disabled={disabled}>
    <legend>Memory selection</legend><p>Choose the knowledge this session may retrieve. Every filter must match. No matches means no retrieved context—not a wider search.</p>
    <div className="recall-scope-kinds" role="group" aria-label="Source type">
      <button type="button" aria-pressed={!draft.kind} onClick={()=>onChange({...draft,kind:''})}>All types</button>
      {sourceKinds.map(kind=><button key={kind} type="button" aria-pressed={draft.kind===kind} onClick={()=>onChange({...draft,kind})}>{labels[kind]}</button>)}
    </div>
    <label>Source prefix<input aria-describedby="scope-source-help" value={draft.source_prefix} onChange={e=>onChange({...draft,source_prefix:e.target.value})} placeholder="e.g. docs/ or https://example.com/" maxLength={1000}/></label><small id="scope-source-help">Literal start of the recorded source. Leave blank for any source.</small>
    <details className="recall-scope-advanced" open={Boolean(draft.since||draft.until||draft.metadata.length)||undefined}>
      <summary>Dates & metadata</summary>
      <div className="recall-scope-dates"><label>Created on or after (UTC)<input value={draft.since} onChange={e=>onChange({...draft,since:e.target.value})} placeholder="2026-09-01 or timestamp"/></label>
        <label>Created on or before (UTC)<input value={draft.until} onChange={e=>onChange({...draft,until:e.target.value})} placeholder="2026-09-30T23:59:59Z"/></label></div>
      <p>Boundaries are inclusive. A date alone means midnight UTC, not the end of that day.</p>
      {draft.metadata.map((row,index)=><div className="recall-scope-metadata" key={index}>
        <label>Key<input aria-label={`Metadata key ${index+1}`} value={row.key} placeholder="collection" maxLength={32} onChange={e=>onChange({...draft,metadata:draft.metadata.map((item,i)=>i===index?{...item,key:e.target.value}:item)})}/></label>
        <label>Value<input aria-label={`Metadata value ${index+1}`} value={row.value} placeholder="manuals" maxLength={256} onChange={e=>onChange({...draft,metadata:draft.metadata.map((item,i)=>i===index?{...item,value:e.target.value}:item)})}/></label>
        <button type="button" aria-label={`Remove metadata filter ${index+1}`} onClick={()=>onChange({...draft,metadata:draft.metadata.filter((_,i)=>i!==index)})}>×</button>
      </div>)}
      <button type="button" className="recall-scope-add" disabled={disabled||draft.metadata.length>=16} onClick={()=>onChange({...draft,metadata:[...draft.metadata,{key:'',value:''}]})}>Add metadata filter</button>
    </details>
    <p className="recall-scope-boundary">Fixed for this session. These filters do not limit transcript capture or erase conversation history.</p>
  </fieldset>;
}

export function RecallScopeSummary({scope}:{scope:RecallScope|undefined}){
  const count=scope?Object.keys(scope).length:0;
  return <details className="recall-scope-summary"><summary><span>Memory selection</span><small>{scope===undefined?'Not reported':count?'Filtered · fixed for this session':'No additional filters'}</small></summary>
    {scope===undefined?<p>This server did not report the session’s recall filters.</p>:<>
      {count?<dl>{scope.kind&&<><dt>Source type</dt><dd>{labels[scope.kind]}</dd></>}
        {scope.source_prefix!==undefined&&<><dt>Source prefix</dt><dd>{scope.source_prefix||'Any present source (excludes records without a source)'}</dd></>}
        {scope.since&&<><dt>Created on or after</dt><dd>{scope.since}</dd></>}{scope.until&&<><dt>Created on or before</dt><dd>{scope.until}</dd></>}
        {scope.where&&<><dt>Metadata</dt><dd>{Object.entries(scope.where).map(([key,value])=><div key={key}>{key} = {value}</div>)}</dd></>}
      </dl>:<p>No extra session filters were requested. Server access rules still apply.</p>}
      <p>Retrieval filters apply together. They do not restrict saved public messages or the session’s own history. Open a new session to choose different filters.</p>
    </>}
  </details>;
}
