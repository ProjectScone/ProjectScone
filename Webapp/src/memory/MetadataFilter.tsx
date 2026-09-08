import {useState} from 'react';
import {blankGroup,blankRule,comparisons,compileFilter,type AppliedFilter,type FilterGroup,type FilterNode,type FilterOperator} from './metadata-filter';
import './metadata-filter.css';

export function MetadataFilter({value,onApply}:{value?:AppliedFilter;onApply:(value:AppliedFilter|undefined)=>void}){
  const [open,setOpen]=useState(false),[draft,setDraft]=useState<FilterGroup>(blankGroup),[error,setError]=useState('');
  function edit(){setDraft(value?.draft??blankGroup());setError('');setOpen(true);}
  function apply(){try{const json=compileFilter(draft);onApply({draft,json});setOpen(false);setError('');}catch(e){setError(e instanceof Error?e.message:'Check the filter rules.');}}
  return <div className="metadata-filter">
    <div className="metadata-filter-summary"><button type="button" className="btn quiet small" aria-expanded={open} onClick={edit}>{value?'Edit metadata filter':'Metadata filters'}</button>
      {value&&<><span>Metadata filter applied</span><button type="button" className="btn quiet small" onClick={()=>{onApply(undefined);setOpen(false);}}>Clear metadata filter</button></>}
    </div>
    {open&&<section className="metadata-editor" aria-label="Metadata filter editor"><header><span className="eyebrow">Narrow by recorded metadata</span><h3>Find the context that matters</h3>
      <p>Combine rules before the engine retrieves sources. A rule on a missing key never matches, even when negated. Numeric tests parse stored text as numbers.</p></header>
      <GroupEditor group={draft} path="root" depth={1} change={setDraft}/>
      {error&&<p className="metadata-error" role="alert">{error}</p>}
      <div className="metadata-actions"><button type="button" className="btn" onClick={apply}>Apply metadata filter</button><button type="button" className="btn quiet" onClick={()=>setOpen(false)}>Cancel filter changes</button><span>Draft changes do not run a search.</span></div>
    </section>}
  </div>;
}

function GroupEditor({group,path,depth,change}:{group:FilterGroup;path:string;depth:number;change:(value:FilterGroup)=>void}){
  const update=(index:number,node:FilterNode)=>change({...group,children:group.children.map((value,i)=>i===index?node:value)});
  const remove=(index:number)=>change({...group,children:group.children.filter((_,i)=>i!==index)});
  return <fieldset className={`metadata-group${depth>=3?' metadata-group-deep':''}`}><legend>{path==='root'?'Matching rules':`Group ${path}`}</legend>
    <label className="metadata-mode">Match<select aria-label={`Match rules in group ${path}`} value={group.mode} onChange={e=>change({...group,mode:e.target.value==='any'?'any':'all'})}><option value="all">All rules</option><option value="any">Any rule</option></select><span>{group.mode==='all'?'Every rule must match':'At least one rule must match'}</span></label>
    {group.children.map((node,index)=>{
      const here=path==='root'?String(index+1):`${path}.${index+1}`;
      return <div className="metadata-node" key={index}>{node.kind==='group'?<GroupEditor group={node} path={here} depth={depth+1} change={next=>update(index,next)}/>:<div className="metadata-rule">
        <label>Metadata key<input aria-label={`Metadata field ${here}`} value={node.field} maxLength={32} placeholder="status" onChange={e=>update(index,{...node,field:e.target.value})}/></label>
        <label>Comparison<select aria-label={`Comparison ${here}`} value={node.operator} onChange={e=>update(index,{...node,operator:e.target.value as FilterOperator})}>{comparisons.map(([operator,label])=><option key={operator} value={operator}>{label}</option>)}</select></label>
        {node.operator==='present'?<p className="metadata-presence">No comparison value needed.</p>:<label>{node.operator==='in'?'Values · one per line':'Value'}{node.operator==='in'?<textarea aria-label={`Value ${here}`} value={node.value} rows={3} onChange={e=>update(index,{...node,value:e.target.value})}/>:<input aria-label={`Value ${here}`} value={node.value} placeholder={['is','has'].includes(node.operator)?'published':'10'} onChange={e=>update(index,{...node,value:e.target.value})}/>}</label>}
        <label className="metadata-negate"><input type="checkbox" aria-label={`Negate rule ${here}`} checked={node.negated} onChange={e=>update(index,{...node,negated:e.target.checked})}/>Negate rule</label>
      </div>}<button type="button" className="metadata-remove" aria-label={`Remove ${node.kind} ${here}`} onClick={()=>remove(index)}>Remove {node.kind}</button></div>;
    })}
    {!group.children.length&&<p className="metadata-empty">Add a rule or remove this empty group before applying.</p>}
    <div className="metadata-add"><button type="button" className="btn quiet small" aria-label={`Add rule to ${path}`} onClick={()=>change({...group,children:[...group.children,blankRule()]})}>+ Add rule</button><button type="button" className="btn quiet small" aria-label={`Add group to ${path}`} disabled={depth>=8} onClick={()=>change({...group,children:[...group.children,blankGroup()]})}>+ Add group</button></div>
  </fieldset>;
}
