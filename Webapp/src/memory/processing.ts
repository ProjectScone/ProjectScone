export interface ProcessingStatus {
  space:string;
  episodes?:number;
  chunks?:number;
  bytes?:number;
  revision?:number;
  pending_review?:number;
  pending_distill?:number;
  failed_distill?:number;
  model?:string|null;
  pending_derivation?:number;
  semantic_lane?:string;
  derivation?:string;
  embedder?:string;
  vector_index?:string;
  document_store?:string;
}

export function parseProcessingStatus(value:unknown):ProcessingStatus {
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid memory status');
  const data=value as Record<string,unknown>;
  if(typeof data.space!=='string'||!data.space.trim())throw Error('Invalid memory status: missing space');
  const result:ProcessingStatus={space:data.space};
  for(const key of ['episodes','chunks','bytes','revision','pending_review','pending_distill','failed_distill','pending_derivation'] as const){
    const count=data[key];
    if(count==null)continue;
    if(typeof count!=='number'||!Number.isSafeInteger(count)||count<0)throw Error(`Invalid memory status: ${key}`);
    result[key]=count;
  }
  for(const key of ['semantic_lane','derivation','embedder','vector_index','document_store'] as const){
    const text=data[key];
    if(text==null)continue;
    if(typeof text!=='string')throw Error(`Invalid memory status: ${key}`);
    result[key]=text;
  }
  if(data.model!==undefined){
    if(data.model!==null&&(typeof data.model!=='string'||!data.model.trim()))throw Error('Invalid memory status: model');
    result.model=data.model as string|null;
  }
  return result;
}
