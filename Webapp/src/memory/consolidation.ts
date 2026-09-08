export type PassScope='distill'|'derive';
export interface PassReceipt {
  space:string;
  scope:PassScope;
  counts:Array<[string,number]>;
  reasons:Array<[string,number]>;
  error:string|null;
  latencyMs:number;
}
const fields={
  distill:[['episodes','Episodes read'],['proposed','Proposals added'],['accepted','Claims accepted'],['closed','Claims closed'],['skipped','Candidates skipped'],['parked','Episodes parked'],['rejected','Candidates rejected'],['expired','Sources expired'],['derived_sent','Inference groups sent'],['derived_proposed','Inferences proposed'],['derived_restated','Inferences restated'],['derived_rejected','Inferences rejected']],
  derive:[['groups','Groups considered'],['sent','Groups sent'],['proposed','Inferences proposed'],['restated','Inferences restated'],['rejected','Candidates rejected']],
} as const;
function object(value:unknown):Record<string,unknown>{
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid processing pass report');
  return value as Record<string,unknown>;
}
function count(value:unknown):number{
  if(typeof value!=='number'||!Number.isSafeInteger(value)||value<0)throw Error('Invalid processing pass count');
  return value;
}
export function parsePassReceipt(value:unknown,space:string,scope:PassScope):PassReceipt {
  const data=object(value);
  if(data.space!==space||data.scope!==scope)throw Error('Processing pass report does not match the requested space and action');
  const counts:Array<[string,number]>=fields[scope].map(([key,label])=>[label,count(data[key])]);
  const reasons=Object.entries(object(data.rejected_reasons)).map(([reason,n]):[string,number]=>[reason,count(n)]);
  if(typeof data.latency_ms!=='number'||!Number.isFinite(data.latency_ms)||data.latency_ms<0)throw Error('Invalid processing pass duration');
  if(scope==='distill'&&data.error!==null&&typeof data.error!=='string')throw Error('Invalid processing pass error');
  return {space,scope,counts,reasons,error:typeof data.error==='string'?data.error:null,latencyMs:data.latency_ms};
}
