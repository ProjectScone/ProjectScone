export interface ModelChoice {provider:string;model:string}
export interface VoiceChoice extends ModelChoice {voice:string}
export interface PersonaIdentity {id:string;name:string|null;fingerprint?:string|null;current?:boolean}
export interface PersonaOption extends PersonaIdentity {name:string;reply:ModelChoice;transcription:ModelChoice|null;speech:VoiceChoice|null;activity:ModelChoice|null;text_ready:boolean;voice_ready:boolean}

function record(value:unknown):Record<string,unknown>{if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid persona response');return value as Record<string,unknown>;}
function label(value:unknown,max:number,pattern?:RegExp):string{
  if(typeof value!=='string'||!value.trim()||value.length>max||(pattern&&!pattern.test(value)))throw Error('Invalid persona label');
  return value;
}
export function readPersonaIdentity(value:unknown):PersonaIdentity{
  const v=record(value);
  const identity={id:label(v.id,128,/^[A-Za-z0-9][A-Za-z0-9._-]*$/),name:v.name===null?null:label(v.name,120)};
  if(!('fingerprint' in v)&&!('current' in v))return identity;
  const fingerprint=v.fingerprint===null?null:label(v.fingerprint,16,/^[a-f0-9]{16}$/);
  if(typeof v.current!=='boolean'||(v.current&&(fingerprint===null||identity.name===null)))throw Error('Invalid persona version');
  return {...identity,fingerprint,current:v.current};
}
function choice(value:unknown):ModelChoice{
  const v=record(value);
  return {provider:label(v.provider,64,/^[a-z][a-z0-9_-]*$/),model:label(v.model,128,/^[A-Za-z0-9][A-Za-z0-9._/-]*$/)};
}
export function readPersonaCatalog(value:unknown):PersonaOption[]{
  const v=record(value);
  if(v.schema_version!==1||!Array.isArray(v.personas)||v.personas.length>1000)throw Error('Unsupported persona catalog');
  if('revision' in v)label(v.revision,16,/^[a-f0-9]{16}$/);
  const seen=new Set<string>();
  return v.personas.map(value=>{
    const item=record(value),identity=readPersonaIdentity({id:item.id,name:item.name});
    const version=('revision' in v||'fingerprint' in item)?{fingerprint:label(item.fingerprint,16,/^[a-f0-9]{16}$/)}:{};
    if(identity.name===null||seen.has(identity.id)||typeof item.text_ready!=='boolean'||typeof item.voice_ready!=='boolean')throw Error('Invalid persona catalog');
    seen.add(identity.id);
    return {...identity,...version,name:identity.name,reply:choice(item.reply),transcription:item.transcription===null?null:choice(item.transcription),
      speech:item.speech===null?null:{...choice(item.speech),voice:label(record(item.speech).voice,128,/^[A-Za-z0-9][A-Za-z0-9._/-]*$/)},
      activity:item.activity===null?null:choice(item.activity),text_ready:item.text_ready,voice_ready:item.voice_ready};
  });
}
