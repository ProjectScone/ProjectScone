import type {ImageAttachment} from '../api';

export interface ImageUnderstandingReceipt {
  schema_version:1;episode_id:number;attachment_id:string;persisted:false;
  understanding:{text:string;attachment_id:string;source:string|null;media_type:string;model:string;width:number;height:number;origin:'model_generated'};
}
const supported=new Set(['image/png','image/jpeg','image/webp']);
const object=(value:unknown):Record<string,unknown>=>{
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid image understanding response.');
  return value as Record<string,unknown>;
};
export function sourceUnderstandingImages(value:unknown):ImageAttachment[]{
  if(value===undefined)return [];
  if(!Array.isArray(value))throw Error('Invalid source attachment metadata.');
  return value.map(item=>{
    const data=object(item);
    if(typeof data.attachment_id!=='string'||!/^[a-f0-9]{64}$/.test(data.attachment_id)||typeof data.media_type!=='string'||typeof data.bytes!=='number'||!Number.isSafeInteger(data.bytes)||data.bytes<1||!(data.filename==null||typeof data.filename==='string'))throw Error('Invalid source attachment metadata.');
    return {attachment_id:data.attachment_id,media_type:data.media_type,bytes:data.bytes,...(typeof data.filename==='string'?{filename:data.filename}:{})};
  }).filter(item=>supported.has(item.media_type));
}
export function parseImageUnderstanding(value:unknown,episodeId:number,attachmentId:string):ImageUnderstandingReceipt{
  const data=object(value),result=object(data.understanding);
  if(data.schema_version!==1||data.episode_id!==episodeId||data.attachment_id!==attachmentId||data.persisted!==false||result.attachment_id!==attachmentId||result.origin!=='model_generated')throw Error('Image understanding does not match the selected source.');
  if(typeof result.text!=='string'||!result.text.trim()||result.text.length>256000||typeof result.model!=='string'||!result.model.trim()||result.model.length>256||!(result.source===null||typeof result.source==='string')||typeof result.media_type!=='string'||!supported.has(result.media_type))throw Error('Invalid image understanding description.');
  const width=result.width,height=result.height;
  if(typeof width!=='number'||!Number.isSafeInteger(width)||width<1||typeof height!=='number'||!Number.isSafeInteger(height)||height<1)throw Error('Invalid image understanding dimensions.');
  return {schema_version:1,episode_id:episodeId,attachment_id:attachmentId,persisted:false,understanding:{text:result.text,attachment_id:attachmentId,source:result.source,media_type:result.media_type,model:result.model,width,height,origin:'model_generated'}};
}
export function imageMemoryBody(receipt:ImageUnderstandingReceipt,prompt:string,requestId:string){
  return {kind:'note',content:`Model-generated image interpretation\nOriginal source episode: ${receipt.episode_id}\nOriginal image: ${receipt.attachment_id}\nTask: ${prompt}\n\n${receipt.understanding.text}`,
    source:`scone:episode:${receipt.episode_id}:attachment:${receipt.attachment_id}`,
    attachment_ids:[receipt.attachment_id],tags:['image-analysis'],dedup_key:`image-analysis:${requestId}`,
    metadata:{origin:'model_generated',vision_model:receipt.understanding.model,source_episode_id:String(receipt.episode_id),source_attachment_id:receipt.attachment_id}};
}
