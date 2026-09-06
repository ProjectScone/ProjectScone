export interface ApiClient {
  request<T>(path: string, options?: RequestInit): Promise<T>;
  image(attachment: ImageAttachment, signal?: AbortSignal): Promise<Blob>;
  uploadImage(file: File, signal?: AbortSignal): Promise<ImageAttachment>;
  conversationStream(sid: string, requestId: string, after: number, signal: AbortSignal): Promise<ReadableStream<Uint8Array>>;
}

export interface ImageAttachment { attachment_id: string; media_type: string; bytes: number; filename?: string | null }
export const PREVIEW_TYPES = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);
const MAX_IMAGE_BYTES = 25 * 1024 * 1024;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

export function createApiClient(key: string, unauthorized: () => void, base = ''): ApiClient {
  return {
    async conversationStream(sid, requestId, after, signal) {
      if (![sid,requestId].every(id=>/^[A-Za-z0-9._:-]{1,128}$/.test(id)&&id!=='.'&&id!=='..')
        ||!Number.isSafeInteger(after)||after<0) throw Error('Invalid conversation stream address');
      const path='/v1/conversations/'+encodeURIComponent(sid)+'/turns/'+encodeURIComponent(requestId)+'/stream?after='+after;
      const response=await fetch(base+path,{
        method:'GET',headers:{Authorization:'Bearer '+key,Accept:'text/event-stream'},signal,
        cache:'no-store',redirect:'error',credentials:'omit',referrerPolicy:'no-referrer',
      });
      if(!response.ok){
        await response.body?.cancel();
        if(response.status===401||response.status===403)unauthorized();
        throw new ApiError(response.status,`Live preview request failed (${response.status})`);
      }
      if(response.headers.get('content-type')?.split(';')[0].trim()!=='text/event-stream'||!response.body){
        await response.body?.cancel();throw Error('Invalid conversation stream response');
      }
      return response.body;
    },
    async uploadImage(file: File, signal?: AbortSignal): Promise<ImageAttachment> {
      if (!PREVIEW_TYPES.has(file.type)) throw Error('Choose a PNG, JPEG, GIF or WebP image');
      if (!file.size || file.size > MAX_IMAGE_BYTES) throw Error('Image size must be between 1 byte and 25 MB');
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),byte=>byte.toString(16).padStart(2,'0')).join('');
      const response=await fetch(base+'/v1/attachments',{
        method:'POST',body:file,headers:{Authorization:'Bearer '+key,'Content-Type':file.type,'X-Filename':file.name.replace(/[^a-zA-Z0-9._-]/g,'_').slice(0,120)},
        cache:'no-store',redirect:'error',credentials:'omit',referrerPolicy:'no-referrer',
        signal:signal?AbortSignal.any([signal,AbortSignal.timeout(30000)]):AbortSignal.timeout(30000),
      });
      const receipt=await response.json().catch(()=>null);
      if(!response.ok){
        if(response.status===401||response.status===403)unauthorized();
        throw new ApiError(response.status,typeof receipt?.error==='string'?receipt.error:`Upload request failed (${response.status})`);
      }
      if(receipt?.attachment_id!==digest||receipt?.media_type!==file.type||receipt?.bytes!==file.size
        ||(receipt.filename!=null&&typeof receipt.filename!=='string'))throw Error('Upload receipt does not match the selected image');
      return receipt as ImageAttachment;
    },
    async image(attachment: ImageAttachment, signal?: AbortSignal): Promise<Blob> {
      if (!/^[a-f0-9]{64}$/.test(attachment.attachment_id) || !PREVIEW_TYPES.has(attachment.media_type)
        || !Number.isSafeInteger(attachment.bytes) || attachment.bytes < 1 || attachment.bytes > MAX_IMAGE_BYTES) throw Error('Unsupported image reference or size');
      const response = await fetch(base + '/v1/attachments/' + attachment.attachment_id, {
        headers:{Authorization:'Bearer ' + key}, cache:'no-store', redirect:'error', credentials:'omit', referrerPolicy:'no-referrer',
        signal:signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
      });
      if (!response.ok) {
        await response.body?.cancel();
        if (response.status === 401 || response.status === 403) unauthorized();
        throw new ApiError(response.status, `Image request failed (${response.status})`);
      }
      const type = response.headers.get('content-type')?.split(';')[0].trim();
      if (type !== attachment.media_type || !PREVIEW_TYPES.has(type)) { await response.body?.cancel(); throw Error('Image response has an unsupported content type'); }
      const length = response.headers.get('content-length');
      if (length && Number(length) > attachment.bytes) { await response.body?.cancel(); throw Error('Image response exceeds its recorded size'); }
      const reader = response.body?.getReader();
      if (!reader) throw Error('Image response contains no bytes');
      const chunks: Uint8Array<ArrayBuffer>[] = []; let total=0;
      try {
        while (true) {
          const {done,value}=await reader.read();if(done)break;
          total+=value.byteLength;
          if(total>attachment.bytes){await reader.cancel();throw Error('Image response exceeds its recorded size');}
          chunks.push(new Uint8Array(value));
        }
      } finally {reader.releaseLock();}
      if(total!==attachment.bytes)throw Error('Image response size does not match its record');
      const blob=new Blob(chunks,{type});
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer())),byte=>byte.toString(16).padStart(2,'0')).join('');
      if(digest!==attachment.attachment_id)throw Error('Image digest does not match its source reference');
      return blob;
    },
    async request<T>(path: string, options: RequestInit = {}): Promise<T> {
      // The client cannot be used to forward a space key to an arbitrary origin.
      if (!path.startsWith('/v1/') && path !== '/healthz') throw new Error('Unsupported API path');
      const headers = new Headers(options.headers);
      headers.set('Authorization', 'Bearer ' + key);
      if (options.body) headers.set('Content-Type', 'application/json');
      const response = await fetch(base + path, { ...options, headers });
      if (response.status === 204 && options.method?.toUpperCase() === 'DELETE') return undefined as T;
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) unauthorized();
        throw new ApiError(response.status, typeof body?.error === 'string' ? body.error : `API request failed (${response.status})`);
      }
      if (body === null) throw new Error('API returned invalid JSON');
      return body as T;
    },
  };
}
