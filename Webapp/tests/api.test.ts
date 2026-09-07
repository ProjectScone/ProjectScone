import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createApiClient, ApiError } from '../src/api.ts';
import { createHash } from 'node:crypto';

test('conversation streams authenticate a read-only cursor and reject redirects, unsafe paths and non-SSE responses',async()=>{
  let status=200,type='text/event-stream; charset=utf-8',denied=false,calls=0;
  const server=createServer((req,res)=>{
    calls++;assert.equal(req.method,'GET');assert.equal(req.url,'/v1/conversations/session/turns/turn/stream?after=7');
    assert.equal(req.headers.authorization,'Bearer allowed');assert.equal(req.headers.accept,'text/event-stream');
    res.writeHead(status,{'content-type':type,location:'/unexpected'});res.end(': keep-alive\n\n');
  });
  await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
  const address=server.address();if(!address||typeof address==='string')throw Error('missing address');
  const client=createApiClient('allowed',()=>{denied=true;},`http://127.0.0.1:${address.port}`);
  const controller=new AbortController();
  try{
    assert.equal(typeof client.conversationStream,'function');
    const stream=await client.conversationStream('session','turn',7,controller.signal);
    assert.equal(await new Response(stream).text(),': keep-alive\n\n');
    for(const bad of ['../secret','..','.','https://elsewhere','with/slash'])await assert.rejects(client.conversationStream(bad,'turn',7,controller.signal));
    await assert.rejects(client.conversationStream('session','turn',-1,controller.signal));assert.equal(calls,1);
    type='application/json';await assert.rejects(client.conversationStream('session','turn',7,controller.signal),/stream/i);
    type='text/event-stream';status=302;await assert.rejects(client.conversationStream('session','turn',7,controller.signal));assert.equal(calls,3);
    status=401;await assert.rejects(client.conversationStream('session','turn',7,controller.signal),e=>e instanceof ApiError&&e.status===401);assert.equal(denied,true);
  }finally{server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));}
});

test('an acknowledged no-content delete succeeds but an empty read still fails', async()=>{
  const server=createServer((req,res)=>{assert.equal(req.headers.authorization,'Bearer allowed');res.writeHead(204);res.end();});
  await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
  const address=server.address();if(!address||typeof address==='string')throw Error('missing address');
  const client=createApiClient('allowed',()=>{},`http://127.0.0.1:${address.port}`);
  try{
    assert.equal(await client.request('/v1/conversations/ended',{method:'DELETE'}),undefined);
    await assert.rejects(client.request('/v1/status'),/invalid JSON/);
  }finally{server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));}
});

test('typed client sends the selected key and reports permission failure', async () => {
  let denied = false;
  const server = createServer((req, res) => {
    res.setHeader('content-type', 'application/json');
    if (req.headers.authorization !== 'Bearer allowed') { res.statusCode = 401; return res.end('{"error":"unknown key"}'); }
    res.end('{"space":"private"}');
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw Error('missing local address');
  const base = `http://127.0.0.1:${address.port}`;
  try {
    assert.deepEqual(await createApiClient('allowed', () => {}, base).request('/v1/status'), { space: 'private' });
    await assert.rejects(createApiClient('wrong', () => { denied = true; }, base).request('/v1/status'), (error: unknown) => error instanceof ApiError && error.status === 401);
    assert.equal(denied, true);
  } finally { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); }
});

test('image downloads authenticate, validate bytes and reject unsafe content', async () => {
  const bytes = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=', 'base64');
  const id = createHash('sha256').update(bytes).digest('hex');
  let type='image/png', status=200, calls=0, denied=false;
  const server=createServer((req,res)=>{calls++;assert.equal(req.headers.authorization,'Bearer allowed');res.writeHead(status,{'content-type':type});res.end(bytes);});
  await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
  const address=server.address();if(!address||typeof address==='string')throw Error('missing address');
  const client=createApiClient('allowed',()=>{denied=true;},`http://127.0.0.1:${address.port}`);
  const attachment={attachment_id:id,media_type:'image/png',bytes:bytes.length,filename:'pixel.png'};
  try {
    const blob=await client.image(attachment);
    assert.equal(blob.type,'image/png');assert.deepEqual(Buffer.from(await blob.arrayBuffer()),bytes);
    await assert.rejects(client.image({...attachment,attachment_id:'../../etc/passwd'}));assert.equal(calls,1);
    await assert.rejects(client.image({...attachment,media_type:'image/svg+xml'}));assert.equal(calls,1);
    await assert.rejects(client.image({...attachment,bytes:26*1024*1024}));assert.equal(calls,1);
    type='text/html';await assert.rejects(client.image(attachment),/type/i);
    type='image/png';await assert.rejects(client.image({...attachment,bytes:bytes.length-1}),/size/i);
    await assert.rejects(client.image({...attachment,attachment_id:'a'.repeat(64)}),/digest/i);
    status=404;await assert.rejects(client.image(attachment),error=>error instanceof ApiError&&error.status===404);assert.equal(denied,false);
    status=401;await assert.rejects(client.image(attachment));assert.equal(denied,true);
  } finally {server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));}
});

test('image uploads send exact bytes and validate the receipt before linking', async () => {
  const bytes=Buffer.from('an opaque raster fixture');
  const digest=createHash('sha256').update(bytes).digest('hex');let bad=false,calls=0;
  const server=createServer((req,res)=>{
    calls++;assert.equal(req.method,'POST');assert.equal(req.url,'/v1/attachments');
    assert.equal(req.headers.authorization,'Bearer allowed');assert.equal(req.headers['content-type'],'image/png');
    const chunks:Buffer[]=[];req.on('data',chunk=>chunks.push(chunk));req.on('end',()=>{
      assert.deepEqual(Buffer.concat(chunks),bytes);
      res.setHeader('content-type','application/json');
      res.end(JSON.stringify({attachment_id:bad?'a'.repeat(64):digest,bytes:bytes.length,media_type:'image/png',filename:'source.png'}));
    });
  });
  await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
  const address=server.address();if(!address||typeof address==='string')throw Error('missing address');
  const client=createApiClient('allowed',()=>{},`http://127.0.0.1:${address.port}`);
  const file=new File([bytes],'source.png',{type:'image/png'});
  try {
    assert.equal((await client.uploadImage(file)).attachment_id,digest);
    await assert.rejects(client.uploadImage(new File(['<svg/>'],'bad.svg',{type:'image/svg+xml'})),/image|type/i);
    await assert.rejects(client.uploadImage(new File([],'empty.png',{type:'image/png'})),/size|empty/i);
    assert.equal(calls,1);
    bad=true;await assert.rejects(client.uploadImage(file),/receipt|digest/i);
  } finally {server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));}
});
