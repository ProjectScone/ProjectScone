// Real Chromium Web Audio + microphone worklet, with synthetic hardware only.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const http=require('node:http');
const path=require('node:path');
const fs=require('node:fs');
const os=require('node:os');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
let browser,server,base,bundle;
before(async()=>{
  const {build}=await import(pathToFileURL(path.resolve(__dirname,'../Webapp/node_modules/vite/dist/node/index.js')));
  const output=fs.mkdtempSync(path.join(os.tmpdir(),'scone-voice-build-'));
  await build({configFile:false,logLevel:'error',build:{outDir:output,emptyOutDir:false,minify:true,lib:{entry:path.resolve(__dirname,'fixtures/voice-browser.ts'),formats:['es'],fileName:()=> 'voice.js'}}});
  bundle=fs.readFileSync(path.join(output,'voice.js'));
  server=http.createServer((req,res)=>{
    if(req.url==='/voice-test'){
      res.setHeader('content-type','text/html');res.end(`<!doctype html><button id="start">Connect microphone</button><script type="module">
import {startVoiceDevice,createApiClient} from '/voice-device.js';
window.states=[];window.failures=[];window.frames=[];window.tracks=[];window.contexts=[];window.stops=0;
const gum=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia=async options=>{window.constraints=options;const stream=await gum(options);window.tracks.push(...stream.getTracks());if(window.deferPermission)await new Promise(resolve=>window.resolvePermission=resolve);return stream;};
const NativeAudioContext=window.AudioContext;
window.outputs=[];window.audioNodes=[];window.stoppedNodes=[];
window.AudioContext=class extends NativeAudioContext{constructor(...args){super(...args);window.contexts.push(this);}createBufferSource(){const node=super.createBufferSource(),start=node.start.bind(node),stop=node.stop.bind(node);window.audioNodes.push(node);node.start=(...args)=>{window.outputs.push({rate:node.buffer.sampleRate,channels:node.buffer.numberOfChannels,sample:node.buffer.getChannelData(0)[0]});start(...args);};node.stop=(...args)=>{window.stoppedNodes.push(node);stop(...args);};return node;}};
document.querySelector('#start').onclick=()=>{
 window.abort=new AbortController();
 window.device=startVoiceDevice((format,events)=>{
   window.format=format;window.events=events;events.state('connecting');
   if(window.nativeSession)return createApiClient(window.nativeKey||'voice-alpha',()=>{},window.nativeBase).voiceConnection(window.nativeSession,format,events,window.abort.signal);
   if(!window.deferReady)queueMicrotask(()=>events.state('ready'));
   return {send:samples=>window.frames.push([...samples]),stop:()=>window.stops++};
 },(state,failure)=>{window.states.push(state);if(failure)window.failures.push(failure);},window.abort.signal);
 window.device.ready.catch(()=>{});
};
window.loaded=true;
</script>`);return;
    }
    if(req.url==='/voice-device.js'){res.setHeader('content-type','application/javascript');res.end(bundle);return;}
    res.writeHead(404);res.end();
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));base='http://127.0.0.1:'+server.address().port;
  browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
});
after(async()=>{await browser?.close();if(server){server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}});
async function pageFor(t){const page=await browser.newPage();t.after(()=>page.close());await page.goto(base+'/voice-test');await page.waitForFunction(()=>window.loaded);return page;}
test('explicit microphone consent starts bounded PCM capture; mute and stop control the real device',async t=>{
  const page=await pageFor(t);
  assert.equal(await page.evaluate(()=>window.tracks.length),0,'rendering does not ask for a microphone');
  await page.click('#start');await page.waitForFunction(()=>window.frames.length>=3);
  const captured=await page.evaluate(()=>({states:window.states,format:window.format,frames:window.frames.map(f=>f.length),constraints:window.constraints}));
  assert.equal(captured.states.at(-1),'listening');assert.equal(captured.format.channels,1);
  assert.ok(captured.frames.every(n=>n===Math.round(captured.format.sampleRate*.02)));
  assert.equal(captured.constraints.video,false);
  await page.evaluate(()=>window.device.mute(true));
  const muted=await page.evaluate(()=>({count:window.frames.length,enabled:window.tracks.map(t=>t.enabled),state:window.states.at(-1)}));
  assert.deepEqual(muted.enabled,[false]);assert.equal(muted.state,'muted');
  await page.waitForTimeout(150);assert.equal(await page.evaluate(()=>window.frames.length),muted.count);
  await page.evaluate(()=>window.device.mute(false));await page.waitForFunction(n=>window.frames.length>n,muted.count);
  await page.evaluate(()=>window.device.stop());await page.waitForFunction(()=>window.contexts.every(c=>c.state==='closed'));
  assert.deepEqual(await page.evaluate(()=>window.tracks.map(t=>t.readyState)),['ended']);
  assert.equal(await page.evaluate(()=>window.stops),1);assert.equal(await page.evaluate(()=>window.states.at(-1)),'closed');
});
test('permission denial fails without opening a voice connection',async t=>{
  const page=await pageFor(t);await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('denied','NotAllowedError');};});
  await page.click('#start');await page.waitForFunction(()=>window.states.includes('failed'));
  assert.equal(await page.evaluate(()=>window.format),undefined);
  assert.deepEqual(await page.evaluate(()=>window.failures),['permission-denied']);
  await page.waitForFunction(()=>window.contexts.every(c=>c.state==='closed'));
});
for(const name of ['NotFoundError','NotReadableError','OverconstrainedError'])test(`microphone ${name} has hardware recovery rather than permission advice`,async t=>{
  const page=await pageFor(t);
  await page.evaluate(name=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('private hardware details',name);};},name);
  await page.click('#start');await page.waitForFunction(()=>window.states.includes('failed'));
  assert.deepEqual(await page.evaluate(()=>window.failures),['microphone-unavailable']);
  assert.equal(await page.evaluate(()=>window.format),undefined);
  await page.waitForFunction(()=>window.contexts.every(c=>c.state==='closed'));
});
test('audio output startup failure releases a late microphone grant and explains output recovery',async t=>{
  const page=await pageFor(t);
  await page.evaluate(()=>{window.deferPermission=true;AudioContext.prototype.resume=async()=>{throw Error('private device details');};});
  await page.click('#start');await page.waitForFunction(()=>window.resolvePermission);
  await page.waitForFunction(()=>window.states.includes('failed'));
  assert.deepEqual(await page.evaluate(()=>window.failures),['output-unavailable']);
  await page.evaluate(()=>window.resolvePermission());
  await page.waitForFunction(()=>window.tracks.every(t=>t.readyState==='ended'));
  assert.equal(await page.evaluate(()=>window.format),undefined);
});
test('unsupported browser does not request a microphone',async t=>{
  const page=await pageFor(t);
  await page.evaluate(()=>{window.AudioContext=undefined;});
  await page.click('#start');await page.waitForFunction(()=>window.states.includes('failed'));
  assert.deepEqual(await page.evaluate(()=>window.failures),['unsupported-browser']);
  assert.equal(await page.evaluate(()=>window.tracks.length),0);
  assert.equal(await page.evaluate(()=>window.format),undefined);
});
test('capture setup failure releases acquired audio without attempting a connection',async t=>{
  const page=await pageFor(t);
  await page.evaluate(()=>{AudioContext.prototype.createMediaStreamSource=()=>{throw Error('private capture details');};});
  await page.click('#start');await page.waitForFunction(()=>window.states.includes('failed'));
  assert.deepEqual(await page.evaluate(()=>window.failures),['capture-failed']);
  await page.waitForFunction(()=>window.tracks.every(t=>t.readyState==='ended')&&window.contexts.every(c=>c.state==='closed'));
  assert.equal(await page.evaluate(()=>window.format),undefined);
});
test('cancellation during permission releases a late microphone grant without connecting',async t=>{
  const page=await pageFor(t);await page.evaluate(()=>window.deferPermission=true);await page.click('#start');
  await page.waitForFunction(()=>window.resolvePermission);await page.evaluate(()=>{window.device.stop();window.resolvePermission();});
  await page.waitForFunction(()=>window.tracks.every(t=>t.readyState==='ended'));
  assert.equal(await page.evaluate(()=>window.format),undefined);assert.equal(await page.evaluate(()=>window.states.at(-1)),'closed');
});
test('output suspended while connecting cannot be admitted as a listening session',async t=>{
  const page=await pageFor(t);await page.evaluate(()=>window.deferReady=true);await page.click('#start');
  await page.waitForFunction(()=>window.events);
  await page.evaluate(()=>window.contexts[0].suspend());
  await page.waitForTimeout(50); // Let the suspension event settle before the server acknowledges.
  await page.evaluate(()=>window.events.state('ready'));
  assert.notEqual(await page.evaluate(()=>window.states.at(-1)),'listening');
  assert.deepEqual(await page.evaluate(()=>window.failures),['output-unavailable']);
  await page.waitForFunction(()=>window.contexts.every(c=>c.state==='closed'));
  assert.deepEqual(await page.evaluate(()=>window.tracks.map(t=>t.readyState)),['ended']);
});
for(const reason of ['abort','server failure','pagehide','device ended'])test(`voice cleanup releases all audio resources on ${reason}`,async t=>{
  const page=await pageFor(t);await page.click('#start');await page.waitForFunction(()=>window.frames.length>=1);
  await page.evaluate(reason=>{
    if(reason==='abort')window.abort.abort();
    else if(reason==='server failure')window.events.state('failed');
    else if(reason==='pagehide')window.dispatchEvent(new Event('pagehide'));
    else window.tracks[0].dispatchEvent(new Event('ended'));
  },reason);
  await page.waitForFunction(()=>window.tracks.every(t=>t.readyState==='ended')&&window.contexts.every(c=>c.state==='closed'));
  assert.equal(await page.evaluate(()=>window.stops),1);
  assert.deepEqual(await page.evaluate(()=>window.failures),reason==='server failure'?['connection-failed']:reason==='device ended'?['microphone-disconnected']:[]);
  const count=await page.evaluate(()=>window.frames.length);await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>window.frames.length),count);
});

async function nativeService(t,{interrupt=false}={}){
  const root=path.resolve(__dirname,'..');
  const {pythonLayout}=require('./python-layout.cjs');
  const child=spawn(process.env.SCONE_TEST_PYTHON||path.join(pythonLayout(root).project,'.venv/bin/python'),['-u',path.join(__dirname,'fixtures/voice-server.py'),...(interrupt?['--interrupt']:[])],{cwd:root,stdio:['pipe','pipe','pipe']});
  let logs='';child.stderr.on('data',data=>logs=(logs+data).slice(-6000));const ended=once(child,'close');
  t.after(async()=>{
    if(child.exitCode===null&&child.signalCode===null)child.stdin.end('stop\n');
    const timeout=setTimeout(()=>child.kill('SIGKILL'),5000);
    try{await ended;}finally{clearTimeout(timeout);}
    assert.equal(child.exitCode,0,logs);assert.doesNotMatch(logs,/Traceback \(most recent call last\)/);
  });
  const port=await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(Error('Native voice startup timed out: '+logs)),15000);let output='';
    child.stdout.on('data',data=>{output+=data;const match=output.match(/VOICE_READY (\d+)/);if(match){clearTimeout(timeout);resolve(Number(match[1]));}});
    child.once('error',error=>{clearTimeout(timeout);reject(error);});
    child.once('exit',()=>{clearTimeout(timeout);reject(Error('Native voice exited: '+logs));});
  });
  const origin='http://127.0.0.1:'+port,headers={authorization:'Bearer voice-alpha','content-type':'application/json'};
  for(let i=0;i<100;i++){
    try{const r=await fetch(origin+'/v1/conversations/capabilities',{headers});if(r.ok){await r.body.cancel();break;}}catch{}
    await new Promise(resolve=>setTimeout(resolve,25));
  }
  const catalog=await(await fetch(origin+'/v1/conversations/personas',{headers})).json();
  const response=await fetch(origin+'/v1/conversations',{method:'POST',headers,body:JSON.stringify({request_id:'browser-voice',capture:true,mode:'voice',persona:'guide',persona_fingerprint:catalog.personas[0].fingerprint})});
  assert.equal(response.status,200);const session=await response.json();assert.equal(session.state,'created');
  return {origin,headers,session,logs:()=>logs,read:async suffix=>(await fetch(origin+'/v1/conversations/'+session.session_id+suffix,{headers})).json()};
}
test('native waiting voice session can be inspected without attaching audio',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const native=await nativeService(t),page=await browser.newPage();t.after(()=>page.close());
  let sockets=0;page.on('websocket',()=>{sockets++;});
  await page.goto(native.origin+'/conversations/'+native.session.session_id);
  await page.getByLabel('Scone space key',{exact:true}).fill('voice-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('region',{name:'Voice session',exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).count(),0);
  assert.equal(sockets,0);
  const after=await native.read('');
  assert.equal(after.state,'created');assert.equal(after.mode,'voice');
  assert.equal(after.revision,native.session.revision);
});
test('workspace creates a voice conversation; explicit microphone consent, mute and stop reach native receipts',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const native=await nativeService(t),page=await browser.newPage();t.after(()=>page.close());
  page.setDefaultTimeout(10000);
  await page.addInitScript(()=>{
    window.capturedTracks=[];
    const gum=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia=async options=>{const stream=await gum(options);window.capturedTracks.push(...stream.getTracks());return stream;};
  });
  await page.goto(native.origin+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('voice-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Voice',exact:true}).check();
  await page.getByRole('radio',{name:'Voice guide',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start voice conversation',exact:true}).click();
  await page.getByRole('button',{name:'Enable microphone',exact:true}).waitFor();
  if(process.env.SCONE_VOICE_SCREENSHOTS){
    for(const [label,width,height] of [['desktop',1440,1100],['mobile',390,844]]){
      await page.setViewportSize({width,height});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'voice workspace fits viewport');
      await page.screenshot({path:path.join(process.env.SCONE_VOICE_SCREENSHOTS,`voice-${label}.png`),fullPage:true});
    }
    await page.setViewportSize({width:1440,height:1100});
  }
  assert.equal(await page.evaluate(()=>window.capturedTracks.length),0,'session creation does not acquire a microphone');
  const sid=new URL(page.url()).pathname.split('/').at(-1);
  const read=async suffix=>(await fetch(native.origin+'/v1/conversations/'+sid+suffix,{headers:native.headers})).json();
  assert.equal((await read('')).state,'created');
  await page.getByLabel('Allow live audio with this persona').check();
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.getByRole('region',{name:'Saved messages',exact:true}).getByText('Use Polaris to calibrate Juniper.',{exact:true}).waitFor();
  if(process.env.SCONE_VOICE_SCREENSHOTS)await page.screenshot({path:path.join(process.env.SCONE_VOICE_SCREENSHOTS,'voice-active.png'),fullPage:true});
  await page.getByRole('button',{name:'Mute microphone',exact:true}).click();
  assert.deepEqual(await page.evaluate(()=>window.capturedTracks.map(t=>t.enabled)),[false]);
  await page.getByRole('button',{name:'Unmute microphone',exact:true}).click();
  assert.deepEqual(await page.evaluate(()=>window.capturedTracks.map(t=>t.enabled)),[true]);
  await page.getByRole('button',{name:'Stop audio',exact:true}).click();
  await page.waitForFunction(()=>window.capturedTracks.every(t=>t.readyState==='ended'));
  await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
  assert.equal((await read('')).state,'ended');
  assert.deepEqual((await read('/transcript')).episodes.map(e=>e.metadata.role).sort(),['assistant','user']);
});
async function waitingWorkspace(t){
  const native=await nativeService(t),page=await browser.newPage();t.after(()=>page.close());page.setDefaultTimeout(10000);
  await page.addInitScript(()=>{
    window.voiceTracks=[];window.voiceContexts=[];
    const gum=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia=async options=>{
      if(window.denyAudio)throw new DOMException('denied','NotAllowedError');
      const stream=await gum(options);window.voiceTracks.push(...stream.getTracks());
      if(window.deferAudio)await new Promise(resolve=>window.releaseAudio=resolve);
      return stream;
    };
    const Native=window.AudioContext;window.AudioContext=class extends Native{constructor(...args){super(...args);window.voiceContexts.push(this);}};
  });
  await page.goto(native.origin+'/conversations/'+native.session.session_id);
  await page.getByLabel('Scone space key',{exact:true}).fill('voice-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByLabel('Allow live audio with this persona').check();
  return {native,page};
}
test('workspace End conversation releases audio before an uncertain HTTP stop is acknowledged',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);
  let release,received=false;const gate=new Promise(resolve=>release=resolve);t.after(()=>release());
  await page.route('**/v1/conversations/*/stop',async route=>{received=true;await gate;await route.fulfill({status:503,contentType:'application/json',body:'{}'});});
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.getByRole('region',{name:'Saved messages',exact:true}).getByText('Use Polaris to calibrate Juniper.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.waitForFunction(()=>window.voiceTracks.length&&window.voiceTracks.every(t=>t.readyState==='ended')&&window.voiceContexts.every(c=>c.state==='closed'),null,{timeout:1500});
  assert.equal(received,true);release();
  await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
  assert.equal((await native.read('')).state,'ended');
});
test('workspace permission denial is recoverable without an automatic audio connection',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);let sockets=0;page.on('websocket',()=>sockets++);
  await page.evaluate(()=>window.denyAudio=true);
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('status').filter({hasText:'Microphone permission denied'}).waitFor();
  assert.match(await page.getByRole('region',{name:'Voice session',exact:true}).innerText(),/browser.*permission/i);
  assert.equal(sockets,0);assert.equal((await native.read('')).state,'created');
  await page.waitForFunction(()=>window.voiceContexts.every(c=>c.state==='closed'));
  await page.evaluate(()=>window.denyAudio=false);
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.getByRole('button',{name:'Stop audio',exact:true}).click();
  await page.waitForFunction(()=>window.voiceTracks.every(t=>t.readyState==='ended'));
});

test('workspace hardware failure gives actionable safe instructions without starting a session',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);let sockets=0;page.on('websocket',()=>sockets++);
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('private device identifier','NotReadableError');};});
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  const voice=page.getByRole('region',{name:'Voice session',exact:true});
  await voice.getByText('Microphone unavailable',{exact:true}).waitFor();
  const text=await voice.innerText();
  assert.match(text,/plugged in/i);assert.match(text,/another app/i);assert.doesNotMatch(text,/private device identifier/);
  assert.equal(sockets,0);assert.equal((await native.read('')).state,'created');
  assert.equal(await page.getByRole('button',{name:'Enable microphone',exact:true}).isEnabled(),true);
});

for(const failure of ['microphone','output'])test(`workspace preserves ${failure} recovery after the native session closes`,{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);let sockets=0;page.on('websocket',()=>sockets++);
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.evaluate(failure=>{
    if(failure==='microphone')window.voiceTracks[0].dispatchEvent(new Event('ended'));
    else void window.voiceContexts[0].suspend();
  },failure);
  await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
  const voice=page.getByRole('region',{name:'Voice session',exact:true});
  const text=await voice.innerText();
  assert.match(text,failure==='microphone'?/Reconnect it before starting a new voice conversation/:/Check your audio output/);
  assert.match(text,/session is closed/i);
  assert.equal(sockets,1);assert.equal((await native.read('')).state,'ended');
  assert.equal(await page.getByRole('button',{name:'Enable microphone',exact:true}).count(),0);
  if(process.env.SCONE_VOICE_SCREENSHOTS){
    for(const [label,width,height] of [['desktop',1440,1000],['mobile',390,844]]){
      await page.setViewportSize({width,height});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.screenshot({path:path.join(process.env.SCONE_VOICE_SCREENSHOTS,`voice-${failure}-recovery-${label}.png`),fullPage:true});
    }
  }
});
test('workspace verification failure releases audio and never reconnects automatically',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);let sockets=0;page.on('websocket',()=>sockets++);
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.route('**/v1/conversations/'+native.session.session_id,route=>route.fulfill({status:503,contentType:'application/json',body:'{}'}));
  await page.waitForFunction(()=>window.voiceTracks.length&&window.voiceTracks.every(t=>t.readyState==='ended')&&window.voiceContexts.every(c=>c.state==='closed'));
  await page.unroute('**/v1/conversations/'+native.session.session_id);
  await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
  assert.equal(sockets,1);assert.equal(await page.getByRole('button',{name:'Enable microphone',exact:true}).count(),0);
});
test('uncertain voice creation freezes mode and persona and retries the same native request',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {page}=await waitingWorkspace(t),bodies=[];
  await page.route('**/v1/conversations',async route=>{
    if(route.request().method()!=='POST')return route.continue();
    bodies.push(route.request().postDataJSON());
    if(bodies.length===1){await route.fetch();return route.fulfill({status:503,contentType:'application/json',body:'{}'});}
    return route.continue();
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Voice',exact:true}).check();
  await page.getByRole('radio',{name:'Voice guide',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start voice conversation',exact:true}).click();
  await page.getByText(/Start was not confirmed/).waitFor();
  assert.equal(await page.getByRole('radio',{name:'Text',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('radio',{name:'Voice guide',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Start voice conversation',exact:true}).click();
  await page.getByRole('dialog',{name:'Start a conversation'}).waitFor({state:'hidden'});
  await page.getByRole('button',{name:'Enable microphone',exact:true}).waitFor();
  assert.equal(bodies.length,2);assert.deepEqual(bodies[0],bodies[1]);assert.equal(bodies[0].mode,'voice');
  assert.equal(await page.evaluate(()=>window.voiceTracks.length),0);
});
test('browsing older saved messages does not disconnect an active microphone',{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.route('**/v1/conversations/*/transcript?*',async route=>{
    const older=new URL(route.request().url()).searchParams.has('before');
    const episodes=Array.from({length:older?1:50},(_,i)=>({episode_id:i+100,content:older?'Older retained voice message':`Retained voice message ${i}`,metadata:{role:'user'}}));
    await route.fulfill({json:{episodes,has_more:!older,next_before:older?null:'older_cursor'}});
  });
  const older=page.getByRole('button',{name:'Older messages',exact:true});
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Older messages'&&!b.disabled));
  await older.click();await page.getByText('Older retained voice message',{exact:true}).waitFor();
  assert.deepEqual(await page.evaluate(()=>window.voiceTracks.map(t=>t.readyState)),['live'],await page.getByRole('region',{name:'Voice session',exact:true}).innerText());
  assert.equal((await native.read('')).state,'running');
  await page.getByRole('button',{name:'Stop audio',exact:true}).click();
});
for(const pending of [false,true])test(`workspace navigation releases audio, including a late permission grant: pending=${pending}`,{skip:!process.env.SCONE_CONVERSATIONS_HTML},async t=>{
  const {native,page}=await waitingWorkspace(t);
  await page.evaluate(pending=>window.deferAudio=pending,pending);
  await page.getByRole('button',{name:'Enable microphone',exact:true}).click();
  if(pending)await page.waitForFunction(()=>window.releaseAudio);
  else await page.getByRole('button',{name:'Mute microphone',exact:true}).waitFor();
  await page.getByRole('link',{name:'Memory',exact:true}).click();
  if(pending)await page.evaluate(()=>window.releaseAudio());
  await page.waitForFunction(()=>window.voiceTracks.length&&window.voiceTracks.every(t=>t.readyState==='ended')&&window.voiceContexts.every(c=>c.state==='closed'));
  if(pending)assert.equal((await native.read('')).state,'created');
});
test('production browser microphone → native voice → PCM playback → saved transcript',async t=>{
  const native=await nativeService(t),page=await pageFor(t);page.setDefaultTimeout(10000);
  await page.evaluate(({origin,session})=>{window.nativeBase=origin;window.nativeSession=session.session_id;},{origin:native.origin,session:native.session});
  await page.click('#start');await page.waitForFunction(()=>window.outputs.length>0);
  const output=await page.evaluate(()=>window.outputs[0]);
  assert.deepEqual(output,{rate:16000,channels:1,sample:.25},'the native PCM reaches an actual AudioBufferSourceNode');
  let transcript;
  for(let i=0;i<100;i++){transcript=await native.read('/transcript');if(transcript.episodes.length>=2)break;await page.waitForTimeout(20);}
  assert.deepEqual(transcript.episodes.map(e=>e.content).sort(),['How is Juniper calibrated?','Use Polaris to calibrate Juniper.'].sort());
  assert.deepEqual(transcript.episodes.map(e=>e.metadata.role).sort(),['assistant','user']);
  await page.evaluate(()=>window.device.stop());
  let receipt;
  for(let i=0;i<100;i++){receipt=await native.read('');if(receipt.state!=='running')break;await page.waitForTimeout(20);}
  assert.equal(receipt.state,'ended',native.logs());
  assert.deepEqual(await page.evaluate(()=>window.tracks.map(t=>t.readyState)),['ended']);
});
test('native cross-space refusal releases the microphone and never starts the waiting session',async t=>{
  const native=await nativeService(t),page=await pageFor(t);page.setDefaultTimeout(10000);
  await page.evaluate(({origin,session})=>{window.nativeBase=origin;window.nativeSession=session.session_id;window.nativeKey='voice-beta';},{origin:native.origin,session:native.session});
  await page.click('#start');await page.waitForFunction(()=>window.states.includes('failed'));
  assert.equal(await page.evaluate(()=>window.outputs.length),0);
  assert.deepEqual(await page.evaluate(()=>window.tracks.map(t=>t.readyState)),['ended']);
  assert.equal((await native.read('')).state,'created');assert.equal((await native.read('/transcript')).episodes.length,0);
});
test('native barge-in stops queued browser speech and retains only the completed replacement reply',async t=>{
  const native=await nativeService(t,{interrupt:true}),page=await pageFor(t);page.setDefaultTimeout(10000);
  await page.evaluate(({origin,session})=>{window.nativeBase=origin;window.nativeSession=session.session_id;},{origin:native.origin,session:native.session});
  await page.click('#start');await page.waitForFunction(()=>window.outputs.length===2);
  assert.equal(await page.evaluate(()=>window.stoppedNodes.includes(window.audioNodes[0])),true,'the first actual audio node was stopped, not merely hidden in the UI');
  assert.equal(await page.evaluate(()=>window.audioNodes[0].buffer===null),true,'interrupted PCM is released');
  let transcript;
  for(let i=0;i<100;i++){transcript=await native.read('/transcript');if(transcript.episodes.length>=3)break;await page.waitForTimeout(20);}
  assert.deepEqual(transcript.episodes.map(e=>e.content).sort(),['How is Juniper calibrated?','Make that shorter.','Use Polaris.'].sort());
  const assistant=transcript.episodes.find(e=>e.metadata.role==='assistant');
  assert.equal(assistant.metadata.playback,'unverified','scheduling browser audio must not rewrite server playback evidence');
  await page.evaluate(()=>window.device.stop());
  await page.waitForFunction(()=>window.tracks.every(t=>t.readyState==='ended')&&window.contexts.every(c=>c.state==='closed'));
  let receipt;
  for(let i=0;i<100;i++){receipt=await native.read('');if(receipt.state!=='running')break;await page.waitForTimeout(20);}
  assert.equal(receipt.state,'ended');
});
