// Isolated HTTP fixtures exercise the built React app. No live Scone/model calls.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const playwright=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const browserName=process.env.SCONE_BROWSER_ENGINE||'chromium';
assert.ok(['chromium','firefox','webkit'].includes(browserName),'Unsupported browser engine');
let browser;
before(async()=>{browser=await playwright[browserName].launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,...(browserName==='chromium'?{args:['--disable-gpu']}: {})});});
after(async()=>{await browser?.close();});

for(const mobile of [false,true])test(`product navigation remains readable and shares the workspace surface, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{mobile});
  for(const route of ['/memory','/playground','/conversations','/learn']){
    await page.goto(new URL(route,page.url()).href);
    await page.locator('h1').waitFor();
    const nav=page.getByRole('navigation',{name:'Workspace',exact:true});
    const surface=await page.locator('.server-strip').evaluate(el=>getComputedStyle(el).backgroundColor);
    assert.equal(await page.locator('.topbar').evaluate(el=>getComputedStyle(el).backgroundColor),surface,'Product navigation must not introduce an unrelated theme.');
    for(const link of await nav.getByRole('link').all()){
      assert.ok(await link.evaluate(el=>parseFloat(getComputedStyle(el).fontSize)>=12),'Destination labels must remain readable without page zoom.');
      assert.equal(await link.evaluate(el=>el.scrollWidth<=el.clientWidth),true,'Full destination names must fit without clipping.');
      const box=await link.boundingBox();assert.ok(box.height>=44,'Navigation has a usable touch target.');
    }
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  }
});

test('workspace pages keep the same title hierarchy and conversation rail alignment',async t=>{
  const {page}=await fixture(t);
  const styles=[];
  for(const route of ['/memory','/playground','/conversations','/learn']){
    await page.goto(new URL(route,page.url()).href);
    await page.locator('h1').waitFor();
    styles.push(await page.locator('h1').evaluate(el=>{
      const s=getComputedStyle(el);
      return {size:s.fontSize,weight:s.fontWeight,line:s.lineHeight,color:s.color};
    }));
    if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`frame-${route.slice(1)}.png`),fullPage:true});
  }
  for(const style of styles)assert.deepEqual(style,styles[0],'Page navigation must not change the product title scale.');
  await page.goto(new URL('/conversations',page.url()).href);
  const rail=await page.locator('.conversation-list').boundingBox();
  const strip=await page.locator('.server-strip').boundingBox();
  const heading=await page.locator('h1').boundingBox();
  assert.equal(rail.x,strip.x,'Session navigation belongs to the working frame, not an inset card.');
  assert.ok(Math.abs(rail.y-(strip.y+strip.height))<2,'Rail must meet the application frame.');
  assert.ok(heading.x>rail.x+rail.width,'Heading belongs to the working pane, not above session navigation.');
});

test('application navigation and connection remain reachable while a long workspace scrolls',async t=>{
  const {page}=await fixture(t);
  await page.goto(new URL('/learn/graph-memory',page.url()).href);
  await page.locator('h1').waitFor();
  await page.evaluate(()=>scrollTo(0,document.body.scrollHeight));
  const nav=page.getByRole('navigation',{name:'Workspace',exact:true});
  for(const link of await nav.getByRole('link').all()){
    const box=await link.boundingBox();
    assert.ok(box.y>=0&&box.y+box.height<=900,'Primary destinations remain visible while reading a long page.');
  }
  const connect=page.getByRole('button',{name:'Memory connection',exact:true});
  const box=await connect.boundingBox();
  assert.ok(box.y>=0&&box.y+box.height<=900,'Connection settings remain reachable without scrolling to the top.');
  await connect.click();
  await page.getByRole('dialog').waitFor();
  assert.match(await page.getByRole('dialog').innerText(),/Memory connection/);
});

for(const mobile of [false,true])test(`keyboard skip link stays above the application frame, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{mobile});
  await page.keyboard.press('Tab');
  const skip=page.getByRole('link',{name:'Skip to workspace',exact:true});
  assert.equal(await skip.evaluate(el=>el===document.activeElement),true);
  assert.equal(await skip.evaluate(el=>{
    const box=el.getBoundingClientRect();
    return el.contains(document.elementFromPoint(box.x+box.width/2,box.y+box.height/2));
  }),true,'The focused skip link must not be painted underneath sticky navigation.');
  const before=page.url();
  await page.keyboard.press('Enter');
  assert.equal(page.url(),before);
  assert.equal(await page.evaluate(()=>document.activeElement?.id),'main');
  assert.equal(await page.locator('#main').count(),1);
});

test('conversation frame keeps session controls and evidence usable across viewport sizes',async t=>{
  const {page}=await fixture(t);
  for(const width of [390,768,1024,1280,1600]){
    await page.setViewportSize({width,height:900});
    await page.getByRole('link',{name:/previous ended/}).click();
    await page.getByText('Earlier conversation.',{exact:true}).waitFor();
    const bounds=await page.evaluate(()=>{
      const box=selector=>{const r=document.querySelector(selector).getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width};};
      return {overflow:document.documentElement.scrollWidth>innerWidth,rail:box('.conversation-list'),thread:box('.conversation-thread'),evidence:box('.conversation-evidence')};
    });
    assert.equal(bounds.overflow,false,`No page overflow at ${width}px`);
    assert.ok(bounds.thread.width>=Math.min(width-40,320),`Transcript remains readable at ${width}px`);
    assert.ok(bounds.evidence.y>=bounds.thread.bottom-1||bounds.evidence.x>=bounds.thread.right-1,`Evidence never overlaps the transcript at ${width}px`);
    if(process.env.SCONE_SCREENSHOT_DIR&&[390,1440,1600].includes(width))await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`workspace-conversation-${width}.png`),fullPage:true});
  }
});
test('transcript retains readable height with session controls in short and narrow windows',async t=>{
  const {page}=await fixture(t,{pagination:true,scoped:true,defaultPersona:true});
  await page.getByRole('link',{name:/previous ended/}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  for(const viewport of [{width:1440,height:720},{width:1440,height:560},{width:1024,height:768},{width:768,height:600},{width:390,height:664}]){
    await page.setViewportSize(viewport);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const metrics=await page.locator('.conversation-messages').evaluate(el=>({height:el.clientHeight,minimum:15*parseFloat(getComputedStyle(document.documentElement).fontSize),overflow:document.documentElement.scrollWidth>innerWidth}));
    assert.ok(metrics.height>=metrics.minimum,`Transcript collapsed to ${metrics.height}px at ${JSON.stringify(viewport)}`);
    assert.equal(metrics.overflow,false);
    const composer=page.locator('.conversation-composer');
    await composer.scrollIntoViewIfNeeded();
    const bounds=await composer.boundingBox();
    assert.ok(bounds.y>=0&&bounds.y+bounds.height<=viewport.height+1,'Composer remains reachable');
  }
  const heightAt=async height=>{
    await page.setViewportSize({width:1440,height});
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    return page.locator('.conversation-messages').evaluate(el=>el.clientHeight);
  };
  const shorter=await heightAt(1400),taller=await heightAt(1600);
  assert.ok(Math.abs(taller-shorter-200)<=2,'Transcript must grow with the actual available viewport');
  await page.locator('.conversation-workspace').evaluate(el=>{
    el.style.height='2400px';
    const notice=document.createElement('div');notice.className='conversation-notice';
    notice.textContent='Connection restored';el.querySelector('.conversation-layout').before(notice);
  });
  await page.waitForFunction(previous=>document.querySelector('.conversation-messages').clientHeight<previous,taller);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`conversation-sizing-${browserName}.png`),fullPage:true});
});
for(const voiceState of ['created','running','ended'])test(`saved voice session preserves evidence without text controls or microphone acquisition: ${voiceState}`,async t=>{
  const {page,posts}=await fixture(t,{voiceState});
  let microphoneRequests=0;page.on('websocket',()=>{microphoneRequests++;});
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=()=>{throw Error('Unexpected microphone request');};});
  await page.getByRole('link',{name:new RegExp(`previous ${voiceState}`)}).click();
  await page.getByRole('region',{name:'Voice session',exact:true}).waitFor();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).count(),0);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).count(),0);
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  await page.getByRole('complementary',{name:'Conversation evidence'}).locator('.source-markdown').getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(microphoneRequests,0);assert.equal(posts(),0);
  if(voiceState==='running'){
    await page.getByRole('button',{name:'End conversation',exact:true}).click();
    await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
    assert.equal(await page.getByLabel('Message',{exact:true}).count(),0);
  }
});
async function fixture(t,{unavailable=false,mobile=false,uncertain=false,reject=false,unknown=false,recovered=false,deletion=false,cancellation=false,pagination=false,scoped=false,streaming=false,capStatus=200,listFailure=false,holdCapabilities=false,holdFirstList=false,personaCatalog,personaStatus=200,voiceState,sourceResponse,defaultPersona=false}={}){
  const html=fs.readFileSync(process.env.SCONE_CONVERSATIONS_HTML||path.resolve(__dirname,'../crates/scone/src/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','fixture-key');
  const sessions=[{session_id:'previous',space:'alpha',state:unavailable?'running':'ended',revision:4,created_at:'2026-09-06T10:00:00Z',active_request_id:null,...(defaultPersona?{persona:null}:{}),...(recovered?{latest_request_id:'a-newer'}:{})}];
  if(voiceState){sessions[0].mode='voice';sessions[0].state=voiceState;}
  const saved={previous:[{episode_id:2,content:'Earlier conversation.',metadata:{role:'user'}}]};
  if(pagination)saved.previous=Array.from({length:123},(_,i)=>({episode_id:i+1,content:`Saved message ${i+1}`,metadata:{role:i%2?'assistant':'user'}}));
  let turn=null,posts=0,checks=0;const requested=[],creates=new Map();
  const streams=new Set(),chunks=[];
  let releaseCapabilities;const capabilityGate=new Promise(resolve=>{releaseCapabilities=resolve;});
  let releaseList;const listGate=new Promise(resolve=>{releaseList=resolve;});
  const frame=(sequence,text)=>`event: text\nid: ${sequence}\ndata: ${JSON.stringify({sequence,text,provisional:true})}\n\n`;
  function emit(text){chunks.push(text);for(const stream of streams)stream.write(frame(chunks.length,text));}
  function finish(status='completed',availability='available'){
    turn.status=status;turn.result_state=status==='completed'?availability:'unavailable';
    if(status==='completed'&&availability==='available'){
      turn.result={text:'Saved complete answer.',user_episode_id:10,assistant_episode_id:11};
      saved.current.push({episode_id:11,content:turn.result.text,metadata:{role:'assistant'}});
    }else turn.result=null;
    const value=sessions.find(s=>s.session_id==='current');value.active_request_id=null;value.latest_request_id=turn.request_id;
    for(const stream of streams)stream.end(`event: terminal\ndata: ${JSON.stringify({request_id:turn.request_id,status,read_receipt:true})}\n\n`);
  }
  const server=http.createServer(async(req,res)=>{
    requested.push(req.url);res.setHeader('content-type','application/json');
    if(!req.url.startsWith('/v1/')){res.setHeader('content-type','text/html');return res.end(html);}
    assert.equal(req.headers.authorization,'Bearer fixture-key');
    if(req.url==='/v1/status')return res.end('{"space":"alpha"}');
    if(req.url==='/v1/conversations/capabilities'){
      if(holdCapabilities)await capabilityGate;
      if(capStatus!==200){res.statusCode=capStatus;return res.end('{"detail":"unavailable"}');}
      return res.end(JSON.stringify({schema_version:unknown?999:1,text_configured:!unavailable,reply_transport:'poll',reply_replay:'process_lifetime',session_deletion:deletion,turn_cancellation:cancellation,transcript_pagination:pagination,recall_scope:scoped,streaming,text_stream:streaming?{transport:'sse',replay:'active_window',max_bytes:65536,max_chunks:256}:null,...(personaCatalog?{personas:personaCatalog.length}:{})}));
    }
    if(req.url==='/v1/conversations/personas'){res.statusCode=personaStatus;return res.end(JSON.stringify({schema_version:1,...(personaCatalog?.some(p=>p.fingerprint)?{revision:'abcdef0123456789'}:{}),personas:personaCatalog}));}
    if(streaming&&/\/stream\?after=\d+$/.test(req.url)){
      assert.equal(req.method,'GET');assert.equal(req.headers.accept,'text/event-stream');
      res.setHeader('content-type','text/event-stream');res.flushHeaders();streams.add(res);res.on('close',()=>streams.delete(res));
      if(turn&&turn.status!=='pending')return res.end(`event: terminal\ndata: ${JSON.stringify({request_id:turn.request_id,status:turn.status,read_receipt:true})}\n\n`);
      const after=Number(new URL(req.url,'http://fixture').searchParams.get('after'));
      for(let i=after;i<chunks.length;i++)res.write(frame(i+1,chunks[i]));
      return;
    }
    let body='';for await(const chunk of req)body+=chunk;
    const data=body?JSON.parse(body):null;
    if(req.url==='/v1/conversations'&&req.method==='POST'){
      assert.equal(data.capture,true);if(creates.has(data.request_id))return res.end(JSON.stringify(creates.get(data.request_id)));
      const persona=personaCatalog?.find(item=>item.id===data.persona);
      if(personaCatalog&&!persona){res.statusCode=422;return res.end('{"error":"persona required"}');}
      const value={...sessions[0],session_id:'current',state:'running',revision:2,...(scoped?{recall_scope:data.recall_scope||{}}:{}),...(persona?{persona:{id:persona.id,name:persona.name,...(persona.fingerprint?{fingerprint:persona.fingerprint,current:true}:{})}}:{})};creates.set(data.request_id,value);sessions.push(value);saved.current=[];return res.end(JSON.stringify(value));
    }
    if(req.url.startsWith('/v1/conversations?')){
      if(listFailure){res.statusCode=503;return res.end('{"error":"list unavailable"}');}
      const response=JSON.stringify({items:sessions,has_more:false,next_after:null});
      if(holdFirstList){holdFirstList=false;await listGate;}
      return res.end(response);
    }
    if(req.url==='/v1/episodes/7')return res.end(JSON.stringify({episode_id:7,content:'Juniper is calibrated with Polaris. <script>not executable</script>',metadata:{},attachments:[]}));
    if(req.url==='/v1/episodes/2'){
      if(sourceResponse)return sourceResponse(req,res);
      return res.end(JSON.stringify({episode_id:2,content:'Earlier conversation.',metadata:{role:'user'},attachments:[]}));
    }
    const match=req.url.match(/^\/v1\/conversations\/([^/]+)(.*)$/);
    if(match){
      const value=sessions.find(s=>s.session_id===match[1]);if(!value){res.statusCode=404;return res.end('{}');}
      if(!match[2]&&req.method==='DELETE'){sessions.splice(sessions.indexOf(value),1);delete saved[value.session_id];res.statusCode=204;return res.end();}
      if(!match[2])return res.end(JSON.stringify(value));
      if(match[2].split('?')[0]==='/transcript'){
        const query=new URL(req.url,'http://fixture').searchParams,limit=Number(query.get('limit')||200),before=query.get('before');
        const records=(saved[value.session_id]||[]).filter(e=>!before||e.episode_id<Number(before.replace('cursor-',''))),episodes=records.slice(-limit),has_more=records.length>limit;
        return res.end(JSON.stringify({episodes,has_more,...(pagination?{next_before:has_more?'cursor-'+episodes[0].episode_id:null}:{})}));
      }
      if(match[2]==='/stop'){value.state='ended';value.revision=4;return res.end(JSON.stringify(value));}
      if(match[2]==='/turns'&&req.method==='POST'){
        posts++;if(reject){res.statusCode=429;return res.end('{"error":"capacity reached"}');}turn={request_id:data.request_id,status:'pending'};value.active_request_id=data.request_id;
        saved.current=[{episode_id:10,content:data.text,metadata:{role:'user'}}];
        res.statusCode=uncertain?503:202;return res.end(JSON.stringify(uncertain?{error:'Delivery uncertain'}:turn));
      }
      if(match[2].startsWith('/turns/')){
        if(match[2].endsWith('/cancel')&&req.method==='POST'){turn.status='cancelled';turn.result=null;turn.result_state='unavailable';value.active_request_id=null;value.latest_request_id=turn.request_id;return res.end(JSON.stringify(turn));}
        if(turn?.status==='cancelled')return res.end(JSON.stringify(turn));
        if(recovered&&value.session_id==='previous')return res.end(JSON.stringify({request_id:'a-newer',status:'completed',result_state:'forgotten',result:null}));
        if(!turn){res.statusCode=404;return res.end('{"error":"receipt unavailable"}');}
        checks++;if(checks>1&&!streaming){turn.status='completed';turn.result={text:'Use Polaris.',user_episode_id:10,assistant_episode_id:11,provider_completion:'unverified',memory_context:{status:'prepared',references:[{episode_id:7,chunk_id:8}]}};value.active_request_id=null;saved.current=[saved.current[0],{episode_id:11,content:'Use Polaris.',metadata:{role:'assistant'}}];}
        return res.end(JSON.stringify(turn));
      }
    }
    res.statusCode=404;res.end('{"error":"not found"}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000},reducedMotion:'reduce'});
  page.setDefaultTimeout(4000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{releaseCapabilities();releaseList();await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/conversations`);
  return {page,requested,posts:()=>posts,emit,finish,streams,releaseCapabilities,releaseList,recoverService:()=>{capStatus=200;},recoverList:()=>{listFailure=false;},recoverPersonas:()=>{personaStatus=200;},endExternally:()=>{const value=sessions.find(s=>s.session_id==='current');value.state='ended';value.revision++;}};
}

test('source inspector rejects text returned for a different episode',async t=>{
  const {page}=await fixture(t,{sourceResponse:(_req,res)=>res.end(JSON.stringify({episode_id:3,content:'Wrong source text',metadata:{},attachments:[]}))});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.getByRole('alert').waitFor({timeout:2000});
  assert.equal(await evidence.getByText('Wrong source text',{exact:true}).count(),0);
  assert.equal(await evidence.getByRole('button',{name:'View source images',exact:true}).count(),0);
});

test('source inspector rejects attachment metadata for a different episode before downloading',async t=>{
  let reads=0;
  const {page,requested}=await fixture(t,{sourceResponse:(_req,res)=>res.end(JSON.stringify({episode_id:++reads===1?2:3,content:'Selected source',metadata:{},attachments:[{attachment_id:'a'.repeat(64),media_type:'image/png',bytes:10,filename:'wrong-source.png'}]}))});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  await page.getByRole('button',{name:'View source images',exact:true}).click();
  const media=page.getByRole('region',{name:'Source images'});
  await media.getByRole('button',{name:'Retry source images',exact:true}).waitFor({timeout:2000});
  assert.equal(await media.getByText('wrong-source.png',{exact:true}).count(),0);
  assert.equal(requested.some(p=>p.startsWith('/v1/attachments/')),false);
});

for(const status of [404,410,503])test(`source inspector retries a cached ${status} without changing the selection`,async t=>{
  let reads=0;
  const {page}=await fixture(t,{sourceResponse:(_req,res)=>{
    res.setHeader('Cache-Control','private, max-age=3600');
    if(++reads===1){res.statusCode=status;return res.end('{"error":"private diagnostic"}');}
    return res.end(JSON.stringify({episode_id:2,content:'Recovered original',metadata:{},attachments:[]}));
  }});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.getByRole('alert').waitFor();
  const message=await evidence.getByRole('alert').innerText();
  assert.match(message,status===410?/forgotten/i:status===404?/unavailable/i:/could not be loaded/i);
  assert.doesNotMatch(message,/private diagnostic/);
  await evidence.getByRole('button',{name:'Retry source',exact:true}).click({timeout:2000});
  await evidence.locator('.source-markdown').getByText('Recovered original',{exact:true}).waitFor({timeout:2000});
  assert.equal(reads,2);
});

test('source inspector clears the previous original while another source loads',async t=>{
  const {page}=await fixture(t);
  await page.route('**/previous/transcript*',route=>route.fulfill({json:{episodes:[{episode_id:2,content:'First message',metadata:{}},{episode_id:7,content:'Second message',metadata:{}}],has_more:false}}));
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.locator('.source-markdown').getByText('Earlier conversation.',{exact:true}).waitFor();
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  let entered,settled;
  const intercepted=new Promise(resolve=>{entered=resolve;}),finished=new Promise(resolve=>{settled=resolve;});
  await page.route('**/v1/episodes/7',async route=>{entered();await gate;try{await route.continue();}catch{}finally{settled();}});
  await page.getByRole('button',{name:'Inspect message episode 7',exact:true}).click();
  await intercepted;
  await evidence.getByRole('status').waitFor();
  assert.equal(await evidence.locator('.source-markdown').count(),0);
  assert.equal(await evidence.getByRole('button',{name:'View source images',exact:true}).count(),0);
  const cancelled=page.waitForEvent('requestfailed',request=>request.url().endsWith('/v1/episodes/7'));
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  await evidence.locator('.source-markdown').getByText('Earlier conversation.',{exact:true}).waitFor();
  release();await finished;await cancelled;
  assert.equal(await evidence.locator('.source-markdown').textContent(),'Earlier conversation.');
});

for(const mobile of [false,true])test(`source inspector retains exact Markdown alongside its formatted original, mobile=${mobile}`,async t=>{
  const content='**Calibration** uses `Polaris`.\n\n- Keep the source.\n\n![remote](https://example.invalid/image.png)';
  const {page}=await fixture(t,{mobile,sourceResponse:(_req,res)=>res.end(JSON.stringify({episode_id:2,content,metadata:{},attachments:[]}))});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByRole('button',{name:'Inspect message episode 2',exact:true}).click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.locator('strong').getByText('Calibration',{exact:true}).waitFor();
  assert.equal(await evidence.locator('code').textContent(),'Polaris');
  assert.equal(await evidence.locator('img').count(),0);
  await evidence.getByText('View original Markdown',{exact:true}).click();
  assert.equal(await evidence.getByLabel('Original source text').textContent(),content);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR){
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`bound-source-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  }
});

const personas=[
  {id:'guide',name:'Research guide',reply:{provider:'local',model:'atlas'},transcription:{provider:'deepgram',model:'nova'},speech:{provider:'cartesia',model:'sonic',voice:'calm'},activity:{provider:'silero',model:'vad'},text_ready:true,voice_ready:false},
  {id:'coach',name:'Writing coach',reply:{provider:'alternate',model:'scribe'},transcription:{provider:'deepgram',model:'nova'},speech:{provider:'elevenlabs',model:'multilingual',voice:'warm'},activity:null,text_ready:true,voice_ready:false},
  {id:'offline',name:'Unavailable persona',reply:{provider:'local',model:'absent'},transcription:null,speech:null,activity:null,text_ready:false,voice_ready:false},
];

for(const mobile of [false,true])test(`persona choice is explicit, labelled and retained after reopening, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{personaCatalog:personas,mobile});const starts=[];
  page.on('request',r=>{if(r.method()==='POST'&&new URL(r.url()).pathname==='/v1/conversations')starts.push(r.postDataJSON());});
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('radio',{name:'Writing coach',exact:true}).waitFor();
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  assert.equal(await dialog.getByRole('button',{name:'Start text conversation',exact:true}).isDisabled(),true);
  assert.equal(await dialog.getByRole('radio',{name:'Unavailable persona',exact:true}).isDisabled(),true);
  await dialog.getByRole('radio',{name:'Writing coach',exact:true}).check();
  await dialog.getByText('elevenlabs / multilingual · warm',{exact:true}).waitFor();
  const palette=await dialog.evaluate(el=>{
    const root=getComputedStyle(document.documentElement);
    const selected=el.querySelector('.persona-option.is-selected');
    return {
      selected:getComputedStyle(selected).backgroundColor,
      radio:getComputedStyle(selected.querySelector('input')).accentColor,
      primary:getComputedStyle(el.querySelector('button.primary')).backgroundColor,
      detail:getComputedStyle(el.querySelector('.persona-detail')).backgroundColor,
      // Resolve tokens as CSS colors, rather than assuming a particular theme.
      soft:root.getPropertyValue('--accent-soft').trim(),
      raised:root.getPropertyValue('--raised').trim(),
    };
  });
  const resolved=await page.evaluate(({soft,raised})=>{
    const probe=document.createElement('span');document.body.append(probe);
    probe.style.backgroundColor=soft;const selected=getComputedStyle(probe).backgroundColor;
    probe.style.backgroundColor=raised;const detail=getComputedStyle(probe).backgroundColor;
    probe.remove();return {selected,detail};
  },palette);
  assert.equal(palette.radio,palette.primary,'persona selection and primary actions use the same accent');
  assert.equal(palette.selected,resolved.selected,'selected personas use the shared selection surface');
  assert.equal(palette.detail,resolved.detail,'provider details use the shared raised surface');
  assert.equal(await page.getByRole('button',{name:/microphone|start voice/i}).count(),0);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`persona-picker-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  const action=dialog.getByRole('button',{name:'Start text conversation',exact:true});
  await action.scrollIntoViewIfNeeded();
  const target=await action.boundingBox();
  assert.ok(target&&target.height>=44,'the primary touch target must be at least 44px tall');
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Session persona',{exact:true}).getByText('Writing coach',{exact:true}).waitFor();
  assert.equal(starts.length,1);assert.equal(starts[0].persona,'coach');
  await page.reload();
  await page.getByLabel('Session persona',{exact:true}).getByText('Writing coach',{exact:true}).waitFor();
  assert.equal(starts.length,1,'reopening must not recreate or choose a different persona');
});

test('persona catalog retry is read-only and never falls back to an unchosen provider',async t=>{
  const {page,recoverPersonas}=await fixture(t,{personaCatalog:personas,personaStatus:503});const starts=[];
  page.on('request',r=>{if(r.method()==='POST')starts.push(r.url());});
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('button',{name:'Retry personas',exact:true}).waitFor();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  assert.equal(await page.getByRole('button',{name:'Start text conversation',exact:true}).isDisabled(),true);
  recoverPersonas();await page.getByRole('button',{name:'Retry personas',exact:true}).click();
  await page.getByRole('radio',{name:'Research guide',exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Start text conversation',exact:true}).isDisabled(),true);
  assert.deepEqual(starts,[]);
});

test('unconfirmed persona acknowledgement freezes selection and retries the identical start',async t=>{
  const {page}=await fixture(t,{personaCatalog:personas});const bodies=[];let mismatch=true;
  await page.route('**/v1/conversations',async route=>{
    if(route.request().method()!=='POST')return route.continue();
    bodies.push(route.request().postDataJSON());
    const response=await route.fetch(),body=await response.json();
    await route.fulfill({response,json:mismatch?{...body,persona:{id:'guide',name:'Research guide'}}:body});
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Writing coach',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByText(/Start was not confirmed/).waitFor();
  assert.equal(new URL(page.url()).pathname,'/conversations');
  assert.equal(await page.getByRole('radio',{name:'Research guide',exact:true}).isDisabled(),true);
  mismatch=false;await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Session persona',{exact:true}).getByText('Writing coach',{exact:true}).waitFor();
  assert.equal(bodies.length,2);assert.deepEqual(bodies[1],bodies[0]);
});

for(const mismatch of ['missing','different'])test(`persona fingerprint ${mismatch} receipt freezes the exact request until replay`,async t=>{
  const versioned=personas.map(p=>({...p,fingerprint:'0123456789abcdef'}));
  const {page}=await fixture(t,{personaCatalog:versioned});const bodies=[];let bad=true;
  await page.route('**/v1/conversations',async route=>{
    if(route.request().method()!=='POST')return route.continue();
    bodies.push(route.request().postDataJSON());const response=await route.fetch(),body=await response.json();
    if(bad)body.persona=mismatch==='missing'?{id:'coach',name:'Writing coach'}:{...body.persona,fingerprint:'fedcba9876543210'};
    await route.fulfill({response,json:body});
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Writing coach',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByText(/Start was not confirmed/).waitFor();
  assert.equal(bodies[0].persona_fingerprint,'0123456789abcdef');
  assert.equal(await page.getByRole('radio',{name:'Research guide',exact:true}).isDisabled(),true);
  bad=false;await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.waitForURL('**/conversations/current');assert.deepEqual(bodies[1],bodies[0]);
  await page.route('**/v1/conversations/current',async route=>{
    const response=await route.fetch(),body=await response.json();
    await route.fulfill({response,json:{...body,persona:{...body.persona,name:'Changed catalog name',current:false}}});
  });
  await page.reload();const identity=page.getByLabel('Session persona',{exact:true});
  await identity.getByText('coach',{exact:true}).waitFor();
  await identity.getByText(/differs from the current catalog/).waitFor();
  assert.equal(await identity.getByText('Changed catalog name',{exact:true}).count(),0);
});

test('a refused conversation start preserves the selected persona without an uncertain receipt',async t=>{
  const {page}=await fixture(t,{personaCatalog:personas});let writes=0;
  await page.route('**/v1/conversations',route=>{
    if(route.request().method()!=='POST')return route.continue();
    writes++;return route.fulfill({status:403,json:{error:'key role read cannot write'}});
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Writing coach',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByRole('alert').filter({hasText:/write access/}).waitFor();
  assert.equal(await page.getByText(/Start was not confirmed|Memory selection is locked/).count(),0);
  assert.equal(await page.getByRole('radio',{name:'Writing coach',exact:true}).isChecked(),true);
  assert.equal(await page.getByRole('radio',{name:'Research guide',exact:true}).isEnabled(),true);
  assert.equal(await page.locator('.server-state').innerText(),'Authenticated');
  assert.equal(writes,1,'A refusal must not trigger an automatic retry.');
});

for(const status of [422,403])test(`a later ${status} error cannot erase the identity of an uncertain create`,async t=>{
  const {page}=await fixture(t,{personaCatalog:personas});const bodies=[];
  await page.route('**/v1/conversations',async route=>{
    if(route.request().method()!=='POST')return route.continue();
    bodies.push(route.request().postDataJSON());
    if(bodies.length===1){await route.fetch();return route.fulfill({status:503,json:{error:'acknowledgement lost'}});}
    return route.fulfill({status,json:{error:'persona unavailable or operation forbidden'}});
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Writing coach',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  const start=page.getByRole('button',{name:'Start text conversation',exact:true});await start.click();
  await page.getByText(/Start was not confirmed/).waitFor();await start.click();
  await page.getByText(/Start was not confirmed/).waitFor();
  assert.equal(await page.getByRole('radio',{name:'Research guide',exact:true}).isDisabled(),true);
  assert.deepEqual(bodies[0],bodies[1]);
});

for(const explicit of [true,false])test(`stale persona recovery requires definitive no-write code, explicit=${explicit}`,async t=>{
  const versioned=personas.map(p=>({...p,fingerprint:'0123456789abcdef'}));
  const {page}=await fixture(t,{personaCatalog:versioned});const bodies=[];let rejected=false;
  await page.route('**/v1/conversations/personas',route=>route.fulfill({json:{schema_version:1,revision:'abcdef0123456789',personas:versioned}}));
  await page.route('**/v1/conversations',async route=>{
    if(route.request().method()!=='POST')return route.continue();
    bodies.push(route.request().postDataJSON());
    if(!rejected){rejected=true;versioned.forEach(p=>p.fingerprint='fedcba9876543210');return route.fulfill({status:409,json:{error:'persona selection is stale',...(explicit?{code:'persona_selection_stale'}:{})}});}
    return route.continue();
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Writing coach',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  if(!explicit){await page.getByText(/Start was not confirmed/).waitFor();assert.equal(await page.getByRole('radio',{name:'Research guide',exact:true}).isDisabled(),true);return;}
  await page.getByText(/configuration changed before this conversation was created/).waitFor();
  const start=page.getByRole('button',{name:'Start text conversation',exact:true});
  await page.getByRole('radio',{name:'Writing coach',exact:true}).waitFor();assert.equal(await start.isDisabled(),true);
  assert.equal(bodies.length,1,'refresh must not create automatically');
  await page.getByRole('radio',{name:'Writing coach',exact:true}).check();await start.click();
  await page.waitForURL('**/conversations/current');assert.notEqual(bodies[0].request_id,bodies[1].request_id);
  assert.equal(bodies[1].persona_fingerprint,'fedcba9876543210');
});

test('malformed persona catalog cannot unlock an unchosen session',async t=>{
  const {page}=await fixture(t,{personaCatalog:personas});
  await page.route('**/v1/conversations/personas',r=>r.fulfill({json:{schema_version:1,personas:[personas[0],personas[0]]}}));
  await page.reload();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('button',{name:'Retry personas',exact:true}).waitFor();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  assert.equal(await page.getByRole('button',{name:'Start text conversation',exact:true}).isDisabled(),true);
});
async function start(page){
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const start=page.getByRole('button',{name:'Start text conversation',exact:true});
  assert.equal(await start.isDisabled(),true);
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await start.click();await page.getByLabel('Message',{exact:true}).waitFor();
}

for(const mobile of [false,true])test(`missing conversation service explains setup and recovers without writes, mobile=${mobile}`,async t=>{
  const {page,requested,recoverService,posts}=await fixture(t,{capStatus:404,mobile});
  await page.getByRole('heading',{name:'Connect a conversation service',exact:true}).waitFor();
  assert.equal(await page.getByLabel('Saved session count').innerText(),'—','unread history is not zero sessions');
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  assert.equal(requested.some(p=>p.startsWith('/v1/conversations?')),false);
  await page.getByText('Service setup details',{exact:true}).click();
  await page.getByText(/scone-memory serve-conversations --help/).waitFor();
  assert.equal(await page.getByRole('button',{name:/microphone|start voice/i}).count(),0);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`scone-readiness-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  recoverService();await page.getByRole('button',{name:'Retry service connection',exact:true}).click();
  await page.getByRole('link',{name:/previous/}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isEnabled(),true);
  assert.equal(posts(),0);assert.equal(requested.filter(p=>p==='/v1/conversations').length,0);
});

test('discovery pending is not presented as a ready or empty workspace',async t=>{
  const {page,releaseCapabilities}=await fixture(t,{holdCapabilities:true});
  await page.getByRole('heading',{name:'Checking conversation service',exact:true}).waitFor();
  assert.equal(await page.getByLabel('Saved session count').innerText(),'—');
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  releaseCapabilities();await page.getByRole('link',{name:/previous/}).waitFor();
  assert.equal(await page.getByLabel('Saved session count').innerText(),'1');
});

test('temporary discovery failure offers retry without asserting service absence',async t=>{
  const {page,recoverService}=await fixture(t,{capStatus:503});
  await page.getByRole('heading',{name:'Connection check failed',exact:true}).waitFor();
  assert.equal(await page.getByRole('heading',{name:'Connect a conversation service',exact:true}).count(),0);
  recoverService();await page.getByRole('button',{name:'Retry service connection',exact:true}).click();
  await page.getByRole('link',{name:/previous/}).waitFor();
});

test('session-list failure does not disable text or discard a draft during list retry',async t=>{
  const {page,requested,recoverList,posts}=await fixture(t,{listFailure:true});
  await page.getByRole('button',{name:'Retry saved sessions',exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isEnabled(),true);
  await start(page);await page.getByLabel('Message',{exact:true}).fill('Keep this unsent draft');
  const discoveryReads=requested.filter(p=>p==='/v1/conversations/capabilities').length;
  recoverList();await page.getByRole('button',{name:'Retry saved sessions',exact:true}).click();
  await page.getByRole('link',{name:/previous/}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep this unsent draft');
  assert.equal(requested.filter(p=>p==='/v1/conversations/capabilities').length,discoveryReads);
  assert.equal(posts(),0);
});

test('a delayed initial session list cannot erase a newly created session',async t=>{
  const {page,releaseList}=await fixture(t,{holdFirstList:true});
  await start(page);await page.getByRole('link',{name:/current/}).waitFor();
  releaseList();await page.getByRole('link',{name:/previous/}).waitFor();
  assert.equal(await page.getByRole('link',{name:/current/}).count(),1,'late history must merge, not replace current sessions');
  assert.equal(await page.getByLabel('Message',{exact:true}).isEnabled(),true);
});

for(const mobile of [false,true])test(`live public chunks stay provisional until a saved receipt, mobile=${mobile}`,async t=>{
  const {page,emit,finish,posts,requested}=await fixture(t,{streaming:true,mobile});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Show the reply');await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();
  emit('Hello 🌿 <script>literal</script>');
  await preview.getByText('Hello 🌿 <script>literal</script>',{exact:true}).waitFor();
  assert.equal(await page.getByRole('heading',{name:'Start with a question.',exact:true}).count(),0,'an active reply must not keep the empty-conversation invitation');
  assert.equal(await page.getByRole('region',{name:'Saved messages'}).getByText('Hello 🌿 <script>literal</script>',{exact:true}).count(),0);
  assert.ok(requested.some(path=>path.endsWith('/stream?after=0')));
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`scone-live-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  finish();await page.getByRole('region',{name:'Saved messages'}).getByText('Saved complete answer.',{exact:true}).waitFor();
  assert.equal(await preview.count(),0);assert.equal(posts(),1);
});

test('stream completion preserves transcript reading position and keeps the composer fixed until capture arrives',async t=>{
  const {page,emit,finish,streams,requested,posts}=await fixture(t,{streaming:true});await start(page);
  const userText='I am reading this retained context. '.repeat(100);
  await page.getByLabel('Message',{exact:true}).fill(userText);await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();emit('A provisional answer remains visible while capture completes.');
  await preview.getByLabel('Provisional reply text').waitFor();
  await page.getByRole('region',{name:'Saved messages'}).getByText(userText,{exact:true}).waitFor();
  await page.locator('.conversation-messages').evaluate(element=>{element.scrollTop=100;});
  const measure=()=>page.evaluate(()=>({page:window.scrollY,composer:document.querySelector('.conversation-composer').getBoundingClientRect().bottom,thread:document.querySelector('.conversation-thread').getBoundingClientRect().height,reading:document.querySelector('.conversation-messages').scrollTop}));
  const before=await measure();let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  await page.route('**/turns/*',async route=>{await gate;await route.continue().catch(()=>{});});
  const requestId=requested.find(url=>url.includes('/stream?')).split('/turns/')[1].split('/')[0];
  for(const stream of streams)stream.end(`event: terminal\ndata: ${JSON.stringify({request_id:requestId,status:'completed',read_receipt:true})}\n\n`);
  await preview.getByText('Live preview ended. Checking the saved reply.',{exact:true}).waitFor();
  assert.equal(await preview.getByLabel('Provisional reply text').innerText(),'A provisional answer remains visible while capture completes.');
  assert.equal(await preview.evaluate(element=>!!element.closest('.conversation-messages')),true,'Live and saved replies share one bounded transcript');
  assert.deepEqual(await measure(),before,'Terminal transport cannot move the page, composer or current reading position');
  finish();release();
  await page.getByRole('region',{name:'Saved messages'}).getByText('Saved complete answer.',{exact:true}).waitFor();
  await preview.waitFor({state:'detached'});
  assert.deepEqual(await measure(),before,'Persisting the reply cannot move the page, composer or current reading position');
  assert.equal(posts(),1);
});

test('an interrupted preview reconnects with its last sequence without resending, and gaps replace the discontinuous prefix',async t=>{
  const {page,emit,streams,posts,requested,finish}=await fixture(t,{streaming:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Stream');await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();emit('First segment');await preview.getByText('First segment',{exact:true}).waitFor();
  for(const stream of streams)stream.end();
  const reconnected=page.waitForRequest(r=>r.url().endsWith('/stream?after=1'));
  await preview.getByRole('button',{name:'Reconnect preview'}).click();await reconnected;
  await preview.getByText('Connecting to live reply…',{exact:true}).waitFor();
  for(const stream of streams)stream.write('event: gap\ndata: {"after":1,"next_sequence":9}\n\nevent: text\nid: 9\ndata: {"sequence":9,"text":"Latest tail","provisional":true}\n\n');
  await preview.getByText('Latest tail',{exact:true}).waitFor();assert.equal(await preview.getByText('First segment',{exact:true}).count(),0);
  await preview.getByText(/Earlier live text is missing/).waitFor();assert.equal(posts(),1);
  finish('failed');await preview.waitFor({state:'detached'});assert.equal(await page.getByText('Latest tail',{exact:true}).count(),0);
});

test('a fast completed turn uses a terminal-only stream and the saved receipt',async t=>{
  const {page,finish,posts}=await fixture(t,{streaming:true});await start(page);
  await page.route('**/turns',async route=>{const response=await route.fetch();finish();await route.fulfill({response});});
  await page.getByLabel('Message',{exact:true}).fill('Fast');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('region',{name:'Saved messages'}).getByText('Saved complete answer.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Live reply preview'}).count(),0);assert.equal(posts(),1);
});

test('switching sessions closes the preview reader and cannot render late text in another session',async t=>{
  const {page,emit,streams,posts}=await fixture(t,{streaming:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Stream');await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();emit('Current only');await preview.getByText('Current only',{exact:true}).waitFor();
  const closed=Promise.all([...streams].map(stream=>new Promise(resolve=>stream.once('close',resolve))));
  await page.getByRole('link',{name:/previous/}).click();await closed;
  emit('Must not cross sessions');await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(await preview.count(),0);assert.equal(await page.getByText(/Current only|Must not cross sessions/).count(),0);assert.equal(posts(),1);
});

test('an externally ended session clears provisional text even when its receipt read stalls',async t=>{
  const {page,emit,endExternally}=await fixture(t,{streaming:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Stream');await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();emit('No longer active');await preview.getByText('No longer active',{exact:true}).waitFor();
  let release;const gate=new Promise(resolve=>{release=resolve;});
  await page.route('**/turns/*',async route=>{await gate;await route.continue().catch(()=>{});});
  endExternally();
  try{
    await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
    assert.equal(await preview.count(),0,'a known ended session must not retain an unconfirmed live preview');
  }finally{release();}
});

test('oversized live previews stop reading without growing unbounded or resubmitting',async t=>{
  const {page,emit,finish,posts}=await fixture(t,{streaming:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Large reply');await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();
  for(let i=0;i<17;i++)emit('a'.repeat(65536));
  await preview.getByText(/Preview size limit reached/).waitFor();
  assert.equal((await preview.getByLabel('Provisional reply text').textContent()).length,1048576);
  assert.equal(await preview.getByRole('button',{name:'Reconnect preview'}).count(),0);
  finish();await page.getByRole('region',{name:'Saved messages'}).getByText('Saved complete answer.',{exact:true}).waitFor();assert.equal(posts(),1);
});

for(const outcome of ['forgotten','unreadable','unavailable'])test(`a terminal ${outcome} receipt never promotes provisional text`,async t=>{
  const {page,emit,finish}=await fixture(t,{streaming:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Stream');await page.getByRole('button',{name:'Send message',exact:true}).click();
  const preview=page.getByRole('region',{name:'Live reply preview'});await preview.waitFor();emit('Not a retained reply');await preview.getByText('Not a retained reply',{exact:true}).waitFor();
  finish('completed',outcome);await preview.waitFor({state:'detached'});assert.equal(await page.getByText('Not a retained reply',{exact:true}).count(),0);
});

for(const mobile of [false,true])test(`session scope is chosen before capture and visible afterward, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{scoped:true,mobile});const writes=[];
  page.on('request',r=>{if(r.method()==='POST')writes.push(r.postDataJSON());});
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('button',{name:'Files',exact:true}).click();
  await dialog.getByLabel('Source prefix',{exact:true}).fill('docs/');
  await dialog.getByText('Dates & metadata',{exact:true}).click();
  await dialog.getByLabel('Created on or after (UTC)',{exact:true}).fill('2026-09-01');
  await dialog.getByRole('button',{name:'Add metadata filter',exact:true}).click();
  await dialog.getByLabel('Metadata key 1',{exact:true}).fill('collection');
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await dialog.getByRole('alert').waitFor();assert.equal(writes.length,0);
  await dialog.getByLabel('Metadata value 1',{exact:true}).fill('manuals');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`scone-scope-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).waitFor();
  assert.deepEqual(writes[0].recall_scope,{kind:'file',source_prefix:'docs/',since:'2026-09-01T00:00:00.000Z',where:{collection:'manuals'}});
  await page.getByText('Memory selection',{exact:true}).click();
  await page.getByText('collection = manuals',{exact:true}).waitFor();
  assert.equal(await page.getByText('docs/',{exact:true}).count(),1);
});

test('an uncertain session start freezes and replays the same scope',async t=>{
  const {page}=await fixture(t,{scoped:true});const bodies=[];let first=true;
  await page.route('**/v1/conversations',async route=>{
    bodies.push(route.request().postDataJSON());
    if(first){first=false;await route.fetch();return route.abort();}return route.continue();
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('button',{name:'Files',exact:true}).click();
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await dialog.getByText(/Start was not confirmed/).waitFor();
  assert.equal(await dialog.getByRole('button',{name:'Notes',exact:true}).isDisabled(),true);
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).waitFor();assert.deepEqual(bodies[0],bodies[1]);
});

test('unsupported scope remains unknown and cannot be selected',async t=>{
  const {page}=await fixture(t);await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Memory selection',{exact:true}).click();
  await page.getByText('This server did not report the session’s recall filters.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  assert.equal(await dialog.getByLabel('Source prefix',{exact:true}).count(),0);
  await dialog.getByText('Memory selection is managed by this server.',{exact:true}).waitFor();
});

for(const missing of [false,true])test(`unconfirmed recall scope prevents navigation and retains retry payload, missing=${missing}`,async t=>{
  const {page}=await fixture(t,{scoped:true});const bodies=[];let first=true;
  await page.route('**/v1/conversations',async route=>{
    bodies.push(route.request().postDataJSON());const response=await route.fetch(),json=await response.json();
    if(first){first=false;if(missing)delete json.recall_scope;else json.recall_scope={};}
    return route.fulfill({json});
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('button',{name:'Files',exact:true}).click();
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await dialog.getByText(/Start was not confirmed/).waitFor();
  assert.match(page.url(),/\/conversations$/);assert.equal(await page.getByLabel('Message',{exact:true}).count(),0);
  assert.equal(await dialog.getByRole('button',{name:'Notes',exact:true}).isDisabled(),true);
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).waitFor();assert.deepEqual(bodies[0],bodies[1]);
});

test('definitive scope rejection permits correction with a new request',async t=>{
  const {page}=await fixture(t,{scoped:true});const bodies=[];
  await page.route('**/v1/conversations',route=>{
    bodies.push(route.request().postDataJSON());
    return bodies.length===1?route.fulfill({status:422,json:{detail:'Rejected filter'}}):route.continue();
  });
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('button',{name:'Files',exact:true}).click();
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await dialog.getByText(/The server rejected these settings/).waitFor();
  await dialog.getByRole('button',{name:'Notes',exact:true}).click();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).waitFor();
  assert.notEqual(bodies[0].request_id,bodies[1].request_id);assert.deepEqual(bodies[1].recall_scope,{kind:'note'});
});

for(const mobile of [false,true])test(`transcript navigation reaches older messages and keeps the page during polling, mobile=${mobile}`,async t=>{
  const {page,requested}=await fixture(t,{pagination:true,mobile});
  await page.getByRole('link',{name:/previous/}).click();
  const messages=page.locator('.conversation-messages');
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  assert.equal(await messages.locator('article').count(),50);
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Saved message 24',{exact:true}).waitFor();
  assert.equal(await page.getByText('Saved message 123',{exact:true}).count(),0);
  const reads=()=>requested.filter(url=>url.includes('/transcript?')&&url.includes('before=cursor-74')).length;
  const prior=reads();await page.waitForResponse(response=>response.url().includes('before=cursor-74'));
  assert.ok(reads()>prior);assert.equal(await page.getByText('Saved message 24',{exact:true}).count(),1);
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Saved message 1',{exact:true}).waitFor();
  assert.equal(await messages.locator('article').count(),23);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`scone-transcript-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  assert.equal(await page.getByRole('button',{name:'Older messages',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Newer messages',exact:true}).click();
  await page.getByText('Saved message 24',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Latest messages',exact:true}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});

test('a failed older page can be retried without resending or losing the selected boundary',async t=>{
  const {page,posts}=await fixture(t,{pagination:true});let fail=true;
  await page.route('**/previous/transcript?*before=*',route=>fail?route.fulfill({status:503,json:{error:'read unavailable'}}):route.continue());
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText(/Connection interrupted. Controls are paused/).waitFor();
  assert.equal(await page.getByText('Saved message 123',{exact:true}).count(),0);
  fail=false;await page.getByRole('button',{name:'Check connection',exact:true}).click();
  await page.getByText('Saved message 24',{exact:true}).waitFor();
  assert.equal(posts(),0);
});

test('late older-page responses cannot replace another selected session',async t=>{
  const {page}=await fixture(t,{pagination:true});let release;
  const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  await page.route('**/previous/transcript?*before=*',async route=>{await gate;await route.fulfill({json:{episodes:[{episode_id:1,content:'Late older text',metadata:{}}],has_more:false,next_before:null}}).catch(()=>{});});
  const requested=page.waitForRequest(request=>request.url().includes('before=cursor-74'));
  await page.getByRole('button',{name:'Older messages',exact:true}).click();await requested;
  await start(page);release();
  await page.getByRole('heading',{name:'Start with a question.',exact:true}).waitFor();
  assert.match(page.url(),/\/current$/);assert.equal(await page.getByText('Late older text',{exact:true}).count(),0);
});

test('closed session deletion requires confirmation and removes the saved session',async t=>{
  const {page}=await fixture(t,{deletion:true,mobile:true});
  const commands=[];page.on('request',r=>{if(r.method()==='DELETE')commands.push(r.url());});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Keep conversation',exact:true}).click();
  assert.deepEqual(commands,[]);
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true,'reopening needs fresh confirmation');
  await dialog.getByLabel('I understand this cannot be undone').check();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'scone-delete-mobile.png'),fullPage:true});
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await page.waitForURL(/\/conversations$/);
  await page.getByText('Conversation deleted.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('link',{name:/previous/}).count(),0);
  assert.equal(await page.getByText('Earlier conversation.',{exact:true}).count(),0);
  assert.equal(commands.length,1);
});

test('unsupported and active sessions expose no delete command',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Delete conversation',exact:true}).count(),0);
  await start(page);await page.getByLabel('Message',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Delete conversation',exact:true}).count(),0);
});

test('a supported server still cannot delete an active session',async t=>{
  const {page}=await fixture(t,{deletion:true});await start(page);
  await page.waitForFunction(()=>!document.querySelector('textarea')?.disabled);
  assert.equal(await page.getByRole('button',{name:'Delete conversation',exact:true}).count(),0);
});

test('a refused delete preserves the session and requires fresh confirmation',async t=>{
  const {page}=await fixture(t,{deletion:true});let deletes=0;
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.route('**/v1/conversations/previous',route=>{
    if(route.request().method()!=='DELETE')return route.continue();
    deletes++;return route.fulfill({status:409,json:{error:'stop first'}});
  });
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await dialog.getByText(/Deletion was refused/).waitFor();
  assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);
  await dialog.getByRole('button',{name:'Keep conversation',exact:true}).click();
  assert.equal(await page.getByText('Earlier conversation.',{exact:true}).count(),1);
  assert.equal(deletes,1);
});

test('Escape cannot dismiss an in-flight deletion or permit a second command',async t=>{
  const {page}=await fixture(t,{deletion:true});let release;
  const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());let deletes=0;
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.route('**/v1/conversations/previous',async route=>{
    if(route.request().method()!=='DELETE')return route.continue();
    deletes++;await gate;await route.fulfill({status:204}).catch(()=>{});
  });
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await dialog.getByRole('button',{name:'Deleting…',exact:true}).waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await dialog.isVisible(),true);
  assert.equal(await dialog.getByRole('button',{name:'Deleting…',exact:true}).isDisabled(),true);
  release();await page.waitForURL(/\/conversations$/);assert.equal(deletes,1);
});

for(const missing of [false,true])test(`uncertain deletion uses a read-only check; session missing=${missing}`,async t=>{
  const {page}=await fixture(t,{deletion:true});let deletes=0;
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.route('**/v1/conversations/previous',route=>{
    if(route.request().method()==='DELETE'){deletes++;return route.fulfill({status:503,json:{error:'unknown outcome'}});}
    if(deletes&&missing)return route.fulfill({status:404,json:{error:'not found'}});
    return route.continue();
  });
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await dialog.getByRole('button',{name:'Check deletion status',exact:true}).click();
  if(missing){await page.waitForURL(/\/conversations$/);await page.getByText('Conversation is no longer available.',{exact:true}).waitFor();}
  else{await dialog.getByText(/still exists/).waitFor();assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);}
  assert.equal(deletes,1,'uncertain outcome must not automatically issue another DELETE');
});
test('unconfigured text keeps saved conversations readable without permitting new work',async t=>{
  const {page,posts}=await fixture(t,{unavailable:true});
  await page.getByText('Text runtime not configured',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  await page.reload();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(posts(),0);
});
test('an unknown conversation contract does not trigger session reads',async t=>{
  const {page,requested}=await fixture(t,{unknown:true});
  await page.getByRole('button',{name:'Retry service connection',exact:true}).waitFor();
  await page.getByRole('heading',{name:'Conversation service needs an update',exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  assert.equal(requested.some(p=>p.startsWith('/v1/conversations?')),false);
});
test('composer Enter sends once, Shift+Enter inserts a newline, and composition never submits',async t=>{
  const {page,posts}=await fixture(t,{streaming:true});
  await start(page);
  const input=page.getByLabel('Message',{exact:true});
  await input.fill('First line');
  await input.press('Shift+Enter');
  await input.press('A');
  assert.equal(await input.inputValue(),'First line\nA');
  assert.equal(posts(),0);
  await input.dispatchEvent('keydown',{key:'Enter',code:'Enter',isComposing:true});
  await input.dispatchEvent('keydown',{key:'Enter',code:'Enter',keyCode:229});
  await input.dispatchEvent('keydown',{key:'Enter',code:'Enter',repeat:true});
  assert.equal(posts(),0);
  const sent=page.waitForRequest(r=>r.method()==='POST'&&new URL(r.url()).pathname==='/v1/conversations/current/turns');
  await input.press('Enter');
  const request=await sent;
  assert.equal(request.postDataJSON().text,'First line\nA');
  await page.waitForFunction(()=>document.querySelector('#conversation-message').value==='');
  assert.equal(posts(),1);
  await input.fill('Keep this draft while reply is pending');
  await input.press('Enter');
  assert.equal(posts(),1);
  assert.equal(await input.inputValue(),'Keep this draft while reply is pending');
});

test('capture consent, send, prepared sources and stop operate through the API',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  assert.equal(posts(),1);
  await page.getByRole('button',{name:'Source episode 7',exact:true}).click();
  await page.locator('.source-markdown').getByText('Juniper is calibrated with Polaris. <script>not executable</script>',{exact:true}).waitFor();
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.getByText('Conversation ended',{exact:true}).waitFor();
  await page.waitForFunction(()=>document.querySelector('nav[aria-label="Saved conversations"] a.active')?.textContent.includes('ended'));
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'scone-conversation-desktop.png'),fullPage:true});
});
test('reopening a session reads saved messages without creating or sending',async t=>{
  const {page,posts}=await fixture(t);
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.reload();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(posts(),0);assert.match(page.url(),/\/conversations\/previous$/);
});
test('uncertain send polls the same request and never automatically submits twice',async t=>{
  const {page,posts}=await fixture(t,{uncertain:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Juniper');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();assert.equal(posts(),1);
});
test('mobile conversation workspace fits the viewport',async t=>{
  const {page}=await fixture(t,{mobile:true});await start(page);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  assert.equal(await page.evaluate(()=>document.querySelector('.topbar nav').getBoundingClientRect().bottom<=document.querySelector('.server-strip').getBoundingClientRect().top+1),true,'navigation must not overlap connection strip');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'scone-conversation-mobile.png'),fullPage:true});
});
test('definitively rejected message preserves draft and does not wait for a nonexistent receipt',async t=>{
  const {page,posts}=await fixture(t,{reject:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Keep this draft');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Message was not accepted. Your draft is kept.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep this draft');
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isEnabled(),true);
  assert.equal(posts(),1);
});
test('late polling cannot reopen a conversation after an acknowledged end',async t=>{
  const {page}=await fixture(t);await start(page);
  await page.waitForFunction(()=>!document.querySelector('textarea')?.disabled);
  let release;const gate=new Promise(resolve=>{release=resolve;});
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});
  await page.route('**/current/transcript',async route=>{entered();await gate;await route.continue();},{times:1});
  await blocked;
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.getByText('Conversation ended',{exact:true}).waitFor();
  release();await page.waitForResponse(response=>response.url().endsWith('/current/transcript'));
  await page.waitForTimeout(150);
  assert.equal(await page.getByRole('button',{name:'End conversation',exact:true}).isDisabled(),true);
  assert.match(await page.locator('nav[aria-label="Saved conversations"] a.active').textContent(),/ended/);
});

test('cancel reply preserves the next draft and never resends the cancelled turn',async t=>{
  const {page,posts}=await fixture(t,{cancellation:true});await start(page);
  const cancels=[];page.on('request',r=>{if(r.url().endsWith('/cancel'))cancels.push(r.url());});
  await page.getByLabel('Message',{exact:true}).fill('First question');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('button',{name:'Cancel reply',exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('Keep this next question');
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByText(/Reply cancelled/).waitFor();
  await page.waitForFunction(()=>!document.querySelector('button[aria-label="Send message"]')?.disabled);
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep this next question');
  assert.equal(posts(),1);assert.equal(cancels.length,1);
  await page.getByRole('button',{name:'Send message',exact:true}).click();await page.getByText('Use Polaris.',{exact:true}).waitFor();
  assert.equal(posts(),2);
});

test('an uncertain cancellation checks the existing outcome without another POST',async t=>{
  const {page,posts}=await fixture(t,{cancellation:true});await start(page);let cancels=0;
  await page.route('**/turns/*/cancel',async route=>{cancels++;await route.fetch();await route.fulfill({status:503,json:{error:'acknowledgment lost'}});});
  await page.getByLabel('Message',{exact:true}).fill('Question');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByText(/Cancellation was not confirmed/).waitFor();
  await page.getByText(/Reply cancelled locally/).waitFor();
  assert.equal(cancels,1);assert.equal(posts(),1);
});

test('an old cancellation response cannot replace a newly selected conversation',async t=>{
  const {page}=await fixture(t,{cancellation:true});await start(page);
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  await page.route('**/turns/*/cancel',async route=>{
    await gate;await route.fulfill({json:{request_id:new URL(route.request().url()).pathname.split('/').at(-2),status:'cancelled',result_state:'unavailable',result:null}}).catch(()=>{});
  });
  await page.getByLabel('Message',{exact:true}).fill('Question');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByRole('button',{name:'Cancelling…',exact:true}).waitFor();
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  release();await page.waitForTimeout(150);
  assert.match(page.url(),/\/previous$/);assert.equal(await page.getByText(/Reply cancelled locally/).count(),0);
});

test('failed receipt reads keep Send paused while cancelled cleanup is active',async t=>{
  const {page,posts}=await fixture(t);await start(page);let reads=0;
  await page.route('**/current/turns/*',route=>{
    reads++;
    return reads===1?route.fulfill({json:{request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'cancelled',result_state:'unavailable',result:null}}):route.fulfill({status:503,json:{error:'temporarily unavailable'}});
  });
  await page.getByLabel('Message',{exact:true}).fill('Cancelled request');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText(/Reply cancelled/).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('Keep this draft');
  await page.getByText('Reply receipt unavailable. No message was resent.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  assert.equal(posts(),1);
});

test('a cancelled receipt settles without replaying the message',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.route('**/current/turns/*',route=>route.fulfill({json:{request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'cancelled',result_state:'unavailable',result:null}}));
  await page.getByLabel('Message',{exact:true}).fill('Cancelled request');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText(/Reply cancelled/).waitFor();
  assert.equal(posts(),1);
});

for(const result_state of ['forgotten','unavailable'])test(`completed reply with ${result_state} content settles without resending`,async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),
    status:'completed',result_state,result:null,
  }}));
  await page.getByLabel('Message',{exact:true}).fill('An uncertain outcome');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText(result_state==='forgotten'?'Reply completed; its saved text was forgotten.':'Reply completed; its text is unavailable.',{exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('A different question');
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isEnabled(),true);
  assert.equal(posts(),1,'settling an outcome must not replay the original turn');
});

test('a forgotten settled reply clears its cached evidence on the next receipt check',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.getByRole('button',{name:'Source episode 7',exact:true}).click();
  await evidence.locator('.source-markdown').getByText('Juniper is calibrated with Polaris. <script>not executable</script>',{exact:true}).waitFor();
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),
    status:'completed',result_state:'forgotten',result:null,
  }}));
  await page.route('**/current/transcript',route=>route.fulfill({json:{episodes:[{episode_id:10,content:'How is Juniper calibrated?',metadata:{role:'user'}}],has_more:false}}));
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  assert.equal(await page.getByText('Use Polaris.',{exact:true}).count(),0);
  assert.equal(await evidence.getByRole('button',{name:'Source episode 7',exact:true}).count(),0);
  assert.equal(await evidence.locator('.conversation-source-text').count(),0);
  assert.equal(posts(),1);
});

test('a late transcript from the previous session cannot replace the selected conversation',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.waitForFunction(()=>!document.querySelector('textarea')?.disabled);
  let release;const gate=new Promise(resolve=>{release=resolve;});
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});
  await page.route('**/current/transcript',async route=>{
    entered();await gate;
    await route.fulfill({json:{episodes:[{episode_id:19,content:'Late response from old session',metadata:{role:'assistant'}}],has_more:false}}).catch(()=>{});
  },{times:1});
  await blocked;
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  release();await page.waitForTimeout(200);
  assert.equal(await page.getByText('Late response from old session',{exact:true}).count(),0);
  assert.match(page.url(),/\/conversations\/previous$/);
  assert.equal(posts(),0);
});

test('a delayed settled receipt cannot unlock sending while a newer turn is pending',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('First question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});let oldId;
  await page.route('**/current/turns/*',async route=>{
    const request_id=new URL(route.request().url()).pathname.split('/').at(-1);
    if(!oldId){oldId=request_id;entered();await gate;
      await route.fulfill({json:{request_id,status:'completed',result:{text:'Old answer',assistant_episode_id:11}}});
    }else await route.fulfill({json:{request_id,status:'pending'}});
  });
  await blocked;
  await page.getByLabel('Message',{exact:true}).fill('Second question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Reply in progress',{exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('Third question');
  const returned=page.waitForResponse(r=>r.url().endsWith('/turns/'+oldId));
  release();await returned;await page.waitForTimeout(200);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  assert.equal(await page.getByText('Reply in progress',{exact:true}).count(),1);
  assert.equal(posts(),2);
});

test('a forgotten reply does not repeatedly close a retained source the user inspects',async t=>{
  const {page}=await fixture(t);await start(page);
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'completed',result_state:'forgotten',result:null,
  }}));
  await page.route('**/v1/episodes/10',route=>route.fulfill({json:{episode_id:10,content:'Retained user question',metadata:{role:'user'}}}));
  await page.getByLabel('Message',{exact:true}).fill('Retained user question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Inspect message episode 10',exact:true}).click();
  const source=page.getByRole('complementary',{name:'Conversation evidence'}).locator('.conversation-source-text');
  await source.waitFor();
  await page.waitForResponse(r=>r.url().includes('/current/turns/'));await page.waitForTimeout(150);
  assert.equal(await source.locator('.source-markdown').textContent(),'Retained user question');
});

test('polling preserves a send that is still waiting for its HTTP acknowledgment',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('First question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});
  await page.route('**/current/turns',async route=>{entered();await gate;await route.continue().catch(()=>{});},{times:1});
  await page.getByLabel('Message',{exact:true}).fill('Second question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();await blocked;
  // Cross a polling interval while the server still knows only the old turn.
  await page.waitForTimeout(2200);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  assert.equal(await page.getByText('Sending message…',{exact:true}).count(),1);
  assert.equal(posts(),1);
  release();
});

test('temporarily unreadable completed reply retries the read without declaring deletion or resending',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  let available=false;
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'completed',
    result_state:available?'available':'unreadable',result:available?{text:'Recovered saved answer',assistant_episode_id:11}:null,
  }}));
  await page.getByLabel('Message',{exact:true}).fill('A question worth preserving');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Reply completed; its saved text cannot be read right now. Checking again…',{exact:true}).waitFor();
  assert.equal(await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).count(),0);
  await page.getByLabel('Message',{exact:true}).fill('Keep my next question');
  available=true;await page.getByText('Reply saved',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep my next question');
  assert.equal(posts(),1);
});

test('reopening and reloading discover the recorded latest outcome without submitting work',async t=>{
  const {page,requested,posts}=await fixture(t,{unavailable:true,recovered:true});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  await page.reload();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  assert.ok(requested.filter(p=>p==='/v1/conversations/previous/turns/a-newer').length>=2);
  assert.equal(requested.some(p=>p.includes('/turns?')),false,'no UUID sorting or history scan to guess the newest turn');
  assert.equal(posts(),0);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
});
