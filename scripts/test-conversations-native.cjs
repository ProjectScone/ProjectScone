// Real React → HTTP service → Scone scheduler → native memory, no live provider.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
const {pythonLayout}=require('./python-layout.cjs');

async function fixture(t,{reopened=false,scoped=false,streaming=false,composed=false,historyOnly=false,personas=false,processing=false}={}){
  const html=process.env.SCONE_CONVERSATIONS_HTML;
  assert.ok(html,'SCONE_CONVERSATIONS_HTML must name the verified isolated webapp artifact');
  const server=spawn(process.env.SCONE_TEST_PYTHON||path.join(pythonLayout(root).project,'.venv/bin/python'),
    ['-u',path.join(__dirname,'fixtures/conversation-server.py'),html,...(reopened?['--reopened']:[]),...(scoped?['--scoped']:[]),...(streaming?['--streaming']:[]),...(composed?['--composed']:[]),...(historyOnly?['--history-only']:[]),...(personas?['--personas']:[]),...(processing?['--processing']:[])],{cwd:root,stdio:['pipe','pipe','pipe']});
  let browser,logs='';server.stderr.on('data',part=>{logs=(logs+part).slice(-6000);});
  const closed=once(server,'close');
  let modelEntered;const modelWaiting=new Promise(resolve=>{modelEntered=resolve;});
  t.after(async()=>{
    if(server.exitCode===null&&server.signalCode===null){
      server.stdin.end('stop\n');
      const timeout=setTimeout(()=>server.kill('SIGKILL'),5000);
      try{await closed;}finally{clearTimeout(timeout);}
    }
    await closed; // stderr is drained before checking shutdown diagnostics
    await browser?.close(); // do not leak Chrome when a shutdown assertion fails
    assert.notEqual(server.signalCode,'SIGKILL','fixture required forced shutdown: '+logs);
    assert.equal(server.exitCode,0,'fixture did not exit cleanly: '+logs);
    assert.doesNotMatch(logs,/Traceback \(most recent call last\)|AttributeError/,'fixture shutdown failed');
  });
  const port=await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Fixture startup timed out: '+logs)),20000);
    let data='';
    server.stdout.on('data',part=>{data+=part;if(data.includes('CONVERSATIONS_MODEL_WAITING'))modelEntered();const m=data.match(/CONVERSATIONS_READY (\d+)/);if(m){clearTimeout(timer);resolve(Number(m[1]));}});
    server.once('error',error=>{clearTimeout(timer);reject(error);});
    server.once('exit',code=>{clearTimeout(timer);reject(Error(`Fixture exited ${code}: ${logs}`));});
  });
  const base=`http://127.0.0.1:${port}`;
  let ready=false;
  for(let i=0;i<100;i++){
    try{if((await fetch(base+'/healthz',{signal:AbortSignal.timeout(500)})).ok){ready=true;break;}}catch{}
    await new Promise(resolve=>setTimeout(resolve,25));
  }
  assert.ok(ready,'isolated API became ready: '+logs);
  browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});
  const page=await browser.newPage({viewport:{width:1320,height:940}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  return {page,base,errors,modelWaiting,release:()=>server.stdin.write('release\n')};
}

test('metadata editor narrows real native recall with nested rules and preserves source identity',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const documents=[
    ['published', {status:'published',priority:'10'}],
    ['draft', {status:'draft',priority:'10'}],
    ['missing status', {priority:'10'}],
    ['nonnumeric priority', {status:'published',priority:'soon'}],
    ['low priority', {status:'published',priority:'2'}],
    ['override', {status:'published',priority:'2',team:'incident, response: primary'}],
  ];
  const ids=[];
  for(const [label,metadata] of documents){
    const response=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({content:`quarterly planning note ${label}`,metadata})});
    assert.equal(response.status,200);ids.push((await response.json()).episode_id);
  }
  const queries=[],writes=[];
  page.on('request',request=>{const url=new URL(request.url());if(url.pathname==='/v1/recall')queries.push(url.searchParams);if(request.method()!=='GET')writes.push(url.pathname);});
  await page.goto(base+'/memory');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'Metadata filters',exact:true}).click();
  const editor=page.getByRole('region',{name:'Metadata filter editor',exact:true});
  await editor.getByLabel('Metadata field 1',{exact:true}).fill('status');
  await editor.getByLabel('Value 1',{exact:true}).fill('draft');
  await editor.getByLabel('Negate rule 1',{exact:true}).check();
  await editor.getByRole('button',{name:'Add group to root',exact:true}).click();
  await editor.getByLabel('Match rules in group 2',{exact:true}).selectOption('any');
  await editor.getByLabel('Metadata field 2.1',{exact:true}).fill('priority');
  await editor.getByLabel('Comparison 2.1',{exact:true}).selectOption('at_least');
  await editor.getByLabel('Value 2.1',{exact:true}).fill('10');
  await editor.getByRole('button',{name:'Add rule to 2',exact:true}).click();
  await editor.getByLabel('Metadata field 2.2',{exact:true}).fill('team');
  await editor.getByLabel('Value 2.2',{exact:true}).fill('incident, response: primary');
  const narrowed=page.waitForResponse(r=>new URL(r.url()).pathname==='/v1/recall'&&new URL(r.url()).searchParams.has('conditions'));
  const before=queries.length;
  await editor.getByRole('button',{name:'Apply metadata filter',exact:true}).click();
  const response=await narrowed;assert.equal(response.status(),200);
  const result=await response.json();
  assert.deepEqual(result.items.map(item=>item.episode_id).sort((a,b)=>a-b),[ids[0],ids[5]].sort((a,b)=>a-b));
  await page.locator('.rows').getByText('quarterly planning note published',{exact:false}).waitFor();
  await page.locator('.rows').getByText('quarterly planning note override',{exact:false}).waitFor();
  assert.equal(queries.length,before+1,'Applying performs only the narrowed request');
  assert.deepEqual(JSON.parse(queries.at(-1).get('conditions')),{all:[{field:'status',not:true,is:'draft'},{any:[{field:'priority',at_least:10},{field:'team',is:'incident, response: primary'}]}]});
  for(const label of ['draft','missing status','nonnumeric priority','low priority'])assert.equal(await page.locator('.rows').getByText(`quarterly planning note ${label}`,{exact:false}).count(),0);
  assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
});

test('a real source bookmark survives sign-in and reload but not a different space or forgetting',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const upload=await fetch(base+'/v1/attachments',{method:'POST',headers:{authorization:headers.authorization,'content-type':'image/png','x-filename':'juniper-original.png'},body:bytes});
  assert.equal(upload.status,200);const attachment=await upload.json();
  const saved=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({content:'**Juniper** retains the source bookmark experiment.',kind:'file',source:'juniper-notes.md',attachment_ids:[attachment.attachment_id]})});
  assert.equal(saved.status,200);const id=(await saved.json()).episode_id;
  const route=`${base}/memory/sources/${id}?space=alpha`,reads=[],writes=[];
  page.on('request',r=>{if(r.method()!=='GET')writes.push(r.url());if(new URL(r.url()).pathname===`/v1/episodes/${id}`)reads.push(r.headers().authorization);});
  assert.equal((await page.goto(route)).status(),200);assert.equal(page.url(),route);
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('heading',{name:'juniper-notes.md',exact:true}).waitFor();
  await page.getByRole('region',{name:'Source original'}).locator('strong').getByText('Juniper',{exact:true}).waitFor();
  await page.reload();
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('heading',{name:'juniper-notes.md',exact:true}).waitFor();
  await page.getByRole('button',{name:'View source images',exact:true}).click();
  await page.waitForFunction(()=>{const image=document.querySelector('.source-image img');return image?.complete&&image.naturalWidth===1;});
  const preview=await page.getByRole('img',{name:'juniper-original.png',exact:true}).getAttribute('src');
  await page.getByRole('button',{name:'Memory connection',exact:true}).click();
  await page.getByLabel('Use a different Scone space key',{exact:true}).fill('conversation-fixture-beta');await page.getByRole('button',{name:'Switch space',exact:true}).click();
  await page.getByRole('heading',{name:'This source belongs to a different space',exact:true}).waitFor();
  assert.equal(await page.getByText('juniper-notes.md',{exact:true}).count(),0);assert.ok(reads.every(key=>key==='Bearer conversation-fixture-alpha'));
  assert.equal(await page.evaluate(async url=>{try{await fetch(url);return false;}catch{return true;}},preview),true,'Space switch revokes the old image preview');
  await page.getByRole('button',{name:'Memory connection',exact:true}).click();
  await page.getByLabel('Use a different Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Switch space',exact:true}).click();
  await page.getByRole('heading',{name:'juniper-notes.md',exact:true}).waitFor();
  assert.equal((await fetch(base+`/v1/episodes/${id}`,{method:'DELETE',headers})).status,200);
  await page.getByRole('button',{name:'Refresh source',exact:true}).click();
  await page.getByRole('heading',{name:/Source unavailable|This source was forgotten/}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Source original'}).count(),0);
  assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
});

test('profile UI reads native context and reflects closure and forgetting within its space',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const post=async(path,body)=>{const r=await fetch(base+path,{method:'POST',headers,body:JSON.stringify(body)});assert.equal(r.status,200);return r.json();};
  const source=await post('/v1/episodes',{content:'Juniper stores the observatory calibration locally.'});
  const claim=await post('/v1/facts',{subject:'Juniper',predicate:'stores',object:'the observatory calibration locally',source_episode_id:source.episode_id,proposed:false});
  const other=await(await fetch(base+'/v1/profile',{headers:{authorization:'Bearer conversation-fixture-beta'}})).json();
  assert.equal(other.static_facts.some(f=>f.fact_id===claim.fact_id),false);
  assert.equal(other.recent.some(r=>r.episode_id===source.episode_id),false);
  const browserWrites=[];page.on('request',r=>{if(!['GET','HEAD'].includes(r.method()))browserWrites.push(r.url());});
  await page.goto(base+'/memory#profile');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  const claims=page.getByRole('region',{name:'Selected claim context',exact:true});
  await claims.getByText('juniper stores the observatory calibration locally',{exact:true}).waitFor();
  const recent=page.getByRole('region',{name:'Recent source context',exact:true});
  await recent.getByRole('button',{name:`Read source episode #${source.episode_id}`,exact:true}).click();
  await recent.locator('.source-markdown').getByText('Juniper stores the observatory calibration locally.',{exact:true}).waitFor();
  await post(`/v1/facts/${claim.fact_id}/close`,{reason:'Test retirement of context'});
  await page.getByRole('button',{name:'Refresh profile',exact:true}).click();
  await page.getByText('No claims were returned in this snapshot.',{exact:true}).waitFor();
  const forgotten=await fetch(base+`/v1/episodes/${source.episode_id}`,{method:'DELETE',headers});assert.equal(forgotten.status,200);
  await page.getByRole('button',{name:'Refresh profile',exact:true}).click();
  await recent.waitFor();
  assert.equal(await recent.getByText(`Episode #${source.episode_id}`,{exact:true}).count(),0,'Forgotten source no longer appears in profile context.');
  assert.deepEqual(browserWrites,[],'Profile inspection never changes stored memory.');assert.deepEqual(errors,[]);
});

for(const role of ['reader','writer'])test(`native ${role} denial preserves authenticated source inspection and leaves the claim unchanged`,{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const post=async(path,data)=>{const response=await fetch(base+path,{method:'POST',headers,body:JSON.stringify(data)});assert.equal(response.status,200);return response.json();};
  const source=await post('/v1/episodes',{content:'Juniper keeps a retained calibration manual.'});
  const claim=await post('/v1/facts',{subject:'Juniper',predicate:'keeps',object:'a retained calibration manual',source_episode_id:source.episode_id,proposed:true});
  await page.goto(base+'/memory#review');
  await page.getByLabel('Scone space key',{exact:true}).fill(`conversation-fixture-${role}`);await page.getByRole('button',{name:'Connect',exact:true}).click();
  const denied=page.waitForResponse(r=>r.url().endsWith(`/v1/facts/${claim.fact_id}/approve`));
  await page.getByRole('button',{name:'Approve',exact:true}).click();
  assert.equal((await denied).status(),403);
  await page.getByRole('alert').filter({hasText:'cannot decide'}).waitFor();
  assert.equal(await page.getByText(/No decision was confirmed|A timed-out request/).count(),0,'A refused request is not an uncertain write.');
  assert.equal(await page.locator('#space').innerText(),'alpha');
  assert.equal(await page.locator('.server-state').innerText(),'Authenticated');
  await page.getByText(`Read full source Episode #${source.episode_id}`,{exact:false}).click();
  await page.locator('.excerpt p').filter({hasText:'Juniper keeps a retained calibration manual.'}).waitFor();
  const facts=await(await fetch(base+'/v1/facts?status=proposed',{headers})).json();
  assert.ok(facts.facts.some(f=>f.fact_id===claim.fact_id),'The refused decision must leave the proposal pending.');
  await page.getByRole('button',{name:'Memory connection',exact:true}).click();
  await page.getByLabel('Use a different Scone space key',{exact:true}).fill('conversation-fixture-reviewer');
  await page.getByRole('button',{name:'Switch space',exact:true}).click();
  await page.getByRole('button',{name:'Approve',exact:true}).click();
  await page.getByText('Nothing awaits review.',{exact:true}).waitFor();
  const approved=await(await fetch(base+'/v1/facts',{headers})).json();
  assert.ok(approved.facts.some(f=>f.fact_id===claim.fact_id&&f.status==='active'),'An explicitly selected review key can decide the same proposal.');
  assert.deepEqual(errors,[]);
});

test('confirmed inference uses the native worker and stores only a scoped proposal',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{processing:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const facts=async(key='conversation-fixture-alpha')=>(await(await fetch(base+'/v1/facts?status=proposed',{headers:{authorization:'Bearer '+key}})).json()).facts;
  const capabilities=await(await fetch(base+'/v1/capabilities',{headers})).json();
  assert.equal(capabilities.features['processing.distill']??false,false);
  assert.equal(capabilities.features['processing.derive']??false,false);
  // Exercise the prepared UI only in this isolated, unscheduled fixture.
  // Production must not advertise manual passes until concurrency is guarded.
  await page.route('**/v1/capabilities',route=>route.fulfill({json:{...capabilities,features:{...capabilities.features,'processing.distill':true,'processing.derive':true}}}));
  await page.goto(base+'/memory#status');await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'Run inference',exact:true}).click();
  assert.equal((await facts()).length,0,'Opening confirmation must not invoke the model');
  await page.getByRole('dialog',{name:'Confirm inference pass'}).getByRole('button',{name:'Run one inference pass',exact:true}).click();
  await page.getByRole('heading',{name:'Pass report received',exact:true}).waitFor();
  const proposals=await facts();assert.equal(proposals.length,1);assert.equal(proposals[0].origin,'inferred');assert.equal(proposals[0].status,'proposed');
  assert.equal((await facts('conversation-fixture-beta')).length,0);
  const detail=await(await fetch(base+`/v1/facts/${proposals[0].fact_id}`,{headers})).json();assert.equal(detail.links.filter(link=>link.kind==='derived_from').length,2);
  await page.getByRole('button',{name:'Run maintenance',exact:true}).click();await page.getByRole('dialog',{name:'Confirm maintenance pass'}).getByRole('button',{name:'Run one maintenance pass',exact:true}).click();
  await page.getByRole('heading',{name:'Pass report received',exact:true}).waitFor();assert.equal((await facts()).length,1,'An unchanged group is not inferred twice');
  assert.deepEqual(errors,[]);
});

test('processing overview reads native backlogs without starting work or changing memory',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const created=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({content:'A native processing status source remains retained.'})});assert.equal(created.status,200);
  const before=await(await fetch(base+'/v1/status',{headers})).json();const writes=[];
  page.on('request',r=>{if(['POST','DELETE','PATCH','PUT'].includes(r.method()))writes.push(r.url());});
  await page.goto(base+'/memory#status');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  const panel=page.getByRole('region',{name:'Memory processing'});
  await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  assert.equal(await panel.getByRole('article',{name:'Sources'}).locator('.processing-count').innerText(),before.episodes.toLocaleString());
  assert.equal(await panel.getByRole('article',{name:'Extraction'}).locator('.processing-count').innerText(),before.pending_distill.toLocaleString());
  assert.match(await panel.getByRole('article',{name:'Inference'}).innerText(),/Off/);
  await page.getByRole('button',{name:'Refresh status',exact:true}).click();await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  const after=await(await fetch(base+'/v1/status',{headers})).json();
  for(const key of ['episodes','chunks','revision','pending_distill','pending_derivation','pending_review'])assert.equal(after[key],before[key],key+' unchanged');
  assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
});

test('integrity UI reads real native findings without repairs and respects space isolation',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const post=async(path,data)=>{const res=await fetch(base+path,{method:'POST',headers,body:JSON.stringify(data)});assert.equal(res.status,200);return res.json();};
  const source=await post('/v1/episodes',{content:'Juniper has a retained calibration source.'});
  const claim=await post('/v1/facts',{subject:'Juniper',predicate:'has',object:'calibration evidence',source_episode_id:source.episode_id});
  assert.equal((await fetch(base+`/v1/episodes/${source.episode_id}`,{method:'DELETE',headers})).status,200);
  const before=await(await fetch(base+'/v1/status',{headers})).json();
  await page.goto(base+'/memory#status');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  const writes=[];page.on('request',r=>{if(r.method()!=='GET')writes.push(r.url());});
  const panel=page.getByRole('region',{name:'Memory integrity'});
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('heading',{name:'References need attention',exact:true}).waitFor();
  await panel.locator('summary').filter({hasText:'Claims with forgotten sources'}).click();
  await panel.getByText(`Claim #${claim.fact_id}`,{exact:true}).waitFor();
  assert.deepEqual(writes,[]);
  const after=await(await fetch(base+'/v1/status',{headers})).json();
  for(const key of ['episodes','chunks','bytes','revision'])assert.equal(after[key],before[key],key+' must not change during an integrity check');
  const beta=await(await fetch(base+'/v1/doctor',{headers:{authorization:'Bearer conversation-fixture-beta'}})).json();
  assert.equal(beta.space,'beta');assert.deepEqual(beta.facts_citing_forgotten,[]);
  assert.equal((await fetch(base+'/v1/doctor')).status,401);
  assert.deepEqual(errors,[]);
});

test('claim relationships navigate real premises, preserve direction and disclose forgotten evidence',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(5000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const post=async(path,data)=>{const response=await fetch(base+path,{method:'POST',headers,body:JSON.stringify(data)});assert.equal(response.status,200);return response.json();};
  const source=await post('/v1/episodes',{content:'Mira works at Cedar. **Original evidence.**'});
  const premise=await post('/v1/facts',{subject:'Mira',predicate:'works_at',object:'Cedar',source_episode_id:source.episode_id,quote:'Mira works at Cedar'});
  const proposal=await post('/v1/facts',{subject:'Mira',predicate:'may_work_in',object:'payments',derived_from:[premise.fact_id],proposed:true});
  await page.goto(base+'/memory#review');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  const writes=[];page.on('request',request=>{if(request.method()!=='GET')writes.push(request.url());});
  await page.getByRole('button',{name:`Inspect premises for claim #${proposal.fact_id}`,exact:true}).click();
  const inspector=page.getByRole('dialog',{name:'Claim relationships',exact:true});
  await inspector.getByRole('heading',{name:`Claim #${proposal.fact_id}`,exact:true}).waitFor();
  await inspector.getByText('Proposed · not accepted',{exact:true}).waitFor();
  await inspector.getByRole('heading',{name:'Inference premises (1)',exact:true}).waitFor();
  await inspector.getByRole('button',{name:`Derived from · claim #${premise.fact_id}`,exact:true}).click();
  await inspector.getByRole('heading',{name:`Claim #${premise.fact_id}`,exact:true}).waitFor();
  await inspector.getByRole('button',{name:`Used to derive · claim #${proposal.fact_id}`,exact:true}).waitFor();
  await inspector.getByRole('button',{name:`Read source episode #${source.episode_id}`,exact:true}).click();
  await inspector.locator('strong').getByText('Original evidence.',{exact:true}).waitFor();
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'claim-relations-native.png'),fullPage:true});
  await inspector.getByRole('button',{name:'Back to previous claim',exact:true}).click();
  await inspector.getByRole('heading',{name:`Claim #${proposal.fact_id}`,exact:true}).waitFor();
  assert.equal(await inspector.getByText('Original evidence.',{exact:true}).count(),0,'source text cannot leak across claim navigation');
  await inspector.getByRole('button',{name:'Close dialog',exact:true}).click();
  assert.equal((await fetch(base+`/v1/episodes/${source.episode_id}`,{method:'DELETE',headers})).status,200);
  await page.getByRole('button',{name:`Inspect premises for claim #${proposal.fact_id}`,exact:true}).click();
  await inspector.getByRole('button',{name:`Derived from · claim #${premise.fact_id}`,exact:true}).click();
  await inspector.getByRole('button',{name:`Read source episode #${source.episode_id}`,exact:true}).click();
  await inspector.getByText('This source was forgotten. Its text is no longer retained.',{exact:true}).waitFor();
  assert.deepEqual(writes,[],'inspection must never approve, link or otherwise mutate memory');
  assert.deepEqual(errors,[]);
});

for(const composed of [false,true])test(`native concepts deep links stay public and navigate into authenticated memory, composed=${composed}`,{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed});
  page.setDefaultTimeout(8000);
  for(const [path,title] of [['/learn','Knowledge that carries forward.'],['/learn/how-it-works','From a source to useful context.'],['/learn/graph-memory','Connected, not unquestionable.']]){
    const response=await page.goto(base+path);
    assert.equal(response.status(),200);
    await page.getByRole('heading',{name:title,exact:true}).waitFor();
    assert.equal(await page.getByRole('dialog').count(),0);
  }
  await page.getByRole('link',{name:'Open Review',exact:true}).click();
  await page.getByRole('button',{name:'Set up memory connection',exact:true}).click();
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('heading',{name:'Review',exact:true}).waitFor();
  assert.equal(new URL(page.url()).hash,'#review');
  assert.deepEqual(errors,[]);
});

test('published concepts quickstart runs against native memory with retained source and isolated recall',{timeout:60000},async t=>{
  const {sourceExample}=await import('../Webapp/src/learn/content.ts');
  const {base,errors}=await fixture(t,{composed:true});
  const process=spawn('/bin/sh',['-ec',sourceExample],{env:{...global.process.env,SCONE_URL:base,SCONE_KEY:'conversation-fixture-alpha'},stdio:['ignore','pipe','pipe']});
  let output='',error='';process.stdout.on('data',part=>output+=part);process.stderr.on('data',part=>error+=part);
  const [code]=await once(process,'close');
  assert.equal(code,0,error);
  const split=output.indexOf('}{');assert.ok(split>0,'both curl calls return JSON');
  const added=JSON.parse(output.slice(0,split+1)),recall=JSON.parse(output.slice(split+1));
  assert.ok(added.episode_id>0);
  assert.ok(recall.items.some(item=>item.episode_id===added.episode_id));
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const source=await(await fetch(base+'/v1/episodes/'+added.episode_id,{headers})).json();
  assert.equal(source.content,'The observatory uses Polaris for calibration.');
  const denied=await fetch(base+'/v1/episodes/'+added.episode_id,{headers:{authorization:'Bearer conversation-fixture-beta'}});
  assert.equal(denied.status,404);
  assert.deepEqual(errors,[]);
});

for(const composed of [false,true])for(const outcome of ['complete','cancel','stop','switch-space'])test(`native Scone public-text preview → ${outcome}, composed=${composed}`,{timeout:60000},async t=>{
  const {page,base,errors,release,modelWaiting}=await fixture(t,{streaming:true,composed});page.setDefaultTimeout(8000);
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  let posts=0;page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/turns'))posts++;});
  await page.getByLabel('Message',{exact:true}).fill('Stream Juniper');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await modelWaiting;
  const preview=page.getByRole('region',{name:'Live reply preview'});
  await preview.getByText('Juniper 🌿',{exact:true}).waitFor();
  const sid=new URL(page.url()).pathname.split('/').at(-1),headers={authorization:'Bearer conversation-fixture-alpha'};
  const read=async()=>await(await fetch(base+'/v1/conversations/'+sid+'/transcript',{headers})).json();
  assert.deepEqual((await read()).episodes.map(e=>e.content),['Stream Juniper'],'public chunks precede final capture');
  assert.ok(!(await page.innerText('body')).includes('private-fixture-thought'));
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`native-live-${outcome}-composed-${composed}.png`),fullPage:true});
  if(outcome==='complete'){
    // A full page reload loses all browser preview state; the active window
    // replays public text without submitting another model request.
    await page.reload();
    await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
    await preview.getByText('Juniper 🌿',{exact:true}).waitFor();
    release();await page.getByRole('region',{name:'Saved messages'}).getByText('Juniper 🌿 uses Polaris.',{exact:true}).waitFor();
    assert.deepEqual((await read()).episodes.map(e=>e.content),['Stream Juniper','Juniper 🌿 uses Polaris.']);
  }else if(outcome==='switch-space'){
    await page.getByRole('button',{name:'Memory connection',exact:true}).click();
    await page.getByLabel('Use a different Scone space key',{exact:true}).fill('conversation-fixture-beta');
    await page.getByRole('button',{name:'Switch space',exact:true}).click();
    await page.locator('#space').getByText('beta',{exact:true}).waitFor();
    await preview.waitFor({state:'detached'});
    assert.equal(await page.getByText('Juniper 🌿',{exact:true}).count(),0);
    assert.deepEqual((await read()).episodes.map(e=>e.content),['Stream Juniper']);
    assert.equal((await fetch(base+'/v1/conversations/'+sid,{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  }else{
    await page.getByRole('button',{name:outcome==='cancel'?'Cancel reply':'End conversation',exact:true}).click();
    await preview.waitFor({state:'detached'});
    await page.getByText(outcome==='cancel'?/Reply cancelled locally/:/Conversation ended/).first().waitFor();
    assert.deepEqual((await read()).episodes.map(e=>e.content),['Stream Juniper']);
  }
  await preview.waitFor({state:'detached'});assert.equal(posts,1);assert.deepEqual(errors,[]);
});

test('batch history pages real native receipts and follows their original sources',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true});page.setDefaultTimeout(6000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'},jobs=[];
  for(let i=0;i<21;i++){
    const response=await fetch(base+'/v1/episodes/batch',{method:'POST',headers,body:JSON.stringify({request_id:`browser-batch-${i}`,records:[{content:`Native batch record ${i}.`,kind:'file'}]})});
    assert.equal(response.status,200);jobs.push((await response.json()).job);
  }
  const writes=[];page.on('request',r=>{if(r.method()!=='GET'&&r.url().includes('/v1/'))writes.push(r.url());});
  await page.goto(base+'/memory#status');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  const panel=page.getByRole('region',{name:'Batch history',exact:true});
  await panel.getByRole('button',{name:`Inspect batch ${jobs[20].job_id}`,exact:true}).click();
  const detail=panel.getByRole('region',{name:'Batch details',exact:true});
  assert.match(await detail.innerText(),/1 searchable · 0 consolidated/);
  await detail.getByRole('button',{name:`Inspect source ${jobs[20].items[0].episode_id}`,exact:true}).click();
  await detail.locator('.source-markdown').getByText('Native batch record 20.',{exact:true}).waitFor();
  await panel.getByRole('button',{name:'Older batches',exact:true}).click();
  await panel.getByRole('button',{name:`Inspect batch ${jobs[0].job_id}`,exact:true}).waitFor();
  assert.equal(await panel.getByRole('button',{name:'Older batches',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Memory connection',exact:true}).click();
  await page.getByLabel('Use a different Scone space key',{exact:true}).fill('conversation-fixture-beta');
  await page.getByRole('button',{name:'Switch space',exact:true}).click();
  await page.getByText('No batch receipts on this page.',{exact:false}).waitFor();
  assert.equal(await panel.locator('.job-card').count(),0);assert.equal(await panel.locator('.job-source').count(),0);
  assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
});

test('Documents browse native inventory, import retained text, and inspect saved image evidence',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);page.setDefaultTimeout(8000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  for(let i=0;i<28;i++){
    const response=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({kind:'file',source:`source-${i}.md`,content:`Native inventory document ${i}.`,created_at:i%2?'2020-01-01':'2026-09-06'})});
    assert.equal(response.status,200);
  }
  await page.goto(base+'/memory#documents');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'Files',exact:true}).click();
  await page.getByRole('button',{name:'Open source-27.md',exact:true}).waitFor();
  assert.equal(await page.locator('.document-card').count(),25);
  assert.equal(await page.locator('.document-card').first().getAttribute('aria-label'),'Open source-27.md','ID order is not a created-date or relevance sort');
  await page.getByRole('button',{name:'Older sources',exact:true}).click();
  await page.getByRole('button',{name:'Open source-2.md',exact:true}).waitFor();
  assert.equal(await page.locator('.document-card').count(),3);
  assert.equal(await page.getByRole('button',{name:'Older sources',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  const content='\ufeff# Atlas guide\r\nA preserved document from the source library.\r\n';
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'atlas.md',mimeType:'text/markdown',buffer:Buffer.from(content)});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByRole('heading',{name:/Source saved · episode #/}).waitFor();
  await page.getByRole('button',{name:'Open atlas.md',exact:true}).click();
  const detail=page.getByRole('region',{name:'Retained source'});
  await page.waitForFunction(()=>document.querySelector('.document-detail pre')?.textContent.includes('Atlas guide'));
  assert.equal(await detail.locator('pre').textContent(),content);
  const alpha=await(await fetch(base+'/v1/sources?limit=25',{headers})).json();
  const id=alpha.items[0].episode_id;assert.equal(alpha.items[0].source,'atlas.md');
  const betaHeaders={authorization:'Bearer conversation-fixture-beta'};
  assert.deepEqual((await(await fetch(base+'/v1/sources',{headers:betaHeaders})).json()).items,[]);
  assert.equal((await fetch(base+'/v1/episodes/'+id,{headers:betaHeaders})).status,404);
  const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const upload=await fetch(base+'/v1/attachments',{method:'POST',headers:{authorization:headers.authorization,'content-type':'image/png','x-filename':'atlas-original.png'},body:bytes});
  assert.equal(upload.status,200);const attachment=await upload.json();
  const attached=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({kind:'note',content:'Atlas reference image',source:'Atlas original',attachment_ids:[attachment.attachment_id]})});
  assert.equal(attached.status,200);
  await page.getByRole('button',{name:'Refresh sources',exact:true}).click();
  await page.getByRole('button',{name:'Open Atlas original',exact:true}).click();
  await detail.getByRole('button',{name:'View source images',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.document-detail img')?.naturalWidth===1);
  assert.match(await detail.getByRole('img',{name:'atlas-original.png'}).getAttribute('src'),/^blob:/);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'documents-native.png'),fullPage:true});
  assert.deepEqual(errors,[]);
});

for(const collection of ['manuals','missing'])test(`browser-selected scope reaches Scone and survives reload: ${collection}`,{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{scoped:true});page.setDefaultTimeout(8000);
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('button',{name:'Files',exact:true}).click();
  await dialog.getByLabel('Source prefix',{exact:true}).fill('docs/');
  await dialog.getByText('Dates & metadata',{exact:true}).click();
  await dialog.getByLabel('Created on or after (UTC)',{exact:true}).fill('2026-09-01');
  await dialog.getByLabel('Created on or before (UTC)',{exact:true}).fill('2026-09-03');
  await dialog.getByRole('button',{name:'Add metadata filter',exact:true}).click();
  await dialog.getByLabel('Metadata key 1',{exact:true}).fill('collection');
  await dialog.getByLabel('Metadata value 1',{exact:true}).fill(collection);
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).waitFor();
  const sid=new URL(page.url()).pathname.split('/').at(-1),headers={authorization:'Bearer conversation-fixture-alpha'};
  const expected={kind:'file',source_prefix:'docs/',where:{collection},since:'2026-09-01T00:00:00.000Z',until:'2026-09-03T00:00:00.000Z'};
  for(let turn=0;turn<2;turn++){
    await page.getByLabel('Message',{exact:true}).fill(`${collection==='missing'?'No matching':'Find'} Juniper guide ${turn}`);
    await page.getByRole('button',{name:'Send message',exact:true}).click();
    await page.getByText(collection==='missing'?'Scoped answer: no retrieved context.':'Scoped answer: the selected guide.',{exact:true}).nth(turn).waitFor();
    const current=await(await fetch(base+'/v1/conversations/'+sid,{headers})).json();
    assert.deepEqual(current.recall_scope,expected);
    const receipt=await(await fetch(base+'/v1/conversations/'+sid+'/turns/'+current.latest_request_id,{headers})).json();
    assert.equal(receipt.status,'completed');assert.equal(receipt.result.memory_context.references.length,collection==='missing'?0:1);
    for(const ref of receipt.result.memory_context.references){
      const source=await(await fetch(base+'/v1/episodes/'+ref.episode_id,{headers})).json();
      assert.equal(source.source,'docs/guide.md');assert.equal(source.content,'Juniper scope-eligible-guide.');
    }
  }
  await page.reload();await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByText('Memory selection',{exact:true}).click();
  await page.getByText(`collection = ${collection}`,{exact:true}).waitFor();
  assert.deepEqual(errors,[]);
});

test('a browser text-file import is retained and recalled by the native engine',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);page.setDefaultTimeout(8000);
  await page.goto(base+'/memory');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  const content='\ufeff# Nebulaforge\r\nUse the amber dial for calibration.\r\n';
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'nebulaforge.md',mimeType:'text/markdown',buffer:Buffer.from(content)});
  await page.getByRole('region',{name:'File preview'}).waitFor();
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByRole('heading',{name:/Source saved · episode #/}).waitFor();
  const id=Number((await page.getByRole('heading',{name:/Source saved · episode #/}).textContent()).match(/#(\d+)/)[1]);
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const saved=await(await fetch(base+'/v1/episodes/'+id,{headers})).json();
  assert.equal(saved.content,content);assert.equal(saved.kind,'file');assert.equal(saved.source,'nebulaforge.md');
  assert.equal((await fetch(base+'/v1/episodes/'+id,{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  await page.getByText('Read saved text',{exact:true}).click();
  assert.equal(await page.locator('.source-saved pre').textContent(),content);
  await page.locator('.source-composer').getByRole('button',{name:'Close',exact:true}).click();
  await page.getByPlaceholder('What have we decided about this project?').fill('Nebulaforge');
  await page.getByPlaceholder('What have we decided about this project?').press('Enter');
  await page.locator('.rows .row').filter({hasText:'amber dial'}).first().waitFor();
  assert.equal(await page.locator('.rows .row').getByRole('link',{name:'nebulaforge.md',exact:true}).count(),0,'a local filename is provenance, not a website URL');
  assert.deepEqual(errors,[]);
});

test('the browser pages a native transcript and opens an older original source',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);page.setDefaultTimeout(8000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const response=await fetch(base+'/v1/conversations',{method:'POST',headers,body:JSON.stringify({request_id:'history-fixture',capture:true})});
  assert.equal(response.status,200);const session=await response.json();
  const ids=[];
  // Explicit synthetic history belongs only to this disposable fixture server.
  // This tests navigation through real storage, not agent capture provenance.
  for(let index=1;index<=123;index++){
    const added=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({content:`Archived fixture message ${index}`,kind:'conversation',metadata:{session_id:session.session_id,role:index%2?'user':'assistant'}})});
    assert.equal(added.status,200);ids.push((await added.json()).episode_id);
  }
  let writes=0;page.on('request',request=>{if(request.method()==='POST')writes++;});
  await page.goto(base+'/conversations/'+session.session_id);
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByText('Archived fixture message 123',{exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Saved messages'}).locator('article').count(),50);
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Archived fixture message 24',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Archived fixture message 1',{exact:true}).waitFor();
  await page.getByRole('button',{name:`Inspect message episode ${ids[0]}`,exact:true}).click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.locator('.source-markdown').getByText('Archived fixture message 1',{exact:true}).waitFor();
  assert.equal(await evidence.getByLabel('Original source text',{exact:true}).textContent(),'Archived fixture message 1');
  await page.getByRole('button',{name:'Latest messages',exact:true}).click();
  await page.getByText('Archived fixture message 123',{exact:true}).waitFor();
  assert.equal(writes,0,'reading history must not create or submit work');
  assert.deepEqual(errors,[]);
});

test('the browser cancels a real Scone reply and completes the next question',{timeout:60000},async t=>{
  const {page,base,errors,modelWaiting}=await fixture(t);page.setDefaultTimeout(8000);
  const writes=[];page.on('request',request=>{if(request.method()==='POST')writes.push(new URL(request.url()).pathname);});
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).fill('Wait for cancellation');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await modelWaiting;
  await page.getByRole('button',{name:'Cancel reply',exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByText(/Reply cancelled locally/).waitFor();
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Scripted answer: use Polaris.',{exact:true}).waitFor();
  const sid=new URL(page.url()).pathname.split('/').at(-1),headers={authorization:'Bearer conversation-fixture-alpha'};
  const transcript=await(await fetch(base+'/v1/conversations/'+sid+'/transcript',{headers})).json();
  assert.deepEqual(transcript.episodes.map(e=>e.content),['Wait for cancellation','How is Juniper calibrated?','Scripted answer: use Polaris.']);
  assert.equal(writes.filter(p=>p.endsWith('/turns')).length,2,'the cancelled turn was not resubmitted');
  assert.equal(writes.filter(p=>p.endsWith('/cancel')).length,1);
  assert.equal((await(await fetch(base+'/v1/conversations/'+sid,{headers})).json()).state,'running');
  assert.deepEqual(errors,[]);
});

test('selected persona reaches the real runtime and remains fixed across browser reload',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true,personas:true});page.setDefaultTimeout(8000);
  const creates=[];page.on('request',r=>{if(r.method()==='POST'&&new URL(r.url()).pathname==='/v1/conversations')creates.push(r.postDataJSON());});
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const catalog=await(await fetch(base+'/v1/conversations/personas',{headers})).json();
  assert.equal(catalog.personas.length,2);assert.ok(!JSON.stringify(catalog).includes('Be the'));
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByRole('radio',{name:'Coach',exact:true}).check();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.waitForURL(url=>/^\/conversations\/[^/]+$/.test(url.pathname));
  const sessionUrl=page.url();
  for(let turn=0;turn<2;turn++){
    await page.getByLabel('Session persona',{exact:true}).getByText('Coach',{exact:true}).waitFor();
    await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
    await page.getByRole('button',{name:'Send message',exact:true}).click();
    await page.getByRole('region',{name:'Saved messages'}).getByText('coach: Juniper uses Polaris.',{exact:true}).nth(turn).waitFor();
    if(!turn){await page.reload();await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();}
  }
  assert.equal(page.url(),sessionUrl);
  const sid=new URL(page.url()).pathname.split('/').at(-1);
  const saved=await(await fetch(base+'/v1/conversations/'+sid,{headers})).json();
  assert.deepEqual(saved.persona,{id:'coach',name:'Coach',fingerprint:catalog.personas.find(p=>p.id==='coach').fingerprint,current:true});
  assert.equal(creates.length,1);assert.equal(creates[0].persona_fingerprint,saved.persona.fingerprint);
  assert.equal((await fetch(base+'/v1/conversations/'+sid,{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  const transcript=await(await fetch(base+'/v1/conversations/'+sid+'/transcript',{headers})).json();
  assert.equal(transcript.episodes.length,4);
  assert.deepEqual(errors,[]);
});

test('composed history-only host connects without exposing keys or enabling writes',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed:true,historyOnly:true});page.setDefaultTimeout(8000);
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  assert.equal((await fetch(base+'/v1/conversations/capabilities')).status,401);
  const capabilities=await(await fetch(base+'/v1/conversations/capabilities',{headers})).json();
  assert.equal(capabilities.text_configured,false);
  assert.equal((await(await fetch(base+'/v1/capabilities',{headers})).json()).features.conversations,true);
  const writes=[];page.on('request',r=>{if(!['GET','HEAD'].includes(r.method()))writes.push(r.url());});
  for(const path of ['/memory','/conversations']){
    const response=await fetch(base+path);assert.equal(response.status,200);
    assert.ok(!(await response.text()).includes('conversation-fixture-alpha'));
  }
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('heading',{name:'Text runtime not configured',exact:true}).waitFor();
  await page.getByText('No saved sessions yet.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  assert.equal(await page.getByLabel('Saved session count',{exact:true}).textContent(),'0');
  await page.getByRole('link',{name:/Open memory/}).click();await page.waitForURL(url=>url.origin===base&&url.pathname==='/memory');
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('heading',{name:'Text runtime not configured',exact:true}).waitFor();
  assert.deepEqual(writes,[],'connection, navigation and history discovery must not create work');
  assert.deepEqual(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length})),{local:0,session:0});
  assert.deepEqual(errors,[]);
});

for(const composed of [false,true])test(`conversation deep links, public capture and sources work through native Scone, composed=${composed}`,{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{composed});
  await page.goto(base+'/conversations');
  assert.equal(await page.getByRole('dialog',{name:'Memory connection'}).count(),1);
  assert.ok(!(await page.content()).includes('conversation-fixture-alpha'),'public page must not contain a space key');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.waitForURL(/\/conversations\/[^/]+$/);
  const sessionUrl=page.url(),sid=new URL(sessionUrl).pathname.split('/').at(-1);
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Scripted answer: use Polaris.',{exact:true}).waitFor();
  const messages=page.getByLabel('Saved messages',{exact:true});
  assert.equal(await messages.locator('article').count(),2);
  await page.getByLabel('Message',{exact:true}).fill('What was my question?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Scripted follow-up: you asked about Juniper calibration.',{exact:true}).waitFor();
  assert.equal(await messages.locator('article').count(),4);
  await messages.locator('article').first().getByRole('button').click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.locator('.source-markdown').getByText('How is Juniper calibrated?',{exact:true}).waitFor();
  assert.equal(await evidence.getByLabel('Original source text',{exact:true}).textContent(),'How is Juniper calibrated?');
  assert.ok(!(await page.innerText('body')).includes('private-fixture-thought'));
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const api='/v1/conversations/'+sid;
  const transcript=await(await fetch(base+api+'/transcript',{headers})).json();
  assert.deepEqual(transcript.episodes.map(e=>e.content),[
    'How is Juniper calibrated?','Scripted answer: use Polaris.',
    'What was my question?','Scripted follow-up: you asked about Juniper calibration.',
  ]);
  assert.equal((await fetch(base+api+'/transcript',{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  await page.reload();
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await messages.locator('article').nth(3).waitFor();
  assert.equal(page.url(),sessionUrl,'direct session refresh preserves the address');
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).isDisabled(),true);
  assert.equal((await(await fetch(base+api+'/transcript',{headers})).json()).episodes.length,4,'refresh and stop do not replay a turn');
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const deletion=page.getByRole('dialog',{name:'Delete this conversation?'});
  await deletion.getByLabel('I understand this cannot be undone').check();
  await deletion.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await page.waitForURL(base+'/conversations');
  await page.getByText('Conversation deleted.',{exact:true}).waitFor();
  assert.equal((await fetch(base+api,{headers})).status,404);
  for(const episode of transcript.episodes)assert.equal((await fetch(base+'/v1/episodes/'+episode.episode_id,{headers})).status,410);
  const original=await(await fetch(base+'/v1/episodes/1',{headers})).json();
  assert.equal(original.content,'Juniper is calibrated with Polaris.','deleting a conversation must not delete its imported context');
  assert.equal(await page.getByRole('navigation',{name:'Saved conversations'}).getByRole('link').count(),0);
  assert.deepEqual(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length})),{local:0,session:0});
  assert.deepEqual(errors,[]);
});

for(const composed of [false,true])test(`recreated native service recovers the actual latest outcome without a model or browser cache, composed=${composed}`,{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{reopened:true,composed});
  page.setDefaultTimeout(7000);
  const writes=[];page.on('request',request=>{if(request.method()!=='GET')writes.push(request.method()+' '+new URL(request.url()).pathname);});
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('navigation',{name:'Saved conversations'}).getByRole('link').click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  const url=page.url(),sid=new URL(url).pathname.split('/').at(-1);
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const endpoint=base+'/v1/conversations/'+sid;
  const session=await(await fetch(endpoint,{headers})).json();
  assert.equal(session.latest_request_id,'a-newer','latest must come from chronology, not UUID order');
  assert.equal(session.active_request_id,null);
  assert.equal((await(await fetch(endpoint+'/transcript',{headers})).json()).episodes.length,3);
  assert.equal((await fetch(endpoint+'/turns/a-newer',{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  await page.reload();
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  assert.equal(page.url(),url);
  assert.equal(await page.getByText('Scripted follow-up: you asked about Juniper calibration.',{exact:true}).count(),0);
  assert.deepEqual(writes,[],'reopening only reads; no create/send/stop commands');
  assert.deepEqual(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length})),{local:0,session:0});
  assert.deepEqual(errors,[]);
});
