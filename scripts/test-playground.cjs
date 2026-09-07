// HTTP-boundary fixtures exercise the shipped browser, not a mocked renderer.
const {test, before, after} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {createHash} = require('node:crypto');
const {chromium} = require(process.env.SCONE_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const capabilityContract = JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/http-capabilities.json'),'utf8'));
let browser;
before(async () => { browser = await chromium.launch({headless:true, executablePath:process.env.SCONE_BROWSER_PATH, args:['--disable-gpu']}); });
after(async () => { await browser?.close(); });
const nodes = [
  {id:'session:codex:s1',kind:'session',label:'Codex · launch',data:{agent:'codex',session_id:'s1',project:'launch'}},
  {id:'turn:1',kind:'turn',label:'prompt',ts:'2026-09-06T03:00:00Z',data:{event:'prompt',text:'Keep launch local <script>bad()</script>',agent:'codex'}},
  {id:'episode:7',kind:'episode',label:'Keep launch local',data:{content:'Keep launch local',episode_id:7}},
  {id:'chunk:9',kind:'chunk',label:'Keep launch local',data:{text:'Keep launch local'}},
  {id:'recall:12',kind:'recall',label:'launch',data:{items:1}},
];
const edges = [
  {source:'session:codex:s1',target:'turn:1',kind:'has'},
  {source:'turn:1',target:'episode:7',kind:'captured_as'},
  {source:'episode:7',target:'chunk:9',kind:'chunked_into'},
  {source:'recall:12',target:'chunk:9',kind:'returned'},
];
async function chooseLayout(page,mode) {
  await page.getByRole('button',{name:'Graph layout',exact:true}).click();
  await page.getByRole('menuitemradio',{name:({constellation:'Constellation',radial:'Radial clusters',flow:'Evidence flow',growth:'Growth spiral'})[mode],exact:true}).click();
}
async function fixture(t, {empty=false,mobile=false,dev=false,noKey=false,memory=false,crowded=false,review=false,backlog=false,chronological=false,paged=false,beliefs=false,claimBacklog=false,markdownQuote=false,derivedReview=false,capabilityMode='ok',rustCapabilities=false,profileAdvertised}={}) {
  let graphCalls=0, unauthorized=false, revision=1, pageLoads=0;
  let pendingFacts=[{fact_id:41,subject:'Ada',predicate:'prefers',object:'local storage',confidence:1,valid_from:'2026-09-06T03:00:00Z',valid_until:null,status:'proposed',source_episode_id:7,origin:'extracted',quote:null,grounded:false}];
  if(backlog) pendingFacts=[
    {...pendingFacts[0],fact_id:43,subject:'Launch',object:'offline mode'},
    {...pendingFacts[0],fact_id:42,object:'encrypted storage',grounded:true,quote:'encrypted storage'},
    {...pendingFacts[0],fact_id:41},
  ];
  if(chronological) pendingFacts=pendingFacts.map(f=>({...f,valid_from:f.fact_id===42?'2020-01-01T00:00:00Z':'2030-01-01T00:00:00Z'}));
  if(paged) pendingFacts=Array.from({length:26},(_,i)=>({...pendingFacts[0],fact_id:41+i}));
  if(markdownQuote) pendingFacts=pendingFacts.map(f=>({...f,quote:'and `target` held at **4.0 GB** afterward · ![proof](https://tracking.invalid/proof.png)',grounded:true}));
  if(derivedReview) pendingFacts=[...pendingFacts,{...pendingFacts[0],fact_id:44,origin:'inferred',source_episode_id:null,quote:null,grounded:null,object:'storage under her control'}];
  let ledger=[{...pendingFacts[0],status:'active',excluded_reason:null,closed_reason:null}];
  if(claimBacklog)ledger=[...ledger,
    {...ledger[0],fact_id:40,object:'cloud storage',status:'closed',valid_from:'2019-01-01T00:00:00Z',valid_until:'2020-01-01T00:00:00Z'},
    ...Array.from({length:26},(_,i)=>({...ledger[0],fact_id:100+i,subject:`Topic ${String(i).padStart(2,'0')}`,valid_from:'2020-01-01T00:00:00Z',quote:i?'local storage':null,grounded:i?true:false,excluded_reason:i===25?'wrong source':null})),
  ];
  let sourceCalls=0;
  const decisions=[];
  let capabilityCalls=0; const requested=[];
  let graphNodes = crowded ? [{id:'session:codex:busy',kind:'session',label:'Codex busy',data:{agent:'codex',session_id:'busy'}}, ...Array.from({length:240},(_,i)=>({id:`tool:${i}`,kind:'tool_call',label:'Bash',ts:`2026-09-06T03:00:${String(i%60).padStart(2,'0')}Z`,data:{agent:'codex'}}))] : nodes;
  let graphEdges = crowded ? graphNodes.slice(1).map(n=>({source:'session:codex:busy',target:n.id,kind:'invoked'})) : edges;
  const file=process.env.SCONE_PLAYGROUND_HTML || path.join(root,'crates/scone/src/playground.html');
  // Existing console is the pre-feature baseline until the playground exists.
  const html=fs.readFileSync(file,'utf8').replaceAll('__SCONE_TOKEN__',noKey?'__SCONE_TOKEN__':'fixture-key');
  const server=http.createServer((req,res)=>{
    requested.push(req.url);
    res.setHeader('Content-Type','application/json');
    if(req.url.startsWith('/playground')||req.url.startsWith('/memory')){res.setHeader('Content-Type','text/html');res.setHeader('ETag','"ui-'+revision+'"');if(req.method!=='HEAD')pageLoads++;return res.end(req.method==='HEAD'?'':html);}
    if(req.headers.authorization!=='Bearer fixture-key'||unauthorized){res.statusCode=401;return res.end('{"error":"unauthorized"}');}
    if(req.url==='/v1/capabilities'){
      capabilityCalls++;
      if(capabilityMode==='offline'){res.statusCode=503;return res.end('{"error":"Capability service unavailable"}');}
      if(capabilityMode==='missing'){res.statusCode=404;return res.end('{"error":"Not found"}');}
      if(capabilityMode==='invalid')return res.end('{"schema_version":1,"features":{"facts.review":"false"}}');
      const contract=capabilityContract[rustCapabilities?'rust':'python'];
      return res.end(JSON.stringify(profileAdvertised===undefined?contract:{...contract,features:{...contract.features,'profile.read':profileAdvertised}}));
    }
    if(req.url.startsWith('/v1/status'))return res.end(JSON.stringify({space:'launch',name:'launch',episodes:1}));
    if(beliefs && req.url==='/v1/facts?all=true&excluded=true')return res.end(JSON.stringify({facts:ledger}));
    if(beliefs && req.method==='POST' && /^\/v1\/facts\/41\/(close|exclude|include)$/.test(req.url)){
      let body='';req.on('data',chunk=>body+=chunk);req.on('end',()=>{
        const data=body?JSON.parse(body):null; decisions.push({path:req.url,body:data});
        const action=req.url.split('/').at(-1);
        ledger=ledger.map(f=>action==='close'?{...f,status:'closed',closed_reason:data.reason,valid_until:new Date().toISOString()}:{...f,excluded_reason:action==='exclude'?data.reason:null});
        res.end(JSON.stringify(action==='close'?{closed:41,reason:data.reason}:ledger[0]));
      });return;
    }
    if(review && req.url==='/v1/facts?status=proposed')return res.end(JSON.stringify({facts:pendingFacts}));
    if((review||beliefs) && req.url==='/v1/episodes/7'){sourceCalls++;return res.end(JSON.stringify({episode_id:7,kind:'conversation',source:'fixture',tags:[],metadata:{},content:'Ada prefers local storage. '+('Recorded source context. '.repeat(50))+'END OF SOURCE',created_at:'2026-09-06T03:00:00Z'}));}
    if(review && req.method==='POST' && /^\/v1\/facts\/\d+\/(approve|decline)$/.test(req.url)){
      const id=Number(req.url.split('/')[3]);
      let body='';req.on('data',chunk=>body+=chunk);req.on('end',()=>{decisions.push({path:req.url,body:body?JSON.parse(body):null});pendingFacts=pendingFacts.filter(f=>f.fact_id!==id);res.end(JSON.stringify({fact_id:id,status:req.url.endsWith('approve')?'active':'declined'}));});return;
    }
    if(req.url.startsWith('/v1/graph')){graphCalls++;return res.end(JSON.stringify({nodes:empty?[]:graphNodes,edges:empty?[]:graphEdges,truncated:false,coverage:{agent:'connector-reported'}}));}
    if(req.url.startsWith('/v1/events'))return res.end(JSON.stringify({events:[],next_after:0,has_more:false}));
    if(req.url.startsWith('/v1/recall'))return res.end(JSON.stringify({items:[{episode_id:7,chunk_id:9,text:'Keep launch local',score:1}],facts:[],degraded:[],event_id:12}));
    res.statusCode=404;res.end('{}');
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000}, reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(r=>server.close(r));assert.deepEqual(errors,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/${memory||review||beliefs?'memory':'playground'}${dev?'?dev=1':''}${review?'#review':beliefs?'#beliefs':''}`);
  return {page,decisions,requested,capabilityCalls:()=>capabilityCalls,restoreCapabilities:()=>{capabilityMode='ok';},sourceCalls:()=>sourceCalls,addProposal:()=>pendingFacts.push({...pendingFacts[0],fact_id:99,object:'new arrival'}),calls:()=>graphCalls,expire:()=>{unauthorized=true;},revise:()=>{revision++;},loads:()=>pageLoads,grow:()=>{
    graphNodes=[...nodes];graphEdges=[...edges];
    for(let s=0;s<3;s++){const id=`session:new:${s}`;graphNodes.push({id,kind:'session',label:'New session',data:{agent:'codex'}});for(let i=0;i<8;i++){const target=`new:${s}:${i}`;graphNodes.push({id:target,kind:'tool_call',label:'Bash'});graphEdges.push({source:id,target,kind:'invoked'});}}
  }};
}
for (const width of [1440,1024,390]) test(`global navigation sits above a single contextual sidebar at ${width}px`,async t=>{
  const {page}=await fixture(t);
  await page.setViewportSize({width,height:1000});
  const nav=page.getByRole('navigation',{name:'Workspace',exact:true});
  await nav.waitFor();
  const links=await nav.locator('a').all();
  const boxes=await Promise.all(links.map(link=>link.boundingBox()));
  assert.ok(boxes.every(box=>box&&Math.abs(box.y-boxes[0].y)<2),'Global destinations share one horizontal row.');
  const pane=await page.locator('.app-pane').boundingBox();
  assert.ok(pane.x<2,'There is no global left navigation column.');
  const rail=await page.locator('.workspace .rail').boundingBox();
  if(width>800)assert.ok(rail.x<2,'The session rail is the only left sidebar.');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await nav.getByRole('link',{name:'Memory',exact:true}).click();
  await page.locator('.memory-page').waitFor();
  assert.equal(await nav.getByRole('link',{name:'Memory',exact:true}).getAttribute('aria-current'),'page');
});

for (const mobile of [false,true]) test(`workspace action controls keep a consistent usable size, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{review:true,backlog:true,mobile});
  await page.locator('.proposal').first().waitFor();
  const controls=[page.getByRole('button',{name:'Refresh queue',exact:true}),page.getByRole('button',{name:'Approve',exact:true}).first(),page.locator('.connection-button')];
  const measures=[];
  for(const control of controls)measures.push(await control.evaluate(el=>{
    const s=getComputedStyle(el);return {height:el.getBoundingClientRect().height,radius:s.borderRadius,font:s.fontSize};
  }));
  for(const measure of measures){
    assert.ok(measure.height>=36,'Actions must remain easy to target, including compact actions.');
    assert.deepEqual(measure,measures[0],'Action styling must not depend on the route or legacy button class.');
  }
  await page.getByRole('button',{name:'Approve',exact:true}).first().click();
  await page.waitForFunction(()=>document.querySelectorAll('.proposal').length===2);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});

for (const mobile of [false,true]) test(`memory search has a labeled query and real overview metrics, mobile=${mobile}`,async t=>{
  const {page,requested}=await fixture(t,{memory:true,mobile});
  await page.route('**/v1/status',route=>route.fulfill({json:{space:'launch',episodes:12,chunks:34,pending_review:3}}));
  await page.reload();
  const query=page.getByRole('searchbox',{name:'Search your memory',exact:true});
  await query.waitFor({timeout:2500});
  await page.getByRole('region',{name:'Memory overview'}).waitFor();
  const metrics=await page.locator('.memory-overview div').evaluateAll(els=>els.map(el=>[el.querySelector('dt')?.textContent,el.querySelector('dd')?.textContent]));
  assert.deepEqual(metrics,[['Sources','12'],['Excerpts','34'],['Awaiting review','3']]);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`workspace-search-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  await query.fill('release criteria');
  const result=page.waitForResponse(response=>response.url().includes('/v1/recall?')&&new URL(response.url()).searchParams.get('q')==='release criteria');
  await query.press('Enter');
  await result;
  assert.ok(requested.some(url=>url.startsWith('/v1/recall?')&&new URL(url,'http://fixture').searchParams.get('q')==='release criteria'));
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});

test('empty memory offers working browser destinations without inventing data',async t=>{
  const {page}=await fixture(t,{memory:true});
  await page.route('**/v1/status',route=>route.fulfill({json:{space:'launch',episodes:0,chunks:0}}));
  await page.reload();
  await page.getByRole('heading',{name:'Your memory starts with a source'}).waitFor({timeout:2500});
  assert.equal(await page.locator('.memory-overview').count(),0);
  await page.getByRole('link',{name:'Connect an agent',exact:true}).click();
  await page.locator('#graph').waitFor();
  assert.match(page.url(),/\/playground$/);
});

for(const count of [undefined,null])test(`an unreported source count does not hide retained results, count=${count}`,async t=>{
  const {page}=await fixture(t,{memory:true});
  await page.route('**/v1/status',route=>route.fulfill({json:{space:'launch',episodes:count}}));
  await page.reload();
  await page.getByRole('region',{name:'Memory overview'}).waitFor({timeout:2500});
  assert.equal(await page.getByRole('heading',{name:'Your memory starts with a source'}).count(),0);
  assert.match(await page.locator('.rows').innerText(),/Keep launch local/);
  assert.equal(await page.locator('.memory-overview dd').textContent(),'Not reported');
});

for (const mobile of [false,true]) test(`shared workspace navigation, palette and canvas remain coherent, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{review:true,backlog:true,mobile});
  await page.locator('.proposal').first().waitFor();
  const nav=page.getByRole('navigation',{name:'Workspace',exact:true});
  assert.equal(await nav.locator('svg[aria-hidden="true"]').count(),4);
  const navigation=await nav.boundingBox();
  for(const link of await nav.getByRole('link').all()){
    const box=await link.boundingBox();
    assert.ok(box.x>=navigation.x&&box.x+box.width<=navigation.x+navigation.width+1,'all primary destinations are visible without horizontal scrolling');
    assert.ok(box.height>=42,'primary navigation keeps usable touch targets');
  }
  const primary=await page.getByRole('button',{name:'Approve',exact:true}).first().evaluate(el=>getComputedStyle(el).backgroundColor);
  const sections=await page.getByRole('navigation',{name:'Sections',exact:true}).boundingBox();
  const content=await page.locator('.memory-page main').boundingBox();
  if(!mobile)assert.ok(sections.x+sections.width<=content.x+1,'section navigation is a stable left rail');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(!mobile){
    for(const width of [1280,1024,900,800,600,390]){
      await page.setViewportSize({width,height:1000});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,`review stays in viewport at ${width}px`);
      for(const link of await nav.getByRole('link').all()){
        const box=await link.boundingBox();
        assert.ok(box.x>=0&&box.x+box.width<=width,`primary navigation stays visible at ${width}px`);
      }
    }
    await page.setViewportSize({width:1440,height:1000});
  }
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`design-review-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  await nav.getByRole('link',{name:'Playground',exact:true}).click();
  await page.locator('[data-node]').first().waitFor();
  assert.equal(await page.locator('.recall-form .primary').evaluate(el=>getComputedStyle(el).backgroundColor),primary);
  let canvas;
  for(const mode of ['constellation','radial','flow','growth']){
    await chooseLayout(page,mode);
    const color=await page.locator('.graph-panel').evaluate(el=>getComputedStyle(el).backgroundColor);
    if(canvas)assert.equal(color,canvas,'changing layout must not switch the workspace palette');
    canvas=color;
    if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`design-${mode}-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  }
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});
test('graph node surfaces and hover states follow the shared palette',async t=>{
  const {page}=await fixture(t);
  await page.locator('[data-node]').first().waitFor();
  // A feature-level literal used to leave old cream surfaces and white outlines
  // behind when the surrounding workspace palette changed.
  await page.evaluate(()=>{
    document.documentElement.style.setProperty('--surface','rgb(241, 246, 252)');
    document.documentElement.style.setProperty('--raised','rgb(222, 233, 247)');
  });
  await chooseLayout(page,'constellation');
  assert.equal(await page.locator('.orb-surface').first().evaluate(el=>getComputedStyle(el).stroke),'rgb(241, 246, 252)');
  await chooseLayout(page,'flow');
  const card=page.locator('.atlas-card:not(.session-anchor)').first();
  await card.hover();
  assert.equal(await card.locator('.card-surface').evaluate(el=>getComputedStyle(el).fill),'rgb(222, 233, 247)');
});

for(const mobile of [false,true])test(`processing overview separates stored sources, extraction, inference and review, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{mobile});const writes=[];
  page.on('request',request=>{if(request.method()!=='GET')writes.push(request.url());});
  await page.route('**/v1/status',route=>route.fulfill({json:{space:'launch',episodes:12,chunks:30,pending_distill:4,pending_review:7,pending_derivation:2,derivation:'off',semantic_lane:'stopped',revision:5}}));
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory processing'});
  await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  assert.match(await panel.getByRole('article',{name:'Inference'}).innerText(),/2[\s\S]*groups[\s\S]*Off/);
  assert.match(await panel.getByRole('article',{name:'Extraction'}).innerText(),/4[\s\S]*episodes[\s\S]*Stopped/);
  assert.match(await panel.getByRole('article',{name:'Review'}).innerText(),/7[\s\S]*proposals/);
  assert.match(await panel.innerText(),/not a per-document readiness receipt/);
  await panel.getByRole('link',{name:'Open review'}).click();
  await page.getByRole('heading',{name:'Review',exact:true}).waitFor();
  assert.equal(new URL(page.url()).hash,'#review');assert.deepEqual(writes,[]);
  await page.goto(new URL('/memory#status',page.url()).href);await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`processing-${mobile?'mobile':'desktop'}.png`),fullPage:true});
});
test('processing hides navigation to unadvertised document and review operations',async t=>{
  const {page}=await fixture(t);
  await page.route('**/v1/capabilities',async route=>{const response=await route.fetch();const caps=await response.json();caps.features['episodes.list']=false;caps.features['facts.review']=false;await route.fulfill({json:caps});});
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory processing'});
  await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  assert.equal(await panel.getByRole('link',{name:'Open documents'}).count(),0);
  assert.equal(await panel.getByRole('link',{name:'Open review'}).count(),0);
});
test('processing navigation cancels a refresh without restoring stale counts',async t=>{
  const {page}=await fixture(t);let waiting=false,release;
  const held=new Promise(resolve=>{release=resolve;});
  await page.route('**/v1/status',async route=>{if(waiting)await held;await route.fulfill({json:{space:'launch',episodes:18}}).catch(()=>{});});
  await page.goto(new URL('/memory#status',page.url()).href);
  await page.getByRole('article',{name:'Sources'}).waitFor();
  waiting=true;await page.getByRole('button',{name:'Refresh status',exact:true}).click();
  await page.getByRole('button',{name:'Reading status…',exact:true}).waitFor();
  assert.equal(await page.getByRole('article',{name:'Sources'}).count(),0);
  await page.locator('nav.sections').getByRole('button',{name:'Search',exact:true}).click();
  release();await page.getByRole('heading',{name:'Search',exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Memory processing'}).count(),0);
});
test('processing on Rust displays scoped failures and model configuration without claiming liveness',async t=>{
  const {page}=await fixture(t,{rustCapabilities:true});
  await page.route('**/v1/status',route=>route.fulfill({json:{space:'launch',episodes:2,chunks:3,revision:1,pending_distill:0,failed_distill:2,model:'fixture-extractor',semantic_lane:'active'}}));
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory processing'});
  await panel.getByRole('heading',{name:'Extraction',exact:true}).waitFor();
  assert.match(await panel.getByRole('article',{name:'Extraction'}).innerText(),/0[\s\S]*episodes awaiting extraction[\s\S]*Configured/);
  assert.match(await panel.getByRole('article',{name:'Extraction'}).innerText(),/2 failed episodes/);
  assert.doesNotMatch(await panel.innerText(),/server-wide/);
  await page.getByText('fixture-extractor',{exact:true}).waitFor();
  assert.match(await panel.getByRole('article',{name:'Inference'}).innerText(),/Not reported/);
  assert.equal(await page.getByRole('button',{name:'Run integrity check',exact:true}).count(),0);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'processing-failures.png'),fullPage:true});
});
for(const model of [null,undefined])test(`processing distinguishes absent model configuration from an unreported field, model=${model}`,async t=>{
  const {page}=await fixture(t,{rustCapabilities:true});
  await page.route('**/v1/status',route=>route.fulfill({json:{space:'launch',episodes:0,pending_distill:0,failed_distill:0,semantic_lane:'paused',model}}));
  await page.goto(new URL('/memory#status',page.url()).href);
  const row=page.locator('.processing-storage dl>div').filter({has:page.getByText('Extraction model',{exact:true})});
  await row.waitFor();assert.equal(await row.locator('dd').innerText(),model===null?'Not configured':'Not reported');
  assert.equal(await page.getByRole('article',{name:'Extraction'}).locator('.processing-mode').innerText(),model===null?'Not configured':'Paused');
  assert.match(await page.getByRole('article',{name:'Extraction'}).innerText(),/0 failed episodes/);
  assert.equal(await page.locator('.processing-failure').count(),0);
});
test('processing refresh clears stale counts and retries malformed or failed reports',async t=>{
  const {page}=await fixture(t);let mode='good';
  await page.route('**/v1/status',route=>route.fulfill(mode==='failed'?{status:503,json:{error:'Status unavailable'}}:{json:mode==='bad'?{space:'launch',pending_distill:-2}:{space:'launch',episodes:42}}));
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory processing'});
  await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  assert.match(await panel.getByRole('article',{name:'Inference'}).innerText(),/Not reported/);
  for(const failure of ['failed','bad']){
    mode=failure;await page.getByRole('button',{name:'Refresh status',exact:true}).click();
    await page.getByRole('alert').filter({hasText:/Status unavailable|Invalid memory status/}).waitFor();
    assert.equal(await page.getByRole('article',{name:'Sources'}).count(),0);
    mode='good';await page.getByRole('button',{name:'Refresh status',exact:true}).click();await panel.getByRole('heading',{name:'Inference',exact:true}).waitFor();
  }
});

const maintenanceReport={space:'launch',scope:'distill',episodes:3,proposed:2,accepted:1,closed:1,skipped:0,parked:0,rejected:1,rejected_reasons:{missing_quote:1},expired:1,derived_sent:1,derived_proposed:1,derived_restated:0,derived_rejected:0,error:null,latency_ms:25};
async function processingActions(t){
  const setup=await fixture(t,{mobile:true});const {page}=setup;
  await page.route('**/v1/capabilities',async route=>{const response=await route.fetch();const caps=await response.json();caps.features['processing.distill']=true;caps.features['processing.derive']=true;await route.fulfill({json:caps});});
  await page.goto(new URL('/memory#status',page.url()).href);
  return setup;
}
test('processing maintenance requires confirmation and reports all effects including partial failure',async t=>{
  const {page}=await processingActions(t);const writes=[];
  await page.route('**/v1/consolidate',route=>{writes.push(route.request().postDataJSON());return route.fulfill({json:{...maintenanceReport,error:'Derivation did not finish'}});});
  await page.getByRole('button',{name:'Run maintenance',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Confirm maintenance pass'});
  await dialog.getByText(/retention.*forget/i).waitFor();assert.equal(writes.length,0);
  await dialog.getByRole('button',{name:'Cancel',exact:true}).click();assert.equal(writes.length,0);
  await page.getByRole('button',{name:'Run maintenance',exact:true}).click();
  await dialog.getByRole('button',{name:'Run one maintenance pass',exact:true}).click();
  const report=page.getByRole('region',{name:'Processing pass result'});
  await report.getByRole('heading',{name:'Pass reported an error',exact:true}).waitFor();
  await report.getByText('Derivation did not finish',{exact:true}).waitFor();
  assert.deepEqual(writes,[{scope:'distill'}]);
  for(const [label,value] of [['Proposals added','2'],['Claims accepted','1'],['Sources expired','1'],['Inferences proposed','1']]){
    const row=report.locator('dl>div').filter({has:page.getByText(label,{exact:true})});assert.equal(await row.locator('dd').innerText(),value);
  }
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});
for(const status of [503,422])test(`processing inference is explicit and an unconfirmed ${status} response is never auto-retried`,async t=>{
  const {page}=await processingActions(t);let writes=0;
  await page.route('**/v1/consolidate',route=>{writes++;assert.deepEqual(route.request().postDataJSON(),{scope:'derive'});return route.fulfill({status,json:{error:'Response interrupted'}});});
  await page.getByRole('button',{name:'Run inference',exact:true}).click();
  await page.getByRole('dialog',{name:'Confirm inference pass'}).getByRole('button',{name:'Run one inference pass',exact:true}).click();
  await page.getByRole('heading',{name:'Pass outcome unconfirmed',exact:true}).waitFor();assert.equal(writes,1);
  assert.equal(await page.getByRole('button',{name:'Run inference',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Run maintenance',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Acknowledge unconfirmed outcome',exact:true}).click();
  assert.equal(writes,1,'Acknowledging must not replay a request');
});
test('processing actions remain absent without explicit capabilities and never probe the pass endpoint',async t=>{
  const {page}=await fixture(t);let calls=0;page.on('request',r=>{if(r.url().includes('/v1/consolidate'))calls++;});
  await page.goto(new URL('/memory#status',page.url()).href);await page.getByRole('heading',{name:'Memory processing',exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:/Run maintenance|Run inference/}).count(),0);assert.equal(calls,0);
});

const integrityClean={space:'launch',episodes:4,chunks:8,facts:3,links:2,tombstones:1,
  chunks_without_episode:[],vectors_without_chunk:[],facts_citing_forgotten:[],
  facts_citing_unknown:[],links_with_missing_ends:[],attachments_unlinked:[],not_inspected:[],healthy:true};
for(const partial of [false,true])test(`integrity is an on-demand read with explicit coverage, partial=${partial}`,async t=>{
  const {page}=await fixture(t);let checks=0;
  await page.route('**/v1/doctor',route=>{checks++;assert.equal(route.request().method(),'GET');return route.fulfill({json:partial?{...integrityClean,vectors_without_chunk:null,not_inspected:['vectors']}:integrityClean});});
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory integrity'});
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).waitFor();assert.equal(checks,0);
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('heading',{name:partial?'Partial check':'No dangling references found',exact:true}).waitFor();
  assert.equal(checks,1);
  if(partial)await panel.getByText('Not inspected: vectors',{exact:true}).waitFor();
  assert.equal(await panel.getByRole('button',{name:/repair|delete|forget/i}).count(),0);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`integrity-${partial?'partial':'clean'}.png`),fullPage:true});
});
test('integrity findings distinguish forgotten sources and page actual IDs on mobile',async t=>{
  const {page}=await fixture(t,{mobile:true});
  await page.route('**/v1/doctor',route=>route.fulfill({json:{...integrityClean,healthy:false,facts_citing_forgotten:Array.from({length:55},(_,i)=>i+100),facts_citing_unknown:[77]}}));
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory integrity'});
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('heading',{name:'References need attention',exact:true}).waitFor();
  await panel.getByText('Claims with forgotten sources',{exact:false}).click();
  const group=panel.locator('details').filter({hasText:'Claims with forgotten sources'});
  assert.equal(await group.locator('li').count(),50);
  await group.getByRole('button',{name:'Show more IDs',exact:true}).click();assert.equal(await group.locator('li').count(),55);
  assert.equal(await group.locator('li').last().innerText(),'Claim #154');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'integrity-findings-mobile.png'),fullPage:true});
});
test('integrity rejects a wrong-space report and retries without any write',async t=>{
  const {page}=await fixture(t);let checks=0;
  await page.route('**/v1/doctor',route=>{assert.equal(route.request().method(),'GET');return route.fulfill({json:{...integrityClean,space:++checks===1?'another-space':'launch'}});});
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory integrity'});
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('alert').waitFor();assert.equal(await panel.locator('.integrity-counts').count(),0);
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('heading',{name:'No dangling references found',exact:true}).waitFor();assert.equal(checks,2);
});
test('unsupported hosts never show or request an integrity operation',async t=>{
  const {page,requested}=await fixture(t,{rustCapabilities:true});
  await page.goto(new URL('/memory#status',page.url()).href);
  await page.getByRole('heading',{name:'Status',exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Memory integrity'}).count(),0);
  assert.equal(requested.some(path=>path.startsWith('/v1/doctor')),false);
});
test('integrity prevents overlapping checks and discards results after navigation',async t=>{
  const {page}=await fixture(t);let checks=0,release;
  const pending=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  await page.route('**/v1/doctor',async route=>{checks++;await pending;await route.fulfill({json:integrityClean}).catch(()=>{});});
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory integrity'});
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  assert.equal(await panel.getByRole('button',{name:'Checking references…',exact:true}).isDisabled(),true);
  await page.getByRole('navigation',{name:'Sections',exact:true}).getByRole('button',{name:'Search',exact:true}).click();
  release();
  await page.getByRole('navigation',{name:'Sections',exact:true}).getByRole('button',{name:'Status',exact:true}).click();
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).waitFor();
  assert.equal(checks,1);assert.equal(await panel.locator('.integrity-result').count(),0);
});
test('an integrity refresh clears the prior verdict while failure remains retryable',async t=>{
  const {page}=await fixture(t);let fail=false;
  await page.route('**/v1/doctor',route=>route.fulfill(fail?{status:503,json:{error:'Check temporarily unavailable'}}:{json:integrityClean}));
  await page.goto(new URL('/memory#status',page.url()).href);
  const panel=page.getByRole('region',{name:'Memory integrity'});
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('heading',{name:'No dangling references found',exact:true}).waitFor();fail=true;
  await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('alert').waitFor();assert.equal(await panel.locator('.integrity-result').count(),0);
  fail=false;await panel.getByRole('button',{name:'Run integrity check',exact:true}).click();
  await panel.getByRole('heading',{name:'No dangling references found',exact:true}).waitFor();
});

test('capability discovery failure is retryable without probing workflows or losing the route',async t=>{
  const {page,requested,restoreCapabilities,capabilityCalls}=await fixture(t,{review:true,capabilityMode:'offline'});
  await page.getByRole('button',{name:'Retry capabilities',exact:true}).waitFor({timeout:2000});
  assert.match(page.url(),/#review$/);
  assert.equal(requested.some(url=>url.startsWith('/v1/facts')||url.startsWith('/v1/events')||url.startsWith('/v1/metrics')||url.startsWith('/v1/scopes')),false);
  restoreCapabilities();
  await page.getByRole('button',{name:'Retry capabilities',exact:true}).click();
  await page.locator('.proposal').waitFor();
  assert.equal(capabilityCalls(),2);
});
const profileFixture={static_facts:[{fact_id:41,subject:'Ada',predicate:'prefers',object:'local storage',confidence:.8,source_episode_id:7}],dynamic:['Ada prefers **local storage**.'],recent:[{episode_id:7,excerpt:'Ada prefers **local storage**.',created_at:'2026-09-06T03:00:00Z'}]};
async function openProfile(t,{mobile=false,advertised=true}={}){
  const setup=await fixture(t,{review:true,mobile,profileAdvertised:advertised});
  await setup.page.route('**/v1/profile',route=>route.fulfill({json:profileFixture}));
  await setup.page.goto(new URL('/memory#profile',setup.page.url()).href);
  setup.page.setDefaultTimeout(2500);
  return setup;
}
for(const mobile of [false,true])test(`profile separates selected claims from source context and opens exact evidence, mobile=${mobile}`,async t=>{
  const {page,sourceCalls,decisions}=await openProfile(t,{mobile});
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  const recent=page.getByRole('region',{name:'Recent source context',exact:true});
  assert.equal(await recent.getByText('Episode #7',{exact:true}).count(),1);
  assert.equal(sourceCalls(),0,'Profile does not fetch full sources until requested.');
  await recent.getByRole('button',{name:'Read source episode #7',exact:true}).click();
  await recent.locator('.source-markdown').getByText(/END OF SOURCE/).waitFor();
  assert.ok((await recent.locator('.formatted-source').boundingBox()).height<=420,'Expanded source text has a bounded reading pane.');
  assert.equal(sourceCalls(),1);assert.deepEqual(decisions,[]);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`profile-${mobile?'mobile':'desktop'}.png`),fullPage:true});
});
test('memory sections follow fragment navigation and browser history without a page reload',async t=>{
  const {page}=await openProfile(t);
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  await page.goto(new URL('/memory#review',page.url()).href);
  await page.getByRole('heading',{name:'Review',exact:true}).waitFor();
  await page.goBack();
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  await page.goto(new URL('/memory#not-a-section',page.url()).href);
  await page.getByRole('heading',{name:'Search',exact:true}).waitFor();
  assert.match(page.url(),/#search$/);
});
test('skip to workspace preserves the selected memory page and focuses its content',async t=>{
  const {page}=await openProfile(t);
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  const skip=page.getByRole('link',{name:'Skip to workspace',exact:true});
  await skip.focus();await page.keyboard.press('Enter');
  assert.match(page.url(),/#profile$/);
  assert.equal(await page.evaluate(()=>document.activeElement?.id),'main');
  assert.equal(await page.getByRole('region',{name:'Selected claim context',exact:true}).count(),1);
});
test('unsupported profile is not probed or presented as an empty context',async t=>{
  const {page,requested}=await openProfile(t,{advertised:false});
  await page.getByRole('heading',{name:'This page is not available on this server'}).waitFor();
  assert.equal(requested.some(url=>url==='/v1/profile'),false);
  assert.equal(await page.getByRole('button',{name:'Profile',exact:true}).count(),0);
});
test('profile lists every native claim source without duplicate disclosure controls',async t=>{
  const {page,sourceCalls}=await openProfile(t);
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  await page.route('**/v1/profile',route=>route.fulfill({json:{...profileFixture,static_facts:[{...profileFixture.static_facts[0],sources:[7,8]}]}}));
  await page.getByRole('button',{name:'Refresh profile',exact:true}).click();
  const claims=page.getByRole('region',{name:'Selected claim context',exact:true});
  await claims.getByRole('button',{name:'Read source episode #8',exact:true}).waitFor();
  assert.equal(await claims.getByRole('button',{name:'Read source episode #7',exact:true}).count(),1);
  assert.equal(sourceCalls(),0);
});
test('profile refresh clears old evidence and invalid responses recover without writes',async t=>{
  const {page}=await openProfile(t);
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  await page.route('**/v1/profile',route=>route.fulfill({json:{...profileFixture,recent:[]}}));
  await page.getByRole('button',{name:'Refresh profile',exact:true}).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('region',{name:'Selected claim context',exact:true}).count(),0);
  await page.route('**/v1/profile',route=>route.fulfill({json:{static_facts:[],recent:[],dynamic:[]}}));
  await page.getByRole('button',{name:'Retry profile',exact:true}).click();
  await page.getByRole('heading',{name:'No profile context yet',exact:true}).waitFor();
  await page.getByRole('button',{name:'Open documents',exact:true}).click();
  await page.getByRole('heading',{name:'Documents',exact:true}).waitFor();
  assert.match(page.url(),/#documents$/);
});
test('profile evidence rejects mismatched source identities and distinguishes forgotten support',async t=>{
  const {page}=await openProfile(t);
  await page.getByRole('region',{name:'Recent source context',exact:true}).waitFor();
  await page.route('**/v1/episodes/7',route=>route.fulfill({json:{episode_id:99,content:'Wrong private source'}}));
  const recent=page.getByRole('region',{name:'Recent source context',exact:true});
  await recent.getByRole('button',{name:'Read source episode #7',exact:true}).click();
  await recent.getByRole('alert').waitFor();assert.equal(await page.getByText('Wrong private source').count(),0);
  await page.route('**/v1/episodes/7',route=>route.fulfill({status:410,json:{error:'forgotten'}}));
  await recent.getByRole('button',{name:'Retry source',exact:true}).click();
  await recent.getByText('This source was forgotten. Its text is no longer retained.',{exact:true}).waitFor();
});
test('late profile responses cannot replace another memory page',async t=>{
  const {page}=await openProfile(t);let finish,entered;
  await page.getByRole('region',{name:'Selected claim context',exact:true}).waitFor();
  const waiting=new Promise(resolve=>{entered=resolve;});
  await page.route('**/v1/profile',async route=>{entered();await new Promise(resolve=>{finish=resolve;});await route.fulfill({json:profileFixture}).catch(()=>{});});
  await page.getByRole('button',{name:'Refresh profile',exact:true}).click();await waiting;
  await page.getByRole('button',{name:'Search',exact:true}).click();finish();
  await page.getByRole('heading',{name:'Search',exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Profile context',exact:true}).count(),0);
});
for(const mobile of [false,true])test(`unavailable memory is a contained actionable workspace state, mobile=${mobile}`,async t=>{
  const {page,restoreCapabilities}=await fixture(t,{review:true,capabilityMode:'offline',mobile});
  const alert=page.getByRole('alert');await alert.waitFor();
  const box=await alert.boundingBox();
  const treatment=await alert.evaluate(el=>{const s=getComputedStyle(el);return {padding:parseFloat(s.paddingLeft),border:parseFloat(s.borderTopWidth),background:s.backgroundColor};});
  assert.ok(treatment.padding>=18&&treatment.border>=1,'Service failures must have the same intentional containment as working panels.');
  assert.notEqual(treatment.background,'rgba(0, 0, 0, 0)');
  const retry=page.getByRole('button',{name:'Retry capabilities',exact:true});
  const control=await retry.boundingBox();assert.ok(control.x>=box.x&&control.x+control.width<=box.x+box.width);
  restoreCapabilities();await retry.click();await page.locator('.proposal').waitFor();
  assert.equal(await page.getByRole('heading',{name:'Server capabilities unavailable'}).count(),0);
});
test('Rust capabilities enable individual review without implying exclusion or analytics support',async t=>{
  const {page,decisions}=await fixture(t,{review:true,beliefs:true,rustCapabilities:true});
  await page.getByRole('button',{name:'Beliefs',exact:true}).click();
  await page.locator('.belief').waitFor();
  assert.equal(await page.getByRole('button',{name:/^Review/}).count(),1);
  assert.equal(await page.getByRole('button',{name:'Analytics',exact:true}).count(),0);
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Close “local storage”/}).waitFor();
  assert.equal(await page.getByRole('button',{name:/Exclude “local storage”/}).count(),0);
  await page.getByRole('button',{name:/^Review/}).click();
  await page.locator('.proposal').waitFor();
  await page.getByRole('button',{name:'Approve',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.proposal'));
  assert.equal(decisions[0].path,'/v1/facts/41/approve');
});
test('missing or malformed capability contracts remain unknown rather than inferred',async t=>{
  for(const capabilityMode of ['missing','invalid']){
    const {page,requested}=await fixture(t,{review:true,capabilityMode});
    await page.getByRole('button',{name:'Retry capabilities',exact:true}).waitFor({timeout:2000});
    assert.equal(await page.locator('.proposal').count(),0);
    assert.equal(requested.some(url=>url.startsWith('/v1/facts')),false);
  }
});
test('graph wheel gestures have one owner and do not hijack ordinary page scrolling',async t=>{
  const {page}=await fixture(t);
  await page.locator('[data-node]').first().waitFor();
  const before=await page.locator('.zoom output').textContent();
  const dispatch=ctrlKey=>page.locator('#graph').evaluate((el,ctrlKey)=>{
    const event=new WheelEvent('wheel',{deltaY:-100,ctrlKey,bubbles:true,cancelable:true});
    el.dispatchEvent(event);return event.defaultPrevented;
  },ctrlKey);
  assert.equal(await dispatch(false),false,'ordinary scrolling belongs to the page');
  assert.equal(await page.locator('.zoom output').textContent(),before,'ordinary scrolling must not change graph scale');
  assert.equal(await dispatch(true),true,'graph pinch must cancel browser zoom');
  await page.waitForFunction(value=>document.querySelector('.zoom output').textContent!==value,before);
  const after=await page.locator('.zoom output').textContent();
  const outside=await page.locator('h1').evaluate(el=>{const e=new WheelEvent('wheel',{deltaY:-100,ctrlKey:true,bubbles:true,cancelable:true});el.dispatchEvent(e);return e.defaultPrevented;});
  assert.equal(outside,false,'browser zoom remains available outside the canvas');
  assert.equal(await page.locator('.zoom output').textContent(),after);
});
test('belief evidence can be inspected without changing the ledger',async t=>{
  const {page,decisions}=await fixture(t,{beliefs:true});
  await page.locator('.belief').waitFor();
  const evidence=page.getByRole('button',{name:'Read source episode #7',exact:true});
  await evidence.waitFor({timeout:2000});
  await evidence.click();
  await page.locator('.source-markdown').getByText(/END OF SOURCE/).waitFor();
  assert.equal(decisions.length,0);
  assert.equal(await page.locator('.timeline').count(),0,'a duration bar should not present active claims as truth');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'belief-purpose-desktop.png'),fullPage:true});
});
test('claim histories are searchable and paged without changing recall or treating missing quotes as false',async t=>{
  const {page,decisions}=await fixture(t,{beliefs:true,claimBacklog:true});
  await page.getByRole('searchbox',{name:'Search claim histories',exact:true}).waitFor({timeout:2000});
  assert.equal(await page.locator('.belief').count(),25);
  await page.getByRole('navigation',{name:'Claim pages'}).getByRole('button',{name:'Next',exact:true}).click();
  assert.equal(await page.locator('.belief').count(),2);
  await page.getByRole('searchbox',{name:'Search claim histories',exact:true}).fill('cloud');
  await page.getByRole('heading',{name:'Ada prefers cloud storage',exact:true}).waitFor();
  assert.equal(await page.locator('.belief').count(),1);
  assert.match(await page.locator('.claim-heading').innerText(),/Historical version/);
  assert.match(await page.locator('.claim-source').first().innerText(),/No stored quotation/);
  assert.doesNotMatch(await page.innerText('body'),/ungrounded|proven false/i);
  await page.getByRole('button',{name:'Current records',exact:true}).click();
  await page.getByText('No claim histories match these filters.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Clear claim filters',exact:true}).click();
  await page.getByRole('button',{name:'Excluded records',exact:true}).click();
  await page.getByRole('heading',{name:'Topic 25 prefers local storage',exact:true}).waitFor();
  assert.equal(await page.locator('.belief').count(),1);
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.equal(decisions.length,0);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'claims-filter-mobile.png'),fullPage:true});
});
test('belief close uses an inline reason and cancellation does not write',async t=>{
  const {page,decisions}=await fixture(t,{beliefs:true,mobile:true});
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Close “local storage”/}).click();
  const reason=page.getByRole('textbox',{name:'Reason for this change'});
  await reason.waitFor({timeout:2000});
  assert.equal(await page.locator('details.actions').evaluate(el=>el.open),false,'action menu must not cover the inline form');
  assert.equal(await page.getByRole('button',{name:'Confirm close',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Cancel',exact:true}).click();assert.equal(decisions.length,0);
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Close “local storage”/}).click();
  await reason.fill('We switched storage.');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'belief-close-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Confirm close',exact:true}).click();
  await page.getByRole('status').filter({hasText:'Belief #41 closed'}).waitFor();
  assert.deepEqual(decisions,[{path:'/v1/facts/41/close',body:{reason:'We switched storage.'}}]);
  await page.getByText(/Historical version · not current/).waitFor();
});
test('belief failures block repeat decisions until ledger refresh',async t=>{
  const {page,decisions}=await fixture(t,{beliefs:true});
  await page.route('**/v1/facts/41/exclude',route=>route.fulfill({status:503,contentType:'application/json',body:'{"error":"Storage unavailable"}'}));
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Exclude “local storage”/}).click();
  await page.getByRole('textbox',{name:'Reason for this change'}).fill('Not relevant.');
  await page.getByRole('button',{name:'Confirm exclude',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Storage unavailable'}).waitFor({timeout:2000});
  assert.equal(decisions.length,0);
  assert.equal(await page.getByRole('button',{name:'Confirm exclude',exact:true}).isDisabled(),true);
  await page.unroute('**/v1/facts/41/exclude');
  await page.getByRole('button',{name:'Refresh beliefs',exact:true}).click();
  await page.getByRole('button',{name:'Confirm exclude',exact:true}).click();
  await page.getByRole('status').filter({hasText:'excluded from recall'}).waitFor();
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Include “local storage”/}).click();
  assert.equal(decisions.length,1,'include must also require confirmation');
  await page.getByRole('button',{name:'Confirm include',exact:true}).click();
  await page.getByRole('status').filter({hasText:'included in recall'}).waitFor();
  assert.deepEqual(decisions.map(d=>d.path),['/v1/facts/41/exclude','/v1/facts/41/include']);
});
test('belief decisions reject a receipt for a different record',async t=>{
  const {page}=await fixture(t,{beliefs:true});
  await page.route('**/v1/facts/41/close',route=>route.fulfill({status:200,contentType:'application/json',body:'{"closed":42,"reason":"Other record"}'}));
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Close “local storage”/}).click();
  await page.getByRole('textbox',{name:'Reason for this change'}).fill('Changed.');
  await page.getByRole('button',{name:'Confirm close',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'unrecognized decision receipt'}).waitFor();
  assert.equal(await page.locator('.belief-notice').count(),0);
  assert.equal(await page.getByRole('button',{name:'Confirm close',exact:true}).isDisabled(),true);
});
test('belief reconciliation discards an action already applied by the server',async t=>{
  const {page,decisions}=await fixture(t,{beliefs:true});
  await page.route('**/v1/facts/41/exclude',async route=>{await route.fetch();await route.fulfill({status:503,contentType:'application/json',body:'{"error":"Reply lost after write"}'});});
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Exclude “local storage”/}).click();
  await page.getByRole('textbox',{name:'Reason for this change'}).fill('Keep private.');
  await page.getByRole('button',{name:'Confirm exclude',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Reply lost after write'}).waitFor();
  await page.getByRole('button',{name:'Refresh beliefs',exact:true}).click();
  await page.getByText(/excluded from recall, Keep private/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Confirm exclude',exact:true}).count(),0,'stale action must not remain available after its state changes');
  assert.equal(decisions.length,1);
});
test('belief saves lock repeated actions and late replies cannot affect a reopened page',async t=>{
  const {page,decisions}=await fixture(t,{beliefs:true});
  let release;const hold=new Promise(resolve=>release=resolve);t.after(()=>release());
  await page.route('**/v1/facts/41/close',async route=>{await hold;await route.fulfill({status:200,contentType:'application/json',body:'{"closed":41,"reason":"Changed"}'});});
  await page.locator('details.actions summary').click();
  await page.getByRole('button',{name:/Close “local storage”/}).click();
  await page.getByRole('textbox',{name:'Reason for this change'}).fill('Changed.');
  await page.getByRole('button',{name:'Confirm close',exact:true}).click();
  assert.equal(await page.getByRole('button',{name:'Saving…',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:/Exclude “local storage”/,includeHidden:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Status',exact:true}).click();
  release();
  await page.getByRole('button',{name:'Beliefs',exact:true}).click();
  await page.locator('.belief').waitFor();
  assert.equal(await page.locator('.belief-notice').count(),0);
  assert.equal(decisions.length,0);
});
test('review failures stay visible and approval only removes a confirmed decision',async t=>{
  const {page,decisions}=await fixture(t,{review:true});
  await page.route('**/v1/facts/41/approve',route=>route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({error:'Storage unavailable'})}));
  await page.getByRole('button',{name:'Approve',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Storage unavailable'}).waitFor({timeout:2000});
  assert.equal(await page.locator('.proposal').count(),1);
  assert.equal(decisions.length,0);
  await page.unroute('**/v1/facts/41/approve');
  assert.equal(await page.getByRole('button',{name:'Approve',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Refresh queue'}).click();
  await page.getByRole('button',{name:'Approve',exact:true}).click();
  await page.getByText('Nothing awaits review.',{exact:true}).waitFor();
  assert.equal(decisions.length,1);
  assert.equal(decisions[0].path,'/v1/facts/41/approve');
  assert.match(await page.getByRole('status').last().innerText(),/approved/i);
});
test('review sources render Markdown without executing HTML or loading remote images',async t=>{
  const {page,decisions}=await fixture(t,{review:true});
  const source='**Heartbeat tick.**\n\nThe `llama-server` is holding **5.8 GB**.\n\n- First check\n- Second check\n\n```sh\necho "unchanged"\n```\n\n[Provider docs](https://example.com/docs)\n\n[Unsafe](javascript:alert(1))\n\n![tracking pixel](https://tracking.invalid/image.png)\n\n<script>window.sourceExecuted=true</script>';
  const remote=[];page.on('request',request=>{if(request.url().includes('tracking.invalid'))remote.push(request.url());});
  await page.route('**/v1/episodes/7',route=>route.fulfill({json:{episode_id:7,kind:'conversation',source:'fixture',tags:[],metadata:{},content:source,created_at:'2026-09-06'}}));
  await page.getByText('Read full source',{exact:false}).click();
  const content=page.locator('.review-source');
  await content.locator('strong').filter({hasText:'Heartbeat tick.'}).waitFor({timeout:2000});
  assert.equal(await content.locator('code').first().textContent(),'llama-server');
  assert.equal(await content.locator('li').count(),2);
  assert.equal(await content.locator('pre code').textContent(),'echo "unchanged"\n');
  const link=content.getByRole('link',{name:'Provider docs'});
  assert.equal(await link.getAttribute('href'),'https://example.com/docs');
  assert.match(await link.getAttribute('rel'),/noreferrer/);
  assert.equal(await content.locator('a[href^="javascript:"],script,iframe,img').count(),0);
  assert.equal(await page.evaluate(()=>window.sourceExecuted),undefined);
  assert.deepEqual(remote,[]);
  const sourcePalette=await content.evaluate(el=>{
    const probe=document.createElement('span');el.append(probe);
    probe.style.color='var(--accent)';const accent=getComputedStyle(probe).color;
    probe.style.color='var(--muted)';const muted=getComputedStyle(probe).color;
    probe.style.backgroundColor='var(--raised)';const raised=getComputedStyle(probe).backgroundColor;
    probe.remove();
    return {accent,muted,raised,link:getComputedStyle(el.querySelector('a')).color,
      summary:getComputedStyle(el.querySelector('.source-original>summary')).color,
      code:getComputedStyle(el.querySelector('p code')).backgroundColor};
  });
  assert.equal(sourcePalette.link,sourcePalette.accent,'Source links must share the workspace accent.');
  assert.equal(sourcePalette.summary,sourcePalette.muted,'Original-source controls must share workspace text tones.');
  assert.equal(sourcePalette.code,sourcePalette.raised,'Code excerpts must share workspace surfaces.');
  await content.getByText('View original Markdown',{exact:true}).click();
  assert.equal(await content.getByLabel('Original source text',{exact:true}).textContent(),source);
  assert.equal(decisions.length,0,'formatting source text must not make a decision');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'source-markdown.png'),fullPage:true});
});
test('review quotations format emphasis and code while retaining inert image descriptions',async t=>{
  const {page}=await fixture(t,{review:true,markdownQuote:true,mobile:true});
  const quote=page.locator('.proposal blockquote');
  await quote.locator('strong').waitFor();
  assert.equal(await quote.locator('strong').textContent(),'4.0 GB');
  assert.equal(await quote.locator('code').textContent(),'target');
  assert.match(await quote.textContent(),/Image reference: proof · not loaded/);
  assert.equal(await quote.locator('img').count(),0);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
});
for(const mobile of [false,true]) test(`derived review separates premise evidence and bulk decisions, mobile=${mobile}`,async t=>{
  const {page,decisions,requested}=await fixture(t,{review:true,derivedReview:true,mobile});
  const derived=page.locator('.proposal[data-fact-id="44"]');
  await derived.waitFor();
  assert.match(await derived.innerText(),/Inferred from claims/);
  assert.doesNotMatch(await derived.innerText(),/No checked quote|No source episode recorded|Source quote checked/);
  assert.equal(requested.includes('/v1/facts/44'),false,'Premises are fetched only when opened');
  await page.route('**/v1/facts/44',route=>route.fulfill({json:{fact:{fact_id:44,subject:'Ada',predicate:'prefers',object:'storage under her control',status:'proposed',origin:'inferred',source_episode_id:null,quote:null,confidence:.8,valid_from:'2026-09-06T03:00:00Z',valid_until:null},sources:[],links:[
    {link_id:9,from_fact:44,to_fact:41,kind:'derived_from',created_at:'2026-09-06T03:00:00Z',source_episode_id:null,quote:null},
    {link_id:10,from_fact:50,to_fact:44,kind:'derived_from',created_at:'2026-09-06T03:00:00Z',source_episode_id:null,quote:null},
  ]}}));
  await derived.getByRole('button',{name:'Inspect premises for claim #44'}).click();
  await page.getByRole('heading',{name:'Inference premises (1)',exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Derived from · claim #41',exact:true}).count(),1);
  assert.equal(await page.getByRole('button',{name:'Used to derive · claim #50',exact:true}).count(),1);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`inference-premises-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  await page.route('**/v1/facts/41',route=>route.fulfill({json:{fact:{fact_id:41,subject:'Ada',predicate:'prefers',object:'local storage',status:'closed',origin:'extracted',source_episode_id:7,quote:'Ada prefers local storage.',confidence:1,valid_from:'2026-09-05T03:00:00Z',valid_until:'2026-09-06T03:00:00Z'},sources:[7],links:[]}}));
  await page.getByRole('button',{name:'Derived from · claim #41',exact:true}).click();
  await page.getByText('Closed · historical record',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Read source episode #7',exact:true}).click();
  await page.getByRole('paragraph').filter({hasText:'END OF SOURCE'}).waitFor();
  await page.getByRole('button',{name:'Back to previous claim',exact:true}).click();
  await page.getByRole('heading',{name:'Inference premises (1)',exact:true}).waitFor();
  await page.getByRole('button',{name:'Close dialog',exact:true}).click();
  await page.getByRole('combobox',{name:'Evidence',exact:true}).selectOption('inferred');
  assert.deepEqual(await page.locator('.proposal').evaluateAll(rows=>rows.map(row=>row.dataset.factId)),['44']);
  await page.getByRole('combobox',{name:'Evidence',exact:true}).selectOption('unchecked');
  assert.deepEqual(await page.locator('.proposal').evaluateAll(rows=>rows.map(row=>row.dataset.factId)),['41']);
  await page.getByRole('combobox',{name:'Evidence',exact:true}).selectOption('all');
  await page.getByRole('button',{name:'Approve all matching (2)'}).click();
  const dialog=page.getByRole('dialog');
  assert.match(await dialog.innerText(),/1 inferred proposal/);
  assert.match(await dialog.innerText(),/1 other proposal.*no checked source quote/);
  await dialog.getByRole('button',{name:'Cancel',exact:true}).click();
  assert.equal(decisions.length,0,'Inspecting, filtering and cancelling never approves a claim');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`inference-review-${mobile?'mobile':'desktop'}.png`),fullPage:true});
});

test('derived review does not manufacture missing premises or hide read failures',async t=>{
  const {page,decisions}=await fixture(t,{review:true,derivedReview:true});
  let fail=true;
  await page.route('**/v1/facts/44',route=>fail?route.fulfill({status:503,json:{error:'Premise service unavailable'}}):route.fulfill({json:{fact:{fact_id:44,subject:'Ada',predicate:'prefers',object:'storage under her control',status:'proposed',origin:'inferred',confidence:.8,valid_from:'2026-09-06T03:00:00Z',valid_until:null},sources:[],links:[]}}));
  await page.getByRole('button',{name:'Inspect premises for claim #44'}).click();
  await page.getByRole('alert').filter({hasText:'No claim details were confirmed'}).waitFor();
  assert.equal(await page.getByRole('heading',{name:/Inference premises/}).count(),0);
  fail=false;
  await page.getByRole('button',{name:'Refresh claim',exact:true}).click();
  await page.getByRole('heading',{name:'Inference premises (0)',exact:true}).waitFor();
  await page.getByText('No premise links were returned. Support for this inference has not been established here.',{exact:true}).waitFor();
  assert.equal(decisions.length,0);
});

test('derived review names unavailable premise inspection without probing unsupported hosts',async t=>{
  const {page,requested,decisions}=await fixture(t,{review:true,derivedReview:true});
  await page.route('**/v1/capabilities',route=>route.fulfill({json:{...capabilityContract.python,features:{...capabilityContract.python.features,'facts.links':false}}}));
  await page.reload();
  const derived=page.locator('.proposal[data-fact-id="44"]');
  await derived.getByText(/This host does not expose premise inspection/).waitFor();
  assert.equal(await derived.getByRole('button',{name:'Inspect premises for claim #44'}).count(),0);
  assert.equal(requested.includes('/v1/facts/44'),false);
  assert.equal(decisions.length,0);
});

test('derived review expands premises independently of other relationships',async t=>{
  const {page}=await fixture(t,{review:true,derivedReview:true});
  await page.route('**/v1/facts/44',route=>route.fulfill({json:{fact:{fact_id:44,subject:'Ada',predicate:'prefers',object:'storage under her control',status:'proposed',origin:'inferred',confidence:.8,valid_from:'2026-09-06T03:00:00Z',valid_until:null},sources:[],links:Array.from({length:42},(_,i)=>({link_id:i+1,from_fact:i<21?44:100+i,to_fact:i<21?i+1:44,kind:'derived_from',created_at:'2026-09-06T03:00:00Z',source_episode_id:null,quote:null}))}}));
  await page.getByRole('button',{name:'Inspect premises for claim #44'}).click();
  await page.getByRole('button',{name:'Show more premises (20 of 21)',exact:true}).click();
  assert.equal(await page.getByRole('button',{name:/^Derived from · claim #/}).count(),21);
  assert.equal(await page.getByRole('button',{name:/^Used to derive · claim #/}).count(),20,'Expanding premises must not expand another list');
  await page.getByRole('button',{name:'Show more relationships (20 of 21)',exact:true}).click();
  assert.equal(await page.getByRole('button',{name:/^Used to derive · claim #/}).count(),21);
});

test('review backlog groups and filters without eagerly fetching sources',async t=>{
  const {page,sourceCalls}=await fixture(t,{review:true,backlog:true});
  await page.locator('.proposal').first().waitFor();
  assert.equal(sourceCalls(),0,'source reads must wait for inspection');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'review-inbox-desktop.png'),fullPage:true});
  assert.deepEqual(await page.locator('.proposal').evaluateAll(rows=>rows.map(r=>r.getAttribute('data-fact-id'))),['41','42','43']);
  await page.getByRole('button',{name:'Ada, 2 proposals',exact:true}).click();
  assert.equal(await page.locator('.proposal').count(),2);
  await page.getByRole('combobox',{name:'Evidence'}).selectOption('quoted');
  assert.deepEqual(await page.locator('.proposal').evaluateAll(rows=>rows.map(r=>r.getAttribute('data-fact-id'))),['42']);
  await page.getByRole('combobox',{name:'Evidence'}).selectOption('all');
  await page.getByRole('combobox',{name:'Sort queue'}).selectOption('newest');
  assert.deepEqual(await page.locator('.proposal').evaluateAll(rows=>rows.map(r=>r.getAttribute('data-fact-id'))),['42','41']);
  await page.getByRole('searchbox',{name:'Search proposals'}).fill('encrypted');
  assert.equal(await page.locator('.proposal').count(),1);
});
test('bulk review freezes matching IDs and requires confirmation',async t=>{
  const {page,decisions,addProposal}=await fixture(t,{review:true,backlog:true});
  await page.getByRole('button',{name:'Ada, 2 proposals',exact:true}).click();
  await page.getByRole('button',{name:'Approve all matching (2)',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Approve 2 proposals?'});
  await dialog.waitFor(); assert.equal(decisions.length,0);
  addProposal();
  await dialog.getByRole('button',{name:'Confirm approval',exact:true}).click();
  await page.getByRole('status').filter({hasText:'2 approved'}).waitFor();
  assert.deepEqual(decisions.map(d=>d.path),['/v1/facts/41/approve','/v1/facts/42/approve']);
  await page.getByRole('button',{name:/All subjects/}).click();
  await page.locator('.proposal[data-fact-id="99"]').waitFor();
  assert.equal(await page.locator('.proposal').count(),2);
});
test('review abandons a hidden decline form and approves matching records across pages',async t=>{
  const {page,decisions}=await fixture(t,{review:true,paged:true});
  await page.locator('.proposal').first().waitFor();
  assert.equal(await page.locator('.proposal').count(),25);
  await page.locator('.proposal').first().getByRole('button',{name:'Decline',exact:true}).click();
  await page.getByRole('textbox',{name:'Reason for declining'}).fill('Abandoned draft');
  await page.getByRole('button',{name:'Next',exact:true}).click();
  assert.equal(await page.locator('.proposal').count(),1);
  await page.getByRole('button',{name:'Previous',exact:true}).click();
  assert.equal(await page.getByRole('textbox',{name:'Reason for declining'}).count(),0);
  await page.getByRole('button',{name:'Approve all matching (26)',exact:true}).click();
  await page.getByRole('button',{name:'Confirm approval',exact:true}).click();
  await page.getByRole('status').filter({hasText:'26 approved'}).waitFor();
  assert.deepEqual(decisions.map(d=>d.path),Array.from({length:26},(_,i)=>`/v1/facts/${41+i}/approve`));
});
test('bulk review applies earlier beliefs first regardless of the display sort',async t=>{
  const {page,decisions}=await fixture(t,{review:true,backlog:true,chronological:true});
  await page.getByRole('button',{name:'Ada, 2 proposals',exact:true}).click();
  await page.getByRole('button',{name:'Approve all matching (2)',exact:true}).click();
  await page.getByRole('button',{name:'Confirm approval',exact:true}).click();
  await page.getByRole('status').filter({hasText:'2 approved'}).waitFor();
  assert.deepEqual(decisions.map(d=>d.path),['/v1/facts/42/approve','/v1/facts/41/approve']);
});
test('bulk review stops on uncertain failure and reports unattempted records',async t=>{
  const {page,decisions}=await fixture(t,{review:true,backlog:true});
  await page.route('**/v1/facts/42/approve',route=>route.fulfill({status:500,contentType:'application/json',body:'{"error":"Storage unavailable"}'}));
  await page.getByRole('button',{name:'Approve all matching (3)',exact:true}).click();
  await page.getByRole('button',{name:'Confirm approval',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Storage unavailable'}).waitFor();
  assert.deepEqual(decisions.map(d=>d.path),['/v1/facts/41/approve']);
  assert.match(await page.locator('.review-result').innerText(),/1 approved/);
  assert.match(await page.locator('.review-result').innerText(),/1 not attempted/);
  assert.equal(await page.getByRole('button',{name:/Approve all matching/}).isDisabled(),true);
});
test('bulk permission denial reports acknowledged, denied and unattempted decisions separately',async t=>{
  const {page,decisions}=await fixture(t,{review:true,backlog:true});
  await page.route('**/v1/facts/42/approve',route=>route.fulfill({status:403,json:{error:'review permission revoked'}}));
  await page.getByRole('button',{name:'Approve all matching (3)',exact:true}).click();
  await page.getByRole('button',{name:'Confirm approval',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'review access'}).waitFor();
  assert.deepEqual(decisions.map(d=>d.path),['/v1/facts/41/approve']);
  const receipt=await page.locator('.review-result').innerText();
  assert.match(receipt,/1 approved/);assert.match(receipt,/1 denied/);assert.match(receipt,/0 unconfirmed/);assert.match(receipt,/1 not attempted/);
  assert.equal(await page.locator('.proposal').count(),2);
  assert.equal(await page.locator('.server-state').innerText(),'Authenticated');
});
test('review source failures offer retry without claiming evidence was deleted',async t=>{
  const {page}=await fixture(t,{review:true});
  await page.route('**/v1/episodes/7',route=>route.fulfill({status:503,contentType:'application/json',body:'{"error":"Source temporarily offline"}'}));
  await page.getByText('Read full source',{exact:false}).click();
  await page.getByRole('alert').filter({hasText:'Source temporarily offline'}).waitFor();
  assert.doesNotMatch(await page.locator('.excerpt').innerText(),/no longer available/);
  await page.unroute('**/v1/episodes/7');
  await page.getByRole('button',{name:'Retry source'}).click();
  await page.locator('.source-markdown').getByText(/END OF SOURCE/).waitFor();
});
test('review duplicate and historical receipts stay attached to the submitted proposal',async t=>{
  for(const [reply,expected] of [[{fact_id:12,status:'active'},/Proposal #41 resolved as duplicate of #12/],[{fact_id:41,status:'closed'},/Proposal #41 approved as historical memory/]]){
    const {page}=await fixture(t,{review:true});
    await page.route('**/v1/facts/41/approve',async route=>{await route.fetch();await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(reply)});});
    await page.getByRole('button',{name:'Approve',exact:true}).click();
    await page.getByText('Nothing awaits review.',{exact:true}).waitFor();
    assert.match(await page.locator('.review-notice').innerText(),expected);
  }
});
test('review inbox and bulk dialog fit mobile and cancellation makes no decision',async t=>{
  const {page,decisions}=await fixture(t,{review:true,backlog:true,mobile:true});
  await page.locator('.proposal').first().waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'review-inbox-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Approve all matching (3)',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Approve 3 proposals?'});await dialog.waitFor();
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'review-bulk-mobile.png'),fullPage:true});
  await page.keyboard.press('Escape');assert.equal(await dialog.count(),0);
  assert.equal(decisions.length,0);
});
test('decline has an inline reason, safe cancellation and full source on mobile',async t=>{
  const {page,decisions}=await fixture(t,{review:true,mobile:true});
  await page.getByRole('button',{name:'Decline',exact:true}).click();
  const reason=page.getByRole('textbox',{name:'Reason for declining'});
  await reason.waitFor({timeout:2000});
  assert.equal(decisions.length,0);
  assert.equal(await page.getByRole('button',{name:'Confirm decline'}).isDisabled(),true);
  await page.getByRole('button',{name:'Cancel',exact:true}).click();
  assert.equal(await reason.count(),0);assert.equal(decisions.length,0);
  await page.getByText('Read full source',{exact:false}).click();
  await page.locator('.source-markdown').getByText(/END OF SOURCE/).waitFor();
  assert.match(await page.locator('.excerpt').innerText(),/END OF SOURCE/);
  await page.getByRole('button',{name:'Decline',exact:true}).click();
  await reason.fill('This is a hypothetical, not an observation.');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'review-decline-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Confirm decline'}).click();
  await page.getByText('Nothing awaits review.',{exact:true}).waitFor();
  assert.deepEqual(decisions,[{path:'/v1/facts/41/decline',body:{reason:'This is a hypothetical, not an observation.'}}]);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});
test('review prevents repeated decisions while the server is saving',async t=>{
  const {page,decisions}=await fixture(t,{review:true});
  let release;const hold=new Promise(resolve=>release=resolve);
  t.after(()=>release());
  await page.route('**/v1/facts/41/approve',async route=>{await hold;await route.continue();});
  await page.getByRole('button',{name:'Approve',exact:true}).click();
  const saving=page.getByRole('button',{name:'Saving…',exact:true});await saving.waitFor();
  assert.equal(await saving.isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Decline',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Refresh queue'}).isDisabled(),true);
  assert.equal(await page.locator('.proposal').count(),1);assert.equal(decisions.length,0);
  release();
  await page.getByText('Nothing awaits review.',{exact:true}).waitFor();
  assert.equal(decisions.length,1);
});
test('missing key identifies the local server and does not pretend to be connecting',async t=>{
  const {page,calls}=await fixture(t,{noKey:true,memory:true});
  const dialog=page.getByRole('dialog',{name:'Memory connection'});
  await dialog.waitFor({timeout:3000});
  assert.match(await dialog.innerText(),/Scone space key/);
  assert.match(await dialog.innerText(),/not a Claude, Codex or OpenAI API key/);
  assert.match(await dialog.innerText(),/127\.0\.0\.1:/);
  await page.getByRole('button',{name:'Close dialog'}).click();
  assert.equal(await page.locator('#space').innerText(),'Not connected');
  assert.equal(calls(),0);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});
test('stored edges lead to a safe source inspector and real recall',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('button',{name:'Inspect prompt',exact:true}).click({timeout:3000});
  assert.match(await page.locator('#inspector').innerText(),/Keep launch local <script>bad\(\)<\/script>/);
  assert.equal(await page.locator('[data-edge]').count(),4);
  await page.getByRole('textbox',{name:'Test memory recall'}).fill('launch');
  await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.locator('#recall-results').getByText('Keep launch local',{exact:false}).first().waitFor();
  assert.match(await page.locator('#recall-results').innerText(),/Keep launch local/);
  assert.doesNotMatch(await page.locator('#connection').innerText(),/agent connected/i);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'playground-desktop.png')});
});
// An outward focus ring inside the scrolling rail is clipped at both sides.
test('session selection keeps keyboard focus visible inside the scrolling rail',async t=>{
  const {page}=await fixture(t);
  const row=page.locator('#sessions .session').last();
  await row.waitFor();
  await page.keyboard.press('Tab');
  await row.focus();
  const focus=await row.evaluate(el=>{
    const css=getComputedStyle(el), rect=el.getBoundingClientRect();
    const list=el.parentElement.getBoundingClientRect();
    const reach=Math.max(0,parseFloat(css.outlineOffset)+parseFloat(css.outlineWidth));
    return {visible:el.matches(':focus-visible'),width:parseFloat(css.outlineWidth),
      contained:rect.left-reach>=list.left && rect.right+reach<=list.right,
      height:rect.height};
  });
  assert.equal(focus.visible,true);
  assert.ok(focus.width>=2,'keyboard focus needs a visible indicator');
  assert.equal(focus.contained,true,'focus ring must not be cut by the session scroller');
  assert.ok(focus.height>=44,'session remains an accessible pointer target');
  await page.keyboard.press('Enter');
  assert.equal(await row.getAttribute('aria-pressed'),'true');
  assert.equal(await page.locator('#sessions .session').first().getAttribute('aria-pressed'),'false');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.locator('.rail').screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'session-rail-focused.png')});
});
test('pause stops refresh and expiry clears private evidence',async t=>{
  const f=await fixture(t);const {page}=f;
  await page.getByRole('button',{name:'Pause live updates'}).click({timeout:3000});
  const calls=f.calls();await page.waitForTimeout(1300);assert.equal(f.calls(),calls);
  f.expire();await page.getByRole('button',{name:'Resume live updates'}).click();
  await page.getByText('Connection needs authentication',{exact:true}).waitFor();
  assert.equal(await page.locator('[data-node]').count(),0);
  assert.doesNotMatch((await page.locator('#inspector').allTextContents()).join(''),/Keep launch/);
  assert.equal(await page.locator('[data-record]').count(),0);
});
test('empty mobile workspace is honest, fits and remains usable',async t=>{
  const {page}=await fixture(t,{empty:true,mobile:true});
  await page.getByText('No recorded interactions yet',{exact:true}).waitFor({timeout:3000});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.equal(await page.locator('[data-node]').count(),0);
  assert.match(await page.locator('#connection').innerText(),/No agent events/);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'playground-mobile.png'),fullPage:true});
});
// Removing the setup surface or deriving connection from API health must fail.
test('agent setup distinguishes received activity from a verified live connection',async t=>{
  const {page}=await fixture(t,{empty:true});
  await page.getByRole('button',{name:'Connect agents',exact:true}).click({timeout:3000});
  const dialog=page.getByRole('dialog',{name:'Agent connections'});
  await dialog.waitFor();
  assert.match(await dialog.innerText(),/No activity received/);
  assert.match(await dialog.innerText(),/Codex App.*unverified/s);
  assert.doesNotMatch(await dialog.innerText(),/Connected successfully/);
  await page.getByRole('button',{name:'Close connections'}).click();
  assert.equal(await dialog.isVisible(),false);
});
// Spatial projection is a view change, never a relationship generator.
test('a claim with an omitted source explains its missing session path without inventing an edge',async t=>{
  const {page}=await fixture(t);
  await page.route('**/v1/graph*',route=>route.fulfill({json:{nodes:[...nodes,{id:'claim:88',kind:'claim',label:'A sourced claim',data:{source_episode_id:777}}],edges,truncated:true,provenance_omitted:3,coverage:{agent:'connector-reported'}}}));
  await page.reload();
  await page.locator('[data-node="claim:88"]').click();
  const inspector=page.getByRole('complementary',{name:'Source inspector'});
  await inspector.getByText('Episode #777 is referenced by this claim but is not included in this snapshot.',{exact:true}).waitFor({timeout:2000});
  assert.match(await inspector.innerText(),/No recorded session path is available in this snapshot/);
  assert.equal(await page.locator('[data-edge]').count(),4);
  assert.equal(await inspector.locator('.relation').count(),0);
  assert.match(await page.locator('#coverage').innerText(),/3 source episodes outside snapshot/);
  assert.match(await page.locator('#coverage').innerText(),/Partial snapshot/);
});

test('depth clears stale highlighting when polling removes a selected record',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('button',{name:'Depth view'}).click();
  await page.locator('[data-node="turn:1"]').focus();await page.keyboard.press('Enter');
  await page.locator('#inspector').waitFor();
  await page.route('**/v1/graph*',route=>route.fulfill({json:{nodes:nodes.filter(n=>n.id!=='turn:1'),edges:edges.filter(e=>e.source!=='turn:1'&&e.target!=='turn:1'),truncated:true,coverage:{agent:'connector-reported'}}}));
  await page.waitForFunction(()=>!document.querySelector('[data-node="turn:1"]'));
  assert.equal(await page.locator('.depth-node.is-dimmed').count(),0,'An expired selection must not dim the current snapshot.');
});

test('depth exposes directed relationship labels and only captures intentional zoom gestures',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('button',{name:'Depth view'}).click();
  await page.locator('[data-node="turn:1"]').focus();
  assert.deepEqual(await page.locator('.depth-edge-caption').allTextContents(),['has','captured as']);
  const activeEdges=await page.locator('[data-edge].highlight').evaluateAll(items=>items.map(el=>el.getAttribute('marker-end')));
  assert.ok(activeEdges.every(marker=>marker&&marker.startsWith('url(')),'Stored direction remains visible.');
  for(const ctrlKey of [false,true]){
    const canceled=await page.locator('#graph').evaluate((el,ctrl)=>{const event=new WheelEvent('wheel',{deltaY:40,ctrlKey:ctrl,bubbles:true,cancelable:true});el.dispatchEvent(event);return event.defaultPrevented;},ctrlKey);
    assert.equal(canceled,ctrlKey,'Ordinary page scrolling must not be trapped.');
  }
});

test('depth orbits individual records in 3D without changing stored relationships',async t=>{
  const {page}=await fixture(t,{crowded:true});
  await page.getByRole('button',{name:'Depth view'}).click();
  await page.waitForFunction(()=>document.querySelectorAll('#graph [data-node]').length===241,{},{timeout:3000});
  const coordinates=await page.locator('#graph [data-node]').evaluateAll(items=>items.map(el=>({z:Number(el.getAttribute('data-z')),transform:el.getAttribute('transform')})));
  assert.ok(new Set(coordinates.map(p=>p.z)).size>10,'The records must occupy real Z coordinates.');
  assert.equal(await page.locator('#graph [data-edge]').count(),240);
  const graph=page.locator('#graph');
  await graph.focus();
  await page.keyboard.press('ArrowRight');
  const rotated=await page.locator('#graph [data-node]').evaluateAll(items=>items.map(el=>el.getAttribute('transform')));
  assert.notDeepEqual(rotated,coordinates.map(p=>p.transform),'Orbit must project different node positions.');
  assert.equal(await page.locator('#graph [data-edge]').count(),240);
  await page.getByRole('button',{name:'Reset 3D camera'}).click();
  await page.locator('[data-node="tool:0"]').focus();
  await page.keyboard.press('Enter');
  await page.locator('.inspector').waitFor();
  assert.match(await page.locator('.inspector').innerText(),/tool:0/);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'depth-records.png'),fullPage:true});
});

test('mobile depth controls fit, and pointer orbit changes the projected scene',async t=>{
  const {page}=await fixture(t,{mobile:true});
  await page.getByRole('button',{name:'Depth view'}).click();
  const map=page.locator('#graph'),before=await page.locator('[data-node]').evaluateAll(els=>els.map(el=>el.getAttribute('transform')));
  await map.scrollIntoViewIfNeeded();
  const box=await map.boundingBox();
  await page.mouse.move(box.x+30,box.y+box.height/2);
  await page.mouse.down();await page.mouse.move(box.x+95,box.y+box.height/2+30,{steps:8});await page.mouse.up();
  await page.waitForFunction(old=>JSON.stringify([...document.querySelectorAll('[data-node]')].map(el=>el.getAttribute('transform')))!==JSON.stringify(old),before);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  const controls=await page.locator('.zoom').boundingBox();
  assert.ok(controls.x>=box.x&&controls.x+controls.width<=box.x+box.width+1);
  const help=await page.locator('.depth-help').boundingBox(),legend=await page.locator('.atlas-legend').boundingBox();
  assert.ok(help.y+help.height<=legend.y,'Help must not overlap the legend.');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'depth-mobile.png'),fullPage:true});
});

test('depth pages very large ledgers without silently dropping records',async t=>{
  const {page}=await fixture(t);
  const large=Array.from({length:1250},(_,i)=>({id:'claim:'+String(i).padStart(4,'0'),kind:'claim',label:'Claim '+i}));
  await page.route('**/v1/graph*',route=>route.fulfill({json:{nodes:large,edges:[],truncated:false}}));
  await page.reload();await page.getByRole('button',{name:'Depth view'}).click();
  await page.waitForFunction(()=>document.querySelectorAll('#graph [data-node]').length>0);
  assert.equal(await page.locator('#graph [data-node]').count(),1000);
  await page.getByRole('button',{name:'Next 3D page'}).click();
  assert.equal(await page.locator('#graph [data-node]').count(),250);
  assert.equal(await page.getByRole('button',{name:'Next 3D page'}).isEnabled(),false);
  await page.locator('[data-node="claim:1249"]').focus();await page.keyboard.press('Enter');
  assert.match(await page.locator('#inspector').innerText(),/claim:1249/);
});

test('source inspector spells out incoming and outgoing evidence direction',async t=>{
  const {page}=await fixture(t);
  await page.route('**/v1/graph*',route=>route.fulfill({json:{nodes:[...nodes,{id:'claim:88',kind:'claim',label:'Sourced claim',data:{source_episode_id:7}}],edges:[...edges,{source:'episode:7',target:'claim:88',kind:'source_of'}],truncated:false}}));
  await page.reload();await page.locator('[data-node="claim:88"]').click();
  const relation=page.locator('#inspector .relation');
  assert.match(await relation.innerText(),/episode:7 → source of → claim:88/);
  await relation.click();
  assert.match(await page.locator('#inspector').innerText(),/This record → claim:88/);
});

test('depth view preserves exact recorded relationships and inspector access',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('button',{name:'Depth view'}).click({timeout:3000});
  const pairs=await page.locator('[data-edge]').evaluateAll(es=>es.map(e=>[e.dataset.source,e.dataset.target]));
  assert.deepEqual(pairs,[['session:codex:s1','turn:1'],['turn:1','episode:7'],['episode:7','chunk:9'],['recall:12','chunk:9']]);
  await page.getByRole('button',{name:'Inspect prompt',exact:true}).click();
  assert.match(await page.locator('#inspector').innerText(),/Keep launch local/);
  await page.getByRole('button',{name:'2D view'}).click();
  assert.equal(await page.locator('[data-edge]').count(),4);
});
test('record search finds captured content without manufacturing graph links',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('button',{name:'Records view'}).click({timeout:3000});
  await page.getByRole('searchbox',{name:'Find a record'}).fill('bad()');
  assert.equal(await page.locator('[data-record]').count(),1);
  await page.locator('[data-record]').click();
  assert.match(await page.locator('#inspector').innerText(),/Keep launch local <script>bad\(\)<\/script>/);
  await page.getByRole('searchbox',{name:'Find a record'}).fill('nothing matches this');
  await page.getByText('No matching records',{exact:true}).waitFor();
  assert.equal(await page.locator('[data-record]').count(),0);
  await page.getByRole('searchbox',{name:'Find a record'}).fill('');
  assert.equal(await page.locator('[data-record]').count(),5);
  await page.getByRole('button',{name:'Graph view'}).click();
  assert.equal(await page.locator('[data-edge]').count(),4);
});
test('development preview reloads after a served asset revision changes',async t=>{
  const f=await fixture(t,{dev:true});
  await f.page.getByText('UI auto-refresh on',{exact:true}).waitFor({timeout:3000});
  f.revise();
  await f.page.waitForFunction(()=>performance.getEntriesByType('navigation')[0]?.type==='reload',{},{timeout:5000});
  assert.equal(f.loads(),2);
});

test('editing a recall query cannot display the previous query results',async t=>{
  const {page}=await fixture(t);
  const input=page.getByRole('textbox',{name:'Test memory recall'});
  await input.fill('launch');
  await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.locator('#recall-results').getByText('Keep launch local',{exact:false}).first().waitFor();
  assert.match(await page.locator('#recall-results').innerText(),/launch/);
  await input.fill('Cats');
  assert.doesNotMatch(await page.locator('#recall-results').innerText(),/Keep launch local/);
});

test('crowded graph expands paged groups without tiny unbounded stacks',async t=>{
  const {page}=await fixture(t,{crowded:true});
  await chooseLayout(page,'flow');
  await page.locator('[data-group]').first().waitFor();
  assert.equal(await page.locator('[data-group]').count(),1);
  assert.equal(await page.locator('[data-node]').count(),1);
  await page.locator('[data-group]').click();
  assert.equal(await page.locator('[data-node]').count(),9);
  const first=await page.locator('[data-node]').evaluateAll(es=>es.map(e=>e.dataset.node));
  await page.getByRole('button',{name:'Next group page'}).click();
  const second=await page.locator('[data-node]').evaluateAll(es=>es.map(e=>e.dataset.node));
  assert.notDeepEqual(first,second);
  const boxes=await page.locator('[data-node] rect').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};}));
  for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++)assert.ok(boxes[i].x+boxes[i].w<=boxes[j].x||boxes[j].x+boxes[j].w<=boxes[i].x||boxes[i].y+boxes[i].h<=boxes[j].y||boxes[j].y+boxes[j].h<=boxes[i].y);
  assert.ok(boxes.every(b=>b.w>100));
  await page.getByRole('button',{name:'Depth view'}).click();
  assert.equal(await page.locator('[data-node]').count(),241,'3D explores individual snapshot records, not 2D overview groups.');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'playground-grouped.png'),fullPage:true});
});

test('graph overview uses the workspace until an inspector is needed and can return to overview',async t=>{
  const {page}=await fixture(t,{crowded:true});
  await chooseLayout(page,'flow');
  await page.locator('[data-group]').waitFor();
  assert.equal(await page.locator('#inspector').isVisible(),false,'empty inspector must not consume map space');
  const group=page.locator('[data-group]');
  assert.ok((await group.boundingBox()).width>=220,'overview group should be legible without manual zoom');
  await page.locator('[data-node]').click();
  await page.locator('#inspector').waitFor();
  await page.getByRole('button',{name:'Close source inspector'}).click();
  assert.equal(await page.locator('#inspector').isVisible(),false);
  await group.focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('[data-node]').count(),9);
  await page.getByRole('button',{name:'← Overview',exact:true}).click();
  await page.locator('[data-group]').waitFor();
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'graph-atlas-desktop.png'),fullPage:true});
});

test('layout picker accepts a real pointer click in 2D and Depth',async t=>{
  const {page}=await fixture(t);
  await page.locator('[data-node]').first().waitFor();
  for(const depth of [false,true]) {
    if(depth)await page.getByRole('button',{name:'Depth view'}).click();
    const picker=page.getByRole('button',{name:'Graph layout',exact:true});
    assert.equal(await picker.evaluate(el=>{const b=el.getBoundingClientRect();return el.contains(document.elementFromPoint(b.x+b.width/2,b.y+b.height/2));}),true,'picker must receive pointer events');
    await page.evaluate(()=>{window.canvasPointerDowns=0;document.querySelector('#graph').addEventListener('pointerdown',()=>window.canvasPointerDowns++);});
    await picker.click({timeout:1500});
    await page.getByRole('menu',{name:'Graph layouts'}).waitFor();
    await page.keyboard.press('ArrowDown');await page.keyboard.press('Enter');
    assert.equal(await page.locator('#graph').getAttribute('data-layout'),'radial');
    await picker.click();
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(()=>window.canvasPointerDowns),0);
    assert.equal(await picker.evaluate(el=>document.activeElement===el),true);
    await chooseLayout(page,'constellation');
  }
});

test('session switches retain the chosen layout and fit before visible frames',async t=>{
  const {page,grow}=await fixture(t);grow();
  await page.locator('[data-group]').first().waitFor();
  await chooseLayout(page,'radial');
  await page.evaluate(()=>{
    window.graphFrames=[];window.observeGraphFrames=true;
    const frame=()=>{if(!window.observeGraphFrames)return;const svg=document.querySelector('#graph'),r=svg?.getBoundingClientRect();
      if(r)window.graphFrames.push({layout:svg.dataset.layout,fits:[...svg.querySelectorAll('.orb-surface')].every(el=>{const b=el.getBoundingClientRect();return b.x>=r.x&&b.y>=r.y&&b.right<=r.right+1&&b.bottom<=r.bottom+1;})});
      requestAnimationFrame(frame);};requestAnimationFrame(frame);
  });
  await page.locator('#sessions button').nth(2).click();
  await page.locator('#sessions button').first().click();
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const frames=await page.evaluate(()=>{window.observeGraphFrames=false;return window.graphFrames;});
  assert.ok(frames.length>0);assert.ok(frames.every(f=>f.layout==='radial'&&f.fits),'switch painted a reset layout or unfitted graph');
  await page.getByRole('button',{name:'Graph layout',exact:true}).click();
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'graph-layout-menu.png'),fullPage:true});
});

test('growth spiral reveals individual events, pages honestly and opens their real source',async t=>{
  const {page}=await fixture(t,{crowded:true});
  await chooseLayout(page,'growth');
  assert.equal(await page.locator('[data-group]').count(),0);
  assert.equal(await page.locator('.growth-event').count(),120);
  assert.match(await page.locator('.graph-navigation').innerText(),/240 dated agent events/);
  const first=await page.locator('[data-node]').evaluateAll(es=>es.map(e=>e.dataset.node));
  await page.getByRole('button',{name:'Next graph page'}).click();
  assert.equal(await page.locator('.growth-event').count(),120);
  const second=await page.locator('[data-node]').evaluateAll(es=>es.map(e=>e.dataset.node));
  assert.equal(first.filter(id=>id.startsWith('tool:')).some(id=>second.includes(id)),false);
  assert.equal(await page.locator('.lifecycle-boundary').count(),0,'must not invent starts or ends');
  const event=page.locator('[data-node^="tool:"]').first();const id=await event.getAttribute('data-node');
  await event.focus();await page.keyboard.press('Enter');
  assert.match(await page.locator('#inspector').innerText(),new RegExp(id));
  await page.getByRole('button',{name:'Close source inspector'}).click();
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'graph-growth-desktop.png'),fullPage:true});
});

function memoryRecall(text,episode=7) {
  return {items:[{episode_id:episode,chunk_id:episode,text,score:1,created_at:'2026-09-06T03:00:00Z',source:null,tags:[],metadata:{}}],facts:[],degraded:[],event_id:12,returned_bytes:text.length,space_bytes:1000};
}
test('source composer requires advertised support and cancellation sends no writes',async t=>{
  for(const rustCapabilities of [false,true]){
    const {page,requested}=await fixture(t,{rustCapabilities});
    await page.route('**/v1/recall?**',route=>route.fulfill({json:memoryRecall('Existing note')}));
    await page.goto(new URL('/memory?q=fixture#search',page.url()).href);
    await page.locator('.rows .row').waitFor();
    if(rustCapabilities){assert.equal(await page.getByRole('button',{name:'Add source',exact:true}).count(),0);continue;}
    await page.getByRole('button',{name:'Add source',exact:true}).click();
    await page.getByRole('textbox',{name:'Source note',exact:true}).fill('An unsaved draft');
    await page.locator('.source-composer').getByRole('button',{name:'Close',exact:true}).click();
    assert.equal(requested.includes('/v1/episodes'),false);assert.equal(requested.includes('/v1/attachments'),false);
  }
});
test('a denied source write retains the draft without claiming an uncertain save',async t=>{
  const {page}=await fixture(t,{memory:true});
  await page.route('**/v1/episodes',route=>route.fulfill({status:403,json:{error:'key role read cannot write'}}));
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  const input=page.getByRole('textbox',{name:'Source note',exact:true});await input.fill('Retain this unsaved draft');
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByRole('alert').filter({hasText:/write access/}).waitFor();
  assert.equal(await input.inputValue(),'Retain this unsaved draft');
  assert.equal(await page.getByText(/The save is unconfirmed/).count(),0);
  assert.equal(await page.locator('.server-state').innerText(),'Authenticated');
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isEnabled(),true);
});

test('a forbidden verification read cannot erase an acknowledged source write',async t=>{
  const {page}=await fixture(t,{memory:true});
  await page.route('**/v1/episodes',route=>route.fulfill({json:{episode_id:9,deduplicated:false}}));
  await page.route('**/v1/episodes/9',route=>route.fulfill({status:403,json:{error:'source read forbidden'}}));
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('An acknowledged write');
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText(/The save is unconfirmed/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),true);
  assert.equal(await page.locator('.server-state').innerText(),'Authenticated');
});

test('a denied source link preserves the acknowledged upload and prevents repeating the write',async t=>{
  const {page}=await fixture(t,{memory:true});let uploads=0;
  const bytes=Buffer.from('an opaque raster fixture'),id=createHash('sha256').update(bytes).digest('hex');
  await page.route('**/v1/attachments',route=>{uploads++;return route.fulfill({json:{attachment_id:id,media_type:'image/png',bytes:bytes.length}});});
  await page.route('**/v1/episodes',route=>route.fulfill({status:403,json:{error:'key role changed after upload'}}));
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('An image with a refused link');
  await page.getByLabel('Original image',{exact:true}).setInputFiles({name:'source.png',mimeType:'image/png',buffer:bytes});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText(/The save is unconfirmed/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),true);
  await page.locator('.source-composer form').evaluate(form=>form.requestSubmit());
  assert.equal(uploads,1);
  assert.equal(await page.locator('.server-state').innerText(),'Authenticated');
});

test('source composer blocks repeated writes and reports an unconfirmed save honestly',async t=>{
  const {page}=await fixture(t);let calls=0, release;
  const pending=new Promise(resolve=>{release=resolve;});
  await page.route('**/v1/recall?**',route=>route.fulfill({json:memoryRecall('Existing note')}));
  await page.route('**/v1/episodes',route=>{calls++;release(route);});
  await page.goto(new URL('/memory?q=fixture#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('Keep an original note');
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  const route=await pending;
  await page.locator('.source-composer form').evaluate(form=>form.requestSubmit());
  assert.equal(calls,1);assert.equal(await page.locator('.source-composer').getByRole('button',{name:'Close',exact:true}).isDisabled(),true);
  await route.fulfill({status:503,json:{error:'Write response unavailable'}});
  await page.getByText(/The save is unconfirmed/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),true);
  assert.equal(await page.getByText(/Source saved · episode/).count(),0);
  await page.locator('.source-composer form').evaluate(form=>form.requestSubmit());assert.equal(calls,1);
});
test('source composer refuses unsupported files before upload',async t=>{
  const {page,requested}=await fixture(t,{mobile:true});
  await page.route('**/v1/recall?**',route=>route.fulfill({json:memoryRecall('Existing note')}));
  await page.goto(new URL('/memory?q=fixture#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('A source note');
  await page.locator('.source-composer input[type=file]').setInputFiles({name:'unsafe.svg',mimeType:'image/svg+xml',buffer:Buffer.from('<svg/>')});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText('Choose a nonempty PNG, JPEG, GIF or WebP image up to 25 MB.',{exact:true}).waitFor();
  assert.equal(requested.includes('/v1/episodes'),false);assert.equal(requested.includes('/v1/attachments'),false);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'source-composer-mobile.png'),fullPage:true});
});
test('source composer does not claim success when the image link is missing',async t=>{
  const {page}=await fixture(t);
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const id=createHash('sha256').update(png).digest('hex');
  await page.route('**/v1/recall?**',route=>route.fulfill({json:memoryRecall('Existing note')}));
  await page.route('**/v1/attachments',route=>route.fulfill({json:{attachment_id:id,media_type:'image/png',bytes:png.length,filename:'source.png'}}));
  await page.route('**/v1/episodes',route=>route.fulfill({json:{episode_id:9,deduplicated:false,chunks:1}}));
  await page.route('**/v1/episodes/9',route=>route.fulfill({json:{episode_id:9,content:'Image evidence',attachments:[]}}));
  await page.goto(new URL('/memory?q=fixture#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('Image evidence');
  await page.getByLabel('Original image',{exact:true}).setInputFiles({name:'source.png',mimeType:'image/png',buffer:png});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText('The saved attachment link could not be verified',{exact:true}).waitFor();
  assert.equal(await page.getByText(/Source saved · episode/).count(),0);
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),true);
});
test('text import previews literal code and sends exact source only after saving',async t=>{
  const {page}=await fixture(t,{mobile:true});const writes=[];
  const content='# Juniper\r\n<script>window.importExecuted=true</script>\r\nCalibration: Polaris.\n';
  await page.route('**/v1/recall?**',route=>route.fulfill({json:memoryRecall('Existing note')}));
  await page.route('**/v1/episodes',route=>{writes.push(route.request().postDataJSON());return route.fulfill({json:{episode_id:9,deduplicated:false}});});
  await page.route('**/v1/episodes/9',route=>route.fulfill({json:{episode_id:9,content,kind:'file',source:'calibration.md',attachments:[]}}));
  await page.goto(new URL('/memory?q=fixture#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  await page.getByLabel('Text file', {exact:true}).setInputFiles({name:'calibration.md',mimeType:'text/markdown',buffer:Buffer.from(content)});
  await page.getByRole('region',{name:'File preview'}).waitFor();
  assert.equal(writes.length,0);
  assert.equal(await page.evaluate(()=>window.importExecuted),undefined);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'text-import-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText('Source saved · episode #9',{exact:true}).waitFor();
  assert.deepEqual(writes,[{content,kind:'file',source:'calibration.md',attachment_ids:[]}]);
  await page.getByText('Read saved text',{exact:true}).click();
  assert.equal(await page.locator('.source-saved pre').textContent(),content);
  assert.equal(await page.evaluate(()=>window.importExecuted),undefined);
});
test('invalid text files cannot be saved and a new selection can recover',async t=>{
  const {page,requested}=await fixture(t);
  await page.goto(new URL('/memory#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'broken.txt',mimeType:'text/plain',buffer:Buffer.from([0xc3,0x28])});
  await page.getByRole('alert').filter({hasText:'UTF-8'}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),true);
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'valid.txt',mimeType:'text/plain',buffer:Buffer.from('Valid source')});
  await page.getByRole('region',{name:'File preview'}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),false);
  await page.locator('.source-composer').getByRole('button',{name:'Close',exact:true}).click();
  assert.equal(requested.includes('/v1/episodes'),false);
});
test('an exact duplicate keeps the earlier recorded filename visible',async t=>{
  const {page}=await fixture(t);const content='A retained source';
  await page.route('**/v1/episodes',route=>route.fulfill({json:{episode_id:9,deduplicated:true}}));
  await page.route('**/v1/episodes/9',route=>route.fulfill({json:{episode_id:9,content,kind:'file',source:'original.md',attachments:[]}}));
  await page.goto(new URL('/memory#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'renamed.md',mimeType:'text/markdown',buffer:Buffer.from(content)});
  await page.getByRole('region',{name:'File preview'}).waitFor();
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText(/That episode was reused; its original source metadata is unchanged/).waitFor();
  await page.getByText('Read saved text',{exact:true}).click();
  await page.getByText('Verified at save · recorded source: original.md',{exact:true}).waitFor();
  assert.equal(await page.locator('.source-saved').getByText('renamed.md',{exact:true}).count(),0);
});
test('text imports do not call changed or deduplicated text an exact saved original',async t=>{
  const {page}=await fixture(t);let writes=0;
  await page.route('**/v1/episodes',route=>{writes++;return route.fulfill({json:{episode_id:9,deduplicated:true}});});
  await page.route('**/v1/episodes/9',route=>route.fulfill({json:{episode_id:9,content:'Earlier trimmed content',kind:'note',attachments:[]}}));
  await page.goto(new URL('/memory#search',page.url()).href);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'source.txt',mimeType:'text/plain',buffer:Buffer.from(' Earlier trimmed content\n')});
  await page.getByRole('region',{name:'File preview'}).waitFor();
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText('The retained text differs from the selected file. Inspect memory before retrying.',{exact:true}).waitFor();
  assert.equal(await page.getByText(/Source saved · episode/).count(),0);
  assert.equal(await page.getByRole('button',{name:'Save source',exact:true}).isDisabled(),true);
  await page.locator('.source-composer form').evaluate(form=>form.requestSubmit());assert.equal(writes,1);
});
test('Memory Search removes old passages and image URLs while a new query is pending',async t=>{
  const {page}=await fixture(t);
  let held;const waiting=new Promise(resolve=>{held=resolve;});
  await page.route('**/v1/recall?**',route=>{
    const q=new URL(route.request().url()).searchParams.get('q');
    if(q==='slow'){held(route);return;}
    return route.fulfill({json:memoryRecall(q==='new'?'New evidence':'Original evidence')});
  });
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const id=createHash('sha256').update(png).digest('hex');
  await page.route('**/v1/episodes/7',route=>route.fulfill({json:{episode_id:7,attachments:[{attachment_id:id,media_type:'image/png',bytes:png.length,filename:'old-source.png'}]}}));
  await page.route('**/v1/attachments/*',route=>route.fulfill({contentType:'image/png',body:png}));
  await page.goto(new URL('/memory?q=original#search',page.url()).href);
  await page.getByRole('button',{name:'View source images'}).click();
  const img=page.getByRole('img',{name:'old-source.png'});await img.waitFor();
  const url=await img.getAttribute('src');
  const query=page.locator('.search input');
  await query.fill('slow');await page.locator('.search button').click();
  const delayed=await waiting;
  assert.equal(await page.locator('.rows .row').count(),0,'previous evidence must disappear before the new response');
  assert.equal(await page.evaluate(async url=>{try{await fetch(url);return true;}catch{return false;}},url),false,'old image URL must be revoked');
  await query.fill('new');await page.locator('.search button').click();
  await page.getByText('New evidence',{exact:false}).waitFor();
  await delayed.fulfill({json:memoryRecall('Obsolete late evidence')}).catch(()=>{});
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  assert.equal(await page.getByText('Obsolete late evidence',{exact:false}).count(),0);
  assert.match(await page.locator('.rows').innerText(),/New evidence/);
});
test('Memory Search retries the current failed query and clears errors when the query changes',async t=>{
  const {page}=await fixture(t);let attempts=0;
  await page.route('**/v1/recall?**',route=>{
    const q=new URL(route.request().url()).searchParams.get('q');
    if(q==='broken'&&++attempts===1)return route.fulfill({status:503,json:{error:'Search temporarily unavailable'}});
    if(q==='broken-again')return route.fulfill({status:503,json:{error:'Another query failed'}});
    return route.fulfill({json:memoryRecall(q==='broken'?'Recovered evidence':'Different evidence')});
  });
  await page.goto(new URL('/memory?q=broken#search',page.url()).href);
  await page.getByRole('button',{name:'Retry search',exact:true}).click({timeout:2000});
  await page.getByText('Recovered evidence',{exact:false}).waitFor();
  assert.equal(attempts,2);
  assert.equal(await page.getByText('Search temporarily unavailable',{exact:true}).count(),0);
  await page.locator('.search input').fill('broken-again');await page.locator('.search button').click();
  await page.getByText('Another query failed',{exact:true}).waitFor();
  await page.locator('.search input').fill('different');await page.locator('.search button').click();
  await page.getByText('Different evidence',{exact:false}).waitFor();
  assert.equal(await page.getByText('Another query failed',{exact:true}).count(),0);
  assert.equal(await page.getByText('Recovered evidence',{exact:false}).count(),0);
});
test('recall shows authenticated source images and revokes previews when the query changes',async t=>{
  const {page}=await fixture(t);
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const id=createHash('sha256').update(png).digest('hex');let downloads=0;
  await page.route('**/v1/episodes/7',route=>route.fulfill({json:{episode_id:7,attachments:[{attachment_id:id,media_type:'image/png',bytes:png.length,filename:'source.png'}]}}));
  await page.route('**/v1/attachments/*',route=>{downloads++;assert.equal(route.request().headers().authorization,'Bearer fixture-key');return route.fulfill({contentType:'image/png',body:png});});
  const query=page.getByRole('textbox',{name:'Test memory recall'});await query.fill('screenshot');await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.getByRole('button',{name:'View source images',exact:true}).click({timeout:2000});
  const img=page.getByRole('img',{name:'source.png',exact:true});await img.waitFor();
  await page.waitForFunction(()=>document.querySelector('.source-image img')?.naturalWidth>0);
  const url=await img.getAttribute('src');assert.match(url,/^blob:/);assert.equal(downloads,1);
  assert.equal(url.includes('fixture-key'),false);
  await query.fill('another question');assert.equal(await img.count(),0);
  assert.equal(await page.evaluate(async url=>{try{await fetch(url);return true;}catch{return false;}},url),false,'private object URL must be revoked');
});
test('source image absence is different from a retryable source lookup failure',async t=>{
  const {page}=await fixture(t);let missing=false;
  await page.route('**/v1/episodes/7',route=>missing?route.fulfill({json:{episode_id:7,attachments:[]}}):route.fulfill({status:503,json:{error:'temporary failure'}}));
  await page.getByRole('textbox',{name:'Test memory recall'}).fill('image');await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.getByRole('button',{name:'View source images',exact:true}).click({timeout:2000});
  await page.getByRole('button',{name:'Retry source images',exact:true}).waitFor();
  assert.equal(await page.getByText('No image attachments were saved with this source.',{exact:true}).count(),0);
  missing=true;await page.getByRole('button',{name:'Retry source images',exact:true}).click();
  await page.getByText('No image attachments were saved with this source.',{exact:true}).waitFor();
});
test('keyboard graph focus keeps its evidence highlighted when another node is hovered',async t=>{
  const {page}=await fixture(t);
  const record=page.locator('[data-node="episode:7"]');await record.waitFor();await record.focus();
  await page.locator('[data-node="session:codex:s1"]').dispatchEvent('mouseover');
  const targets=await page.locator('#edges .highlight').evaluateAll(es=>es.map(e=>e.dataset.target).sort());
  assert.deepEqual(targets,['chunk:9','episode:7']);
});
test('graph layouts preserve records and relationships with keyboard exploration on desktop and mobile',async t=>{
  for(const mobile of [false,true]) {
    const {page}=await fixture(t,{mobile});
    await page.locator('[data-node="episode:7"]').waitFor();
    const snapshot=()=>page.locator('[data-edge]').evaluateAll(es=>es.map(e=>[e.dataset.source,e.dataset.target]));
    const original=await snapshot(),shapes=[],backgrounds=[];
    for(const mode of ['constellation','radial','flow']) {
      await chooseLayout(page,mode);
      assert.deepEqual(await snapshot(),original);
      assert.equal(await page.locator('[data-node]').count(),5);
      shapes.push(await page.locator('[data-node]').evaluateAll(es=>es.map(e=>e.getAttribute('transform'))));
      backgrounds.push(await page.locator('.graph-panel').evaluate(el=>getComputedStyle(el).backgroundColor));
      const record=page.locator('[data-node="episode:7"]');await record.focus();
      assert.equal(await page.locator('#edges .highlight').count(),2);
      await page.keyboard.press('Enter');await page.locator('#inspector').waitFor();
      assert.match(await page.locator('#inspector').innerText(),/Keep launch local/);
      await page.getByRole('button',{name:'Close source inspector'}).click();
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'layout controls overflow');
      if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`graph-${mode}-${mobile?'mobile':'desktop'}.png`),fullPage:true});
    }
    assert.notDeepEqual(shapes[0],shapes[1]);assert.notDeepEqual(shapes[1],shapes[2]);
    assert.equal(backgrounds[0],backgrounds[1],'network perspectives must share the same light canvas');
  }
});

test('a late recall response cannot revive results after the query changes',async t=>{
  const {page}=await fixture(t);
  let release;
  const gate=new Promise(r=>{release=r;});
  let received;
  const started=new Promise(r=>{received=r;});
  await page.route('**/v1/recall?**',async route=>{received();await gate;await route.fulfill({json:{items:[{episode_id:7,text:'Stale launch result',score:1}]}}).catch(()=>{});});
  const input=page.getByRole('textbox',{name:'Test memory recall'});
  await input.fill('launch');await page.getByRole('button',{name:'Recall',exact:true}).click();await started;
  await input.fill('Cats');release();
  await page.getByText('Follow a question back to its source.',{exact:true}).waitFor();
  assert.doesNotMatch(await page.locator('#recall-results').innerText(),/Stale launch result/);
  assert.equal(await page.getByRole('button',{name:'Recall',exact:true}).isEnabled(),true);
});

test('long recall passages expand in page flow and are cleared when access expires',async t=>{
  const f=await fixture(t), {page}=f;
  const passage='A source-grounded decision. '.repeat(50)+'END OF SOURCE';
  await page.route('**/v1/recall?**',route=>route.fulfill({json:{items:[{episode_id:7,text:passage,score:1}]}}));
  await page.getByRole('textbox',{name:'Test memory recall'}).fill('decision');
  await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.getByText('Read full passage',{exact:true}).click();
  assert.match(await page.locator('.recall-full').innerText(),/END OF SOURCE/);
  const bounds=await page.locator('.recall-panel').evaluate(e=>({client:e.clientHeight,scroll:e.scrollHeight,overflow:getComputedStyle(e).overflowY}));
  assert.equal(bounds.overflow,'visible');assert.ok(bounds.scroll<=bounds.client+1);
  assert.match(await page.locator('#recall-results').innerText(),/not confidence/);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'recall-expanded.png'),fullPage:true});
  f.expire();await page.getByText('Connection needs authentication',{exact:true}).waitFor();
  assert.doesNotMatch(await page.locator('#recall-results').innerText(),/END OF SOURCE/);
});

test('live topology growth refits newly visible session groups into the canvas',async t=>{
  const f=await fixture(t),{page}=f;
  await page.locator('[data-node]').first().waitFor();
  f.grow();
  await page.locator('[data-group]').first().waitFor();
  await page.waitForFunction(()=>{
    const canvas=document.querySelector('#graph').getBoundingClientRect();
    return [...document.querySelectorAll('[data-node] rect,[data-group] rect')].every(e=>{const b=e.getBoundingClientRect();return b.x>=canvas.x&&b.y>=canvas.y&&b.right<=canvas.right+1&&b.bottom<=canvas.bottom+1;});
  },{},{timeout:4000});
  assert.equal(await page.locator('[data-group]').count(),4);
});
