// Isolated real React workflows. Every API request is intercepted; no live data writes.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
let server,browser,base;
const required=['recall','facts.read','facts.review','facts.close','facts.exclude','facts.include','events.read','metrics.read','scopes.read','status.read'];
const capabilities={schema_version:1,implementation:'fixture',features:{...Object.fromEntries(required.map(key=>[key,key==='recall'||key==='status.read'])),'episodes.read':true}};
const quote='We selected Cedar because its retained-source audit was complete.';
function response(query='Why did we choose Cedar?'){
  const evidence_graph={nodes:[
    {id:'query:current',kind:'query',label:query,data:{event_id:4}},
    {id:'chunk:2',kind:'chunk',label:'The Cedar decision',data:{chunk_id:2,episode_id:3,text:quote,score:.7,similarity:.62}},
    {id:'episode:3',kind:'episode',label:'Decision log',data:{episode_id:3,source:'Decision log',preview:quote,kind:'note'}},
    {id:'claim:5',kind:'claim',label:'Team uses Cedar',data:{fact_id:5,subject:'Team',predicate:'uses',object:'Cedar',origin:'extracted',status:'active',source_episode_id:3,quote,grounded:true,provenance_status:'retained'}},
    {id:'claim:6',kind:'claim',label:'Team uses Maple',data:{fact_id:6,subject:'Team',predicate:'uses',object:'Maple',origin:'stated',status:'closed',provenance_status:'unstated'}},
    {id:'concept:team',kind:'concept',label:'Team',data:{name:'Team',basis:'retained_claim'}},
    {id:'concept:cedar',kind:'concept',label:'Cedar',data:{name:'Cedar',basis:'retained_claim'}},
  ],edges:[
    {source:'query:current',target:'chunk:2',kind:'returned',label:'Returned by recall',data:{category:'retrieval'}},
    {source:'query:current',target:'claim:5',kind:'returned',label:'Returned claim',data:{category:'retrieval'}},
    {source:'query:current',target:'claim:6',kind:'returned',label:'Returned claim',data:{category:'retrieval'}},
    {source:'episode:3',target:'chunk:2',kind:'chunked_into',label:'Contains source bytes',data:{category:'membership'}},
    {source:'episode:3',target:'claim:5',kind:'source_of',label:'Recorded claim source',data:{category:'provenance',quote,verified:true}},
    {source:'claim:5',target:'claim:6',kind:'contradicts',label:'contradicts',data:{category:'fact_relation',link_id:8,source_episode_id:3,quote,provenance_status:'retained'}},
      {source:'concept:team',target:'concept:cedar',kind:'relation',label:'uses',data:{category:'fact_relation',predicate:'uses',fact_id:5,source_episode_id:3,quote,provenance_status:'retained'}},
    {source:'claim:5',target:'concept:team',kind:'asserts',data:{role:'subject',fact_id:5}},
    {source:'claim:5',target:'concept:cedar',kind:'asserts',data:{role:'object',fact_id:5}},
    {source:'chunk:2',target:'concept:cedar',kind:'mentions',data:{quote:'Cedar'}},
  ],truncated:false,provenance_missing:0,provenance_omitted:0,notices:[],counts:{chunks:1,claims:2,sources:1,links:1}};
  return {event_id:4,items:[{chunk_id:2,episode_id:3,text:quote,score:.7,created_at:'2026-09-07T12:00:00Z',source:'Decision log',metadata:{},tags:[]}],facts:[],degraded:[],returned_bytes:80,space_bytes:300,evidence_graph};
}
before(async()=>{
  const root=path.resolve(__dirname,'../Webapp');
  const {createServer}=await import(pathToFileURL(path.join(root,'node_modules/vite/dist/node/index.js')).href);
  server=await createServer({root,server:{port:0,strictPort:false,proxy:{'/v1':undefined,'/healthz':undefined,'/__native_console':undefined}}});
  await server.listen();base=`http://127.0.0.1:${server.httpServer.address().port}`;
  browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});
});
after(async()=>{await browser?.close();await server?.close();});
async function fixture(t,{mode='',handle,viewport={width:1280,height:1000}}={}){
  const page=await browser.newPage({viewport});t.after(()=>page.close());
  const errors=[],requests=[];page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/v1/**',async route=>{
    const request=route.request(),url=new URL(request.url());requests.push({method:request.method(),path:url.pathname,auth:request.headers().authorization});
    assert.equal(request.method(),'GET','Evidence inspection must stay read-only');
    if(handle&&await handle(route,url))return;
    if(url.pathname==='/v1/status')return route.fulfill({json:{space:request.headers().authorization==='Bearer fixture-b'?'beta':'alpha',episodes:1,chunks:1}});
    if(url.pathname==='/v1/capabilities')return route.fulfill({json:capabilities});
    if(url.pathname==='/v1/recall'){
      if(url.searchParams.get('q')!=='*')assert.equal(url.searchParams.get('evidence_graph'),'true');
      return route.fulfill({json:response(url.searchParams.get('q'))});
    }
    if(url.pathname==='/v1/scopes')return route.fulfill({json:{scopes:{}}});
    if(url.pathname==='/v1/episodes/3')return route.fulfill({json:{episode_id:3,kind:'note',content:quote,source:'Decision log',created_at:'2026-09-07T12:00:00Z',attachments:[]}});
    if(url.pathname==='/v1/conversations/fixture')return route.fulfill({json:{session_id:'fixture',space:'alpha',state:'running',revision:1,created_at:'2026-09-07T12:00:00Z',latest_request_id:'fixture'}});
    if(url.pathname==='/v1/conversations/fixture/transcript')return route.fulfill({json:{episodes:Array.from({length:20},(_,i)=>({episode_id:i+20,content:'A retained conversation message. '.repeat(18),metadata:{role:i%2?'assistant':'user'}})),has_more:false,next_before:null}});
    if(url.pathname==='/v1/conversations/fixture/turns/fixture')return route.fulfill({json:{request_id:'fixture',status:'completed',result:{text:'Cedar was selected.',assistant_episode_id:10,memory_context:{status:'prepared',references:[{episode_id:3,chunk_id:2}],evidence_graph:response().evidence_graph}}}});
    throw Error(`Unexpected fixture request ${url.pathname}`);
  });
  await page.goto(base+'/tests/fixtures/query-evidence.html'+(mode?'?mode='+mode:''));
  t.after(()=>assert.deepEqual(errors,[],'Browser must not emit runtime errors'));
  return {page,requests};
}
async function recall(page,query='Why did we choose Cedar?'){
  await page.getByRole('textbox',{name:'Test memory recall',exact:true}).fill(query);
  await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.getByRole('heading',{name:'Query evidence map',exact:true}).waitFor();
}
async function screenshot(page,name){if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,name+'.png'),fullPage:await page.getByRole('dialog').count()===0});}
async function assertNoOverflow(page){assert.deepEqual(await page.evaluate(()=>{
  const bad=[];if(document.documentElement.scrollWidth>innerWidth)bad.push('document');
  for(const element of document.querySelectorAll('.query-evidence-summary,.evidence-explorer,.en-main,.en-inspector,.en-toolbar'))if(element.scrollWidth>element.clientWidth+1)bad.push(element.className);
  return bad;
}),[],'Graph and inspector must fit their containers');}
async function openExplorer(page){
  await page.getByRole('button',{name:'Open evidence explorer',exact:false}).click();
  const dialog=page.getByRole('dialog',{name:'Evidence explorer',exact:true});await dialog.waitFor();return dialog;
}
async function evidenceView(dialog){await dialog.getByRole('button',{name:'Evidence',exact:true}).click();}
async function selectNode(dialog,name){await dialog.getByRole('button',{name,exact:true}).click();}

test('Playground explores real concept links, pans and zooms, then traces membership to its source',async t=>{
  const {page,requests}=await fixture(t);await recall(page);
  const dialog=await openExplorer(page);
  assert.equal(await dialog.locator('[data-evidence-node]').count(),2);
  await dialog.getByRole('button',{name:'uses: Team → Cedar',exact:true}).click();
  assert.equal(await dialog.locator('.en-selected h3').innerText(),'uses');
  assert.equal(await dialog.locator('.en-selected blockquote').innerText(),quote);
  const zoom=await dialog.getByLabel('Graph zoom',{exact:true}).innerText();
  await dialog.getByRole('button',{name:'Zoom in',exact:true}).click();
  assert.notEqual(await dialog.getByLabel('Graph zoom',{exact:true}).innerText(),zoom);
  await dialog.getByRole('button',{name:'Fit graph',exact:true}).click();
  const svg=dialog.getByRole('group',{name:'Interactive evidence network',exact:true}),box=await svg.boundingBox();
  const before=await svg.locator(':scope > g').first().getAttribute('transform');
  await page.mouse.move(box.x+15,box.y+70);await page.mouse.down();await page.mouse.move(box.x+75,box.y+120);await page.mouse.up();
  assert.notEqual(await svg.locator(':scope > g').first().getAttribute('transform'),before);
  await dialog.getByRole('button',{name:'Fit graph',exact:true}).click();
  await screenshot(page,'query-evidence-desktop');
  await evidenceView(dialog);await selectNode(dialog,'chunk: The Cedar decision');
  assert.match(await dialog.locator('.en-selected').innerText(),/Ranking score 0.700.*Similarity 0.620/s);
  await dialog.getByRole('button',{name:/Contains passage.*Source membership/}).click();
  assert.match(await dialog.locator('.en-selected').innerText(),/This is a membership link/);
  await dialog.locator('.en-selected').getByRole('button',{name:'Decision log',exact:true}).click();
  const link=dialog.locator('.en-selected').getByRole('link',{name:'Open source page',exact:true});
  await link.waitFor();assert.equal(await link.getAttribute('href'),'/memory/sources/3?space=alpha');
  await link.click();await page.locator('.source-page-original').waitFor();
  assert.match(await page.locator('.source-page-original').innerText(),/We selected Cedar/);
  assert.equal(requests.filter(request=>request.path==='/v1/episodes/3').length,1);
  assert.equal(await page.getByRole('dialog').count(),0);
});

test('Memory Search preserves stored claim direction, quote and source navigation in mobile explorer',async t=>{
  const {page}=await fixture(t,{mode:'memory'});
  await page.getByRole('searchbox',{name:'Search your memory',exact:true}).fill('Why did we choose Cedar?');
  await page.getByRole('button',{name:'Search',exact:true}).last().click();
  const dialog=await openExplorer(page);await evidenceView(dialog);
  await selectNode(dialog,'claim: Team uses Cedar');
  await dialog.getByRole('button',{name:/Contradicts.*Recorded claim relationship/}).click();
  assert.equal(await dialog.locator('.en-selected blockquote').innerText(),quote);
  assert.match(await dialog.locator('.en-selected').innerText(),/not proof that either statement is correct/);
  await assertNoOverflow(page);await screenshot(page,'query-evidence-claim-selected');
  await page.setViewportSize({width:390,height:844});await assertNoOverflow(page);await screenshot(page,'query-evidence-mobile');
  const link=dialog.locator('.en-selected').getByRole('link',{name:'Open source page',exact:true});await link.waitFor();
  await link.click();await page.locator('.source-page-original').waitFor();
  assert.match(await page.locator('.source-page-original').innerText(),/We selected Cedar/);
});

test('new queries and credential changes revoke open explorers and discard delayed results',async t=>{
  let release,arrived;const held=new Promise(resolve=>{release=resolve;}),requested=new Promise(resolve=>{arrived=resolve;});
  const {page}=await fixture(t,{handle:async(route,url)=>{
    if(url.pathname!=='/v1/recall'||url.searchParams.get('q')!=='Delayed old query')return false;
    arrived();await held;await route.fulfill({json:response('Stale response must disappear')}).catch(()=>{});return true;
  }});
  await recall(page);await page.getByRole('textbox',{name:'Test memory recall',exact:true}).fill('A new question');
  assert.equal(await page.locator('.query-evidence-summary').count(),0);
  await recall(page,'Replacement query');await openExplorer(page);
  await page.evaluate(()=>[...document.querySelectorAll('button')].find(button=>button.textContent==='Switch fixture connection').click());
  await page.getByRole('heading',{name:'Follow a question back to its source.',exact:true}).waitFor();
  assert.equal(await page.getByRole('dialog').count(),0);assert.equal(await page.locator('.query-evidence-summary').count(),0);
  await page.getByRole('textbox',{name:'Test memory recall',exact:true}).fill('Delayed old query');
  await page.getByRole('button',{name:'Recall',exact:true}).click();await requested;
  await page.getByRole('button',{name:'Switch fixture connection',exact:true}).click();release();
  await page.getByRole('heading',{name:'Follow a question back to its source.',exact:true}).waitFor();
  await recall(page,'New credential query');
  assert.doesNotMatch(await page.locator('body').innerText(),/Stale response must disappear|Replacement query/);
  const dialog=await openExplorer(page);await evidenceView(dialog);await selectNode(dialog,'chunk: The Cedar decision');
  const link=dialog.locator('.en-selected').getByRole('link',{name:'Open source page',exact:true});await link.waitFor();
  assert.equal(await link.getAttribute('href'),'/memory/sources/3?space=alpha');
});

test('280px reply launcher stays compact; modal traps focus and preserves selection, scroll and opener focus',async t=>{
  const {page,requests}=await fixture(t,{mode:'reply'});
  const summary=page.getByRole('region',{name:'Evidence supplied to this reply',exact:true});await summary.waitFor();
  assert.ok((await summary.boundingBox()).height<250,'Reply evidence must remain a compact launcher');
  assert.equal(await page.locator('.en-svg').count(),0);
  await screenshot(page,'query-evidence-narrow-reply');
  const scrollBefore=await page.evaluate(()=>window.scrollY),dialog=await openExplorer(page);
  await dialog.getByRole('button',{name:'uses: Team → Cedar',exact:true}).click();
  await page.evaluate(()=>[...document.querySelectorAll('button')].find(button=>button.textContent==='Poll identical reply receipt').click());
  await page.getByText('Receipt poll 2',{exact:true}).waitFor();
  assert.equal(await dialog.locator('.en-selected h3').innerText(),'uses');
  assert.equal(await dialog.locator('.en-selected blockquote').innerText(),quote);
  assert.match(await dialog.locator('.en-context').innerText(),/not a model reasoning trace/);
  assert.equal(requests.filter(request=>request.path==='/v1/recall').length,0);
  await dialog.getByRole('button',{name:'Close evidence explorer',exact:true}).focus();
  await page.keyboard.press('Shift+Tab');
  assert.equal(await page.evaluate(()=>document.querySelector('dialog').contains(document.activeElement)),true);
  for(let i=0;i<18;i++){await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.querySelector('dialog').contains(document.activeElement)),true);}
  await page.keyboard.press('Escape');assert.equal(await page.getByRole('dialog').count(),0);
  assert.match(await page.evaluate(()=>document.activeElement.textContent),/Open evidence explorer/);
  assert.equal(await page.evaluate(()=>window.scrollY),scrollBefore);
  await page.setViewportSize({width:390,height:844});await openExplorer(page);await assertNoOverflow(page);await screenshot(page,'query-evidence-mobile-reply');
});

test('100 concept nodes remain accessible, groupable and bounded without label collisions',async t=>{
  const crowded=response('Large concept network');crowded.evidence_graph.nodes=Array.from({length:100},(_,i)=>({id:`concept:${i}`,kind:'concept',label:`Retained concept ${i} with a deliberately long name`,data:{name:`Concept ${i}`,basis:'retained_claim'}}));
  crowded.evidence_graph.edges=Array.from({length:99},(_,i)=>({source:`concept:${i}`,target:`concept:${i+1}`,kind:'relation',label:'depends on',data:{predicate:'depends on',fact_id:i+1}}));
  const {page}=await fixture(t,{handle:async(route,url)=>{if(url.pathname!=='/v1/recall')return false;await route.fulfill({json:crowded});return true;}});
  await recall(page);const dialog=await openExplorer(page);
  assert.equal(await dialog.locator('[data-evidence-node]').count(),100);
  assert.equal(await dialog.locator('[data-evidence-edge]').count(),99);
  assert.ok(await dialog.locator('.en-node-label').count()<=28);
  await dialog.getByLabel('Community colors',{exact:true}).uncheck();
  await dialog.getByLabel('Community colors',{exact:true}).check();
  await dialog.getByLabel('Community filter',{exact:true}).selectOption({index:1});
  assert.equal(await dialog.locator('[data-evidence-node]').count(),100);
  await dialog.getByRole('searchbox',{name:'Find a node',exact:true}).fill('concept 99');
  await dialog.locator('.en-record-list').getByRole('button').click();
  assert.match(await dialog.locator('.en-selected h3').innerText(),/concept 99/);
  await dialog.getByLabel('Focus selection',{exact:true}).check();
  assert.equal(await dialog.locator('[data-evidence-node]').count(),2);
  await dialog.getByRole('button',{name:'Reset',exact:true}).click();
  assert.equal(await dialog.locator('[data-evidence-node]').count(),100);
  await page.setViewportSize({width:390,height:844});await assertNoOverflow(page);await screenshot(page,'query-evidence-100-mobile');
});

test('detected communities use recorded relations and selecting a group preserves its real subgraph',async t=>{
  const grouped=response('How are these concepts connected?'),names=['Scone','Rust','Memory','Cedar','Decisions','Sources','Maple','Agents','Recall'];
  grouped.evidence_graph.nodes=names.map((label,i)=>({id:`concept:${i}`,kind:'concept',label,data:{name:label,basis:'retained_claim'}}));
  const pairs=[[0,1],[0,2],[1,2],[3,4],[3,5],[4,5],[6,7],[6,8],[7,8],[2,3],[5,6]];
  grouped.evidence_graph.edges=pairs.map(([source,target],i)=>({source:`concept:${source}`,target:`concept:${target}`,kind:'relation',label:'uses',data:{predicate:'uses',fact_id:i+1}}));
  const {page}=await fixture(t,{handle:async(route,url)=>{if(url.pathname!=='/v1/recall')return false;await route.fulfill({json:grouped});return true;}});
  await recall(page);const dialog=await openExplorer(page);
  assert.match(await dialog.locator('.en-display-options').innerText(),/Detected communities/);
  assert.equal(await dialog.locator('.en-group-legend button').count(),3);
  await screenshot(page,'query-evidence-communities');
  await dialog.getByLabel('Community filter',{exact:true}).selectOption({index:2});
  assert.equal(await dialog.locator('[data-evidence-node]').count(),3);
  assert.equal(await dialog.locator('[data-evidence-edge]').count(),3);
});

test('literal concepts without semantic relations open the connected evidence view without inventing links',async t=>{
  const literal=response('Find retained mentions');
  literal.evidence_graph.nodes=literal.evidence_graph.nodes.filter(node=>node.kind!=='claim');
  for(const node of literal.evidence_graph.nodes)if(node.kind==='concept')node.data.basis='literal_mention';
  literal.evidence_graph.edges=literal.evidence_graph.edges.filter(edge=>['returned','chunked_into','mentions'].includes(edge.kind)&&!edge.target.startsWith('claim:'));
  const {page}=await fixture(t,{handle:async(route,url)=>{if(url.pathname!=='/v1/recall')return false;await route.fulfill({json:literal});return true;}});
  await recall(page);const dialog=await openExplorer(page);
  assert.equal(await dialog.getByRole('button',{name:'Evidence',exact:true}).getAttribute('aria-pressed'),'true');
  assert.equal(await dialog.locator('[data-evidence-node]').count(),5);
  assert.equal(await dialog.locator('[data-evidence-edge]').count(),3);
  await dialog.getByRole('button',{name:'Concepts',exact:true}).click();
  assert.equal(await dialog.locator('[data-evidence-node]').count(),2);
  assert.equal(await dialog.locator('[data-evidence-edge]').count(),0);
  assert.match(await dialog.locator('.en-display-options').innerText(),/Unlinked concepts/);
  await dialog.getByRole('button',{name:'Reset',exact:true}).click();
  assert.equal(await dialog.getByRole('button',{name:'Evidence',exact:true}).getAttribute('aria-pressed'),'true');
  assert.equal(await dialog.locator('[data-evidence-edge]').count(),3);
});

test('opening reply evidence leaves chat height, saved-message scroll and composer position unchanged',async t=>{
  const {page}=await fixture(t,{mode:'chat',viewport:{width:1880,height:1040}});
  await page.getByRole('button',{name:'Open evidence explorer',exact:false}).waitFor();
  const composer=await page.locator('.conversation-composer').boundingBox();
  assert.ok(composer.y+composer.height<=1040,'Composer must remain reachable inside the desktop viewport');
  assert.ok(Math.abs((await page.locator('.conversation-evidence').boundingBox()).height-(await page.locator('.conversation-thread').boundingBox()).height)<1,'The sidebar must scroll within the same height as the chat');
  const metrics=()=>page.evaluate(()=>({thread:document.querySelector('.conversation-thread').getBoundingClientRect().height,page:document.documentElement.scrollHeight,composer:document.querySelector('.conversation-composer').getBoundingClientRect().bottom,scroll:document.querySelector('.conversation-messages').scrollTop}));
  await page.locator('.conversation-messages').evaluate(element=>{element.scrollTop=170;});
  const before=await metrics();await screenshot(page,'query-evidence-chat-compact');
  const dialog=await openExplorer(page);assert.deepEqual(await metrics(),before,'Opening the modal must not stretch or scroll the chat');
  await dialog.getByRole('button',{name:'Close evidence explorer',exact:true}).click();assert.deepEqual(await metrics(),before);
  assert.match(await page.evaluate(()=>document.activeElement.textContent),/Open evidence explorer/);
});
