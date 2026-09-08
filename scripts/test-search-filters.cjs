const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const contract=require('../tests/fixtures/http-capabilities.json').python;
let browser;
before(async()=>{browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH});});
after(async()=>{await browser?.close();});
async function fixture(t,{mobile=false,supported=true}={}){
  const html=fs.readFileSync(path.resolve(__dirname,'../python/memory/src/scone_memory/api/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','filter-fixture');
  const queries=[],writes=[];const state={fail:false};
  const server=http.createServer((req,res)=>{
    const url=new URL(req.url,'http://fixture');
    if(url.pathname==='/memory'){res.setHeader('content-type','text/html');return res.end(html);}
    if(url.pathname==='/favicon.ico'){res.statusCode=204;return res.end();}
    assert.equal(req.headers.authorization,'Bearer filter-fixture');res.setHeader('content-type','application/json');
    if(req.method!=='GET')writes.push(req.url);
    if(url.pathname==='/v1/capabilities')return res.end(JSON.stringify({...contract,features:{...contract.features,'recall.conditions':supported}}));
    if(url.pathname==='/v1/status')return res.end(JSON.stringify({space:'alpha',episodes:10}));
    if(url.pathname==='/v1/scopes')return res.end('{"scopes":{}}');
    if(url.pathname==='/v1/recall'){
      queries.push(url.searchParams);
      if(state.fail&&url.searchParams.has('conditions')){res.statusCode=422;return res.end('{"error":"Filter refused by server"}');}
      return res.end(JSON.stringify({items:[{episode_id:7,chunk_id:8,text:url.searchParams.has('conditions')?'Narrowed evidence':'Broad evidence',metadata:{},score:1}],facts:[],degraded:[]}));
    }
    res.statusCode=404;res.end('{}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/memory`);
  await page.getByText('Broad evidence',{exact:false}).waitFor();
  return {page,queries,state};
}
for(const mobile of [false,true])test(`metadata editor applies exact nested conditions only after confirmation, mobile=${mobile}`,async t=>{
  const {page,queries}=await fixture(t,{mobile});const before=queries.length;
  await page.getByRole('button',{name:'Metadata filters',exact:true}).click({timeout:2000});
  const editor=page.getByRole('region',{name:'Metadata filter editor',exact:true});
  await editor.getByLabel('Metadata field 1',{exact:true}).fill('status');
  await editor.getByLabel('Value 1',{exact:true}).fill('a,b: c');
  await editor.getByRole('button',{name:'Add group to root',exact:true}).click();
  await editor.getByLabel('Match rules in group 2',{exact:true}).selectOption('any');
  await editor.getByLabel('Metadata field 2.1',{exact:true}).fill('priority');
  await editor.getByLabel('Comparison 2.1',{exact:true}).selectOption('at_least');
  await editor.getByLabel('Value 2.1',{exact:true}).fill('10');
  await editor.getByLabel('Negate rule 1',{exact:true}).check();
  assert.equal(queries.length,before,'Editing is not submitting');
  await editor.getByRole('button',{name:'Apply metadata filter',exact:true}).click();
  await page.getByText('Narrowed evidence',{exact:false}).waitFor();
  const filtered=queries.slice(before);assert.equal(filtered.length,1);
  assert.deepEqual(JSON.parse(filtered[0].get('conditions')),{all:[{field:'status',not:true,is:'a,b: c'},{any:[{field:'priority',at_least:10}]}]});
  assert.equal(await page.getByText('Broad evidence',{exact:false}).count(),0);
  await page.getByRole('button',{name:'Edit metadata filter',exact:true}).click();
  assert.match(await editor.innerText(),/missing key never matches/i);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await editor.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`metadata-filter-${mobile?'mobile':'desktop'}.png`)});
});
test('invalid drafts and cancellation preserve the applied filter; failures retry it exactly',async t=>{
  const {page,queries,state}=await fixture(t);
  await page.getByRole('button',{name:'Metadata filters',exact:true}).click({timeout:2000});
  const editor=page.getByRole('region',{name:'Metadata filter editor',exact:true});
  await editor.getByLabel('Metadata field 1',{exact:true}).fill('priority');
  await editor.getByLabel('Comparison 1',{exact:true}).selectOption('above');
  await editor.getByLabel('Value 1',{exact:true}).fill('soon');
  const before=queries.length;await editor.getByRole('button',{name:'Apply metadata filter',exact:true}).click();
  await editor.getByRole('alert').waitFor();assert.equal(queries.length,before);
  await editor.getByLabel('Value 1',{exact:true}).fill('9');state.fail=true;
  await editor.getByRole('button',{name:'Apply metadata filter',exact:true}).click();
  await page.getByRole('button',{name:'Retry search',exact:true}).waitFor();
  assert.equal(queries.length,before+1,'Failure never falls back to broad recall');
  state.fail=false;await page.getByRole('button',{name:'Retry search',exact:true}).click();
  await page.getByText('Narrowed evidence',{exact:false}).waitFor();
  assert.equal(queries.at(-1).get('conditions'),queries.at(-2).get('conditions'));
  await page.getByRole('button',{name:'Edit metadata filter',exact:true}).click();
  await editor.getByLabel('Value 1',{exact:true}).fill('11');
  const applied=queries.length;await editor.getByRole('button',{name:'Cancel filter changes',exact:true}).click();
  assert.equal(queries.length,applied);
  await page.getByRole('button',{name:'Clear metadata filter',exact:true}).click();
  await page.getByText('Broad evidence',{exact:false}).waitFor();assert.equal(queries.at(-1).has('conditions'),false);
});
test('unsupported servers do not expose metadata condition controls',async t=>{
  const {page}=await fixture(t,{supported:false});
  assert.equal(await page.getByRole('button',{name:'Metadata filters',exact:true}).count(),0);
});
test('the deepest supported filter stays usable on a narrow screen',async t=>{
  const {page}=await fixture(t,{mobile:true});
  await page.getByRole('button',{name:'Metadata filters',exact:true}).click();
  let group='root';
  for(let depth=1;depth<8;depth++){
    await page.getByRole('button',{name:`Add group to ${group}`,exact:true}).click();
    group=group==='root'?'2':`${group}.2`;
  }
  assert.equal(await page.getByRole('button',{name:`Add group to ${group}`,exact:true}).isDisabled(),true);
  await page.getByLabel(`Metadata field ${group}.1`,{exact:true}).fill('priority');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'Eight supported groups must not widen the page');
  const field=await page.getByLabel(`Metadata field ${group}.1`,{exact:true}).boundingBox();
  assert.ok(field.width>=150,'Nested inputs must remain large enough to edit');
});
