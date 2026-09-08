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
async function fixture(t,{supported=true,mobile=false}={}){
  const html=fs.readFileSync(process.env.SCONE_PLAYGROUND_HTML||path.resolve(__dirname,'../python/memory/src/scone_memory/api/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','review-fixture');
  const requests=[];
  const server=http.createServer((req,res)=>{
    const url=new URL(req.url,'http://fixture');
    if(url.pathname==='/memory'){res.setHeader('content-type','text/html');return res.end(html);}
    if(url.pathname==='/favicon.ico'){res.statusCode=204;return res.end();}
    requests.push({method:req.method,path:req.url});
    assert.equal(req.headers.authorization,'Bearer review-fixture');res.setHeader('content-type','application/json');
    if(url.pathname==='/v1/capabilities')return res.end(JSON.stringify({...contract,features:{...contract.features,'status.read':supported,'facts.read':supported}}));
    if(url.pathname==='/v1/facts')return res.end('{"facts":[]}');
    if(url.pathname==='/v1/status')return res.end('{"space":"alpha","episodes":10,"pending_distill":539}');
    res.statusCode=404;res.end('{}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.method==='GET'),'empty state never starts processing or changes claims');});
  await page.goto(`http://127.0.0.1:${server.address().port}/memory#review`);
  await page.getByText('Nothing awaits review.',{exact:true}).waitFor();
  return {page,requests};
}
for(const mobile of [false,true])test(`empty Review explains proposals and offers supported destinations, mobile=${mobile}`,async t=>{
  const {page}=await fixture(t,{mobile});
  const state=page.locator('.review-empty');
  assert.match(await state.innerText(),/Review only shows proposed claims/);
  assert.match(await state.innerText(),/An empty queue does not mean extraction is complete/);
  assert.doesNotMatch(await state.innerText(),/Nothing is approved automatically/);
  assert.match(await state.getByRole('link',{name:'Check processing status'}).getAttribute('href'),/\/memory#status$/);
  assert.match(await state.getByRole('link',{name:'View memory claims'}).getAttribute('href'),/\/memory#beliefs$/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await state.getByRole('link',{name:'Check processing status'}).click();
  await page.getByRole('heading',{name:'Memory processing',exact:true}).waitFor();
});
test('empty Review hides destinations absent from the host capability contract',async t=>{
  const {page}=await fixture(t,{supported:false});
  assert.equal(await page.locator('.review-empty').getByRole('link').count(),0);
});
