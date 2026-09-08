// Exercise the packaged UI; examples never run against the user's server.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
let browser;
before(async()=>{browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});});
after(async()=>{await browser?.close();});
async function fixture(t,{mobile=false,clipboardFails=false}={}) {
  const html=fs.readFileSync(process.env.SCONE_LEARN_HTML||path.resolve(__dirname,'../crates/scone/src/playground.html'),'utf8');
  const requests=[];
  const server=http.createServer((req,res)=>{
    requests.push({url:req.url,method:req.method});
    if(req.url.startsWith('/v1/')){res.writeHead(401,{'content-type':'application/json'});return res.end('{"error":"key required"}');}
    res.writeHead(200,{'content-type':'text/html'});res.end(html);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000},reducedMotion:'reduce'});
  page.setDefaultTimeout(2500);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(fails=>{
    window.copiedText=null;
    Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>{if(fails)throw Error('denied');window.copiedText=text;}}});
  },clipboardFails);
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);});
  return {page,requests,base:`http://127.0.0.1:${server.address().port}`};
}

for(const mobile of [false,true])test(`concepts can be read and navigated without credentials, mobile=${mobile}`,async t=>{
  const {page,requests,base}=await fixture(t,{mobile});
  await page.goto(base+'/learn');
  await page.getByRole('heading',{name:'What is Scone?',exact:true}).waitFor();
  assert.equal(await page.getByRole('dialog').count(),0,'reading concepts must not open a key dialog');
  const nav=page.getByRole('navigation',{name:'Scone concepts'});
  if(mobile)await page.getByRole('button',{name:'Browse documentation',exact:true}).click();
  await nav.getByRole('link',{name:'How it works',exact:true}).click();
  await page.getByRole('heading',{name:'How Scone works',exact:true}).waitFor();
  await page.reload();
  await page.getByRole('heading',{name:'How Scone works',exact:true}).waitFor();
  if(mobile)await page.getByRole('button',{name:'Browse documentation',exact:true}).click();
  await nav.getByRole('link',{name:'Graph memory',exact:true}).click();
  await page.getByRole('heading',{name:'Graph memory',exact:true}).waitFor();
  await page.getByRole('button',{name:'Copy page',exact:true}).click();
  const copied=await page.evaluate(()=>window.copiedText);
  assert.match(copied,/^# Graph memory/);
  assert.match(copied,/Updates/);
  assert.doesNotMatch(copied,/__SCONE_TOKEN__|Bearer [A-Za-z0-9]{10}/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(requests.filter(r=>r.url.startsWith('/v1/')),[],'concepts browsing must not read or write memory');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`learn-graph-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  await page.getByRole('link',{name:'Open Review',exact:true}).click();
  assert.equal(new URL(page.url()).pathname,'/memory');
  assert.equal(new URL(page.url()).hash,'#review');
  await page.getByRole('heading',{name:'Connect to Scone',exact:true}).waitFor();
});

for(const mobile of [false,true])test(`documentation covers the complete product without a second app sidebar, mobile=${mobile}`,async t=>{
  const {page,base,requests}=await fixture(t,{mobile});await page.goto(base+'/learn');
  await page.getByRole('banner',{name:'Scone documentation'}).waitFor();
  assert.equal(await page.locator('.server-strip,.app-shell .topbar').count(),0);
  if(mobile)await page.getByRole('button',{name:'Browse documentation',exact:true}).click();
  const nav=page.getByRole('navigation',{name:'Scone concepts'});
  for(const name of ['Quickstart','Add sources','Search & retrieval','Review & control','Profiles','Conversations & voice','Spaces & keys','API reference'])assert.equal(await nav.getByRole('link',{name,exact:true}).count(),1);
  await nav.getByRole('link',{name:'Quickstart',exact:true}).click();
  await page.getByRole('heading',{name:'Your first memory',exact:true}).waitFor();
  await page.getByRole('button',{name:'Copy example',exact:true}).click();
  assert.match(await page.evaluate(()=>window.copiedText),/Bearer \$SCONE_KEY/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(requests.filter(r=>r.url.startsWith('/v1/')),[]);
  if(process.env.SCONE_SCREENSHOT_DIR){await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`docs-quickstart-${mobile?'mobile':'desktop'}.png`),fullPage:true});}
});
for(const mobile of [false,true])test(`documentation reading tools stay in their intended columns, mobile=${mobile}`,async t=>{
  const {page,base}=await fixture(t,{mobile});await page.goto(base+'/learn');
  await page.getByRole('heading',{name:'What is Scone?',exact:true}).waitFor();
  if(mobile){
    assert.equal(await page.getByRole('navigation',{name:'Scone concepts'}).isVisible(),false,'Closed browse menu does not consume article space');
    const header=await page.getByRole('banner',{name:'Scone documentation'}).boundingBox();
    const heading=await page.getByRole('heading',{name:'What is Scone?',exact:true}).boundingBox();
    assert.ok(heading.y<header.y+header.height+180,'Reading starts near the header, not after a hidden sidebar');
    await page.getByRole('button',{name:'Browse documentation',exact:true}).click();
    await page.getByRole('navigation',{name:'Scone concepts'}).getByRole('link',{name:'API reference',exact:true}).click();
    await page.getByRole('table').first().waitFor();
    assert.equal(await page.getByRole('navigation',{name:'Scone concepts'}).isVisible(),false);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  }else{
    const main=await page.locator('#main').boundingBox();
    const outline=await page.getByRole('navigation',{name:'On this page'}).boundingBox();
    assert.ok(outline.x>=main.x+main.width,'Section navigation belongs beside the article');
    assert.ok(outline.y<main.y+60,'Section navigation must not fall beneath the full article');
  }
});
test('documentation search finds a workflow and opens its direct route',async t=>{
  const {page,base,requests}=await fixture(t);await page.goto(base+'/learn');
  await page.getByRole('searchbox',{name:'Search documentation',exact:true}).fill('transcription');
  await page.getByRole('navigation',{name:'Documentation search results'}).getByRole('link',{name:/Conversations & voice/}).click();
  assert.equal(new URL(page.url()).pathname,'/learn/conversations');
  await page.getByRole('heading',{name:'Conversations that remember',exact:true}).waitFor();
  await page.reload();await page.getByRole('heading',{name:'Conversations that remember',exact:true}).waitFor();
  await page.getByRole('searchbox',{name:'Search documentation',exact:true}).fill('unfindablefixtureterm');
  await page.getByText('No matching documentation. Try a feature or API name.',{exact:true}).waitFor();
  assert.deepEqual(requests.filter(r=>r.url.startsWith('/v1/')),[]);
});

test('quickstart copy preserves shell placeholders without executing the example',async t=>{
  const {page,requests,base}=await fixture(t);
  await page.goto(base+'/learn/how-it-works');
  await page.getByRole('button',{name:'Copy example',exact:true}).click();
  const example=await page.evaluate(()=>window.copiedText);
  assert.match(example,/\$SCONE_URL\/v1\/episodes/);
  assert.match(example,/Bearer \$SCONE_KEY/);
  assert.match(example,/--data-urlencode/);
  assert.ok(example.includes('Polaris'));
  assert.deepEqual(requests.filter(r=>r.method!=='GET'),[]);
});

test('clipboard denial offers selectable text, and page navigation clears copy feedback',async t=>{
  const {page,base}=await fixture(t,{clipboardFails:true});
  await page.goto(base+'/learn/how-it-works');
  await page.getByRole('button',{name:'Copy example',exact:true}).click();
  await page.getByRole('status').filter({hasText:'Select and copy the text below'}).waitFor();
  assert.ok((await page.getByRole('region',{name:'Source and recall example'}).locator('pre').textContent()).includes('Polaris'));
  await page.getByRole('navigation',{name:'Scone concepts'}).getByRole('link',{name:'Overview',exact:true}).click();
  assert.equal(await page.getByText('Select and copy the text below.').count(),0);
});

for(const mobile of [false,true])test(`copy recovery text stays inside its workspace layout, mobile=${mobile}`,async t=>{
  const {page,base}=await fixture(t,{mobile,clipboardFails:true});
  await page.goto(base+'/learn/how-it-works');
  await page.getByRole('button',{name:'Copy page',exact:true}).click();
  await page.locator('.workspace-heading .learn-copy [role=status]').filter({hasText:/\S/}).waitFor();
  const boxes=await page.locator('.workspace-heading .learn-copy').evaluate(el=>{
    const outer=el.getBoundingClientRect();
    return {outer:{top:outer.top,bottom:outer.bottom},children:Array.from(el.children).map(child=>{
      const r=child.getBoundingClientRect();return {top:r.top,bottom:r.bottom};
    })};
  });
  for(const child of boxes.children){assert.ok(child.top>=boxes.outer.top-1);assert.ok(child.bottom<=boxes.outer.bottom+1,'Copy action and recovery must fit their header rather than overlap the article.');}
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});

test('keyboard navigation and section links keep the selected concept and reading position',async t=>{
  const {page,base}=await fixture(t);
  await page.goto(base+'/learn');
  const link=page.getByRole('navigation',{name:'Scone concepts'}).getByRole('link',{name:'Graph memory',exact:true});
  await link.focus();await page.keyboard.press('Enter');
  const title=page.getByRole('heading',{name:'Graph memory',exact:true});
  await title.waitFor();
  assert.equal(await title.evaluate(element=>element===document.activeElement),true);
  await page.getByRole('navigation',{name:'On this page'}).getByRole('link',{name:'Derivations: an inference names its premises',exact:true}).click();
  assert.equal(new URL(page.url()).hash,'#derivations');
  const section=page.locator('#derivations');
  const belowHeader=()=>page.evaluate(()=>{
    const section=document.getElementById('derivations')?.getBoundingClientRect();
    const header=document.querySelector('.docs-header')?.getBoundingClientRect();
    return Boolean(section&&header&&section.top>=header.bottom&&section.top<header.bottom+48);
  });
  assert.equal(await belowHeader(),true,'Section anchors sit just below the sticky navigation, not behind it.');
  await page.reload();
  await section.waitFor();
  await page.waitForFunction(()=>{
    const section=document.getElementById('derivations')?.getBoundingClientRect();
    const header=document.querySelector('.docs-header')?.getBoundingClientRect();
    return section&&header&&section.top>=header.bottom&&section.top<header.bottom+48;
  });
  assert.equal(new URL(page.url()).pathname,'/learn/graph-memory');
});
