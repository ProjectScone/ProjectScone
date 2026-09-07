// Browser-level UI contract, using the real shipped HTML and a deterministic HTTP
// boundary. Run: NODE_PATH=<directory containing playwright> node --test scripts/test-rust-console.cjs
// No model, database, account, or network outside the disposable loopback server.
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { createServer } = require('node:http');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { chromium } = require(process.env.SCONE_PLAYWRIGHT_MODULE || 'playwright');
const { Script } = require('node:vm');

const html = readFileSync(process.env.SCONE_CONSOLE_HTML || resolve(__dirname, '../crates/scone/src/console.html'), 'utf8')
  .replaceAll('__SCONE_TOKEN__', 'ui-test-only');
new Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
let browser;
before(async () => {
  browser = await chromium.launch({ headless: true, timeout: 15000, args: ['--disable-gpu'],
    ...(process.env.SCONE_BROWSER_PATH ? { executablePath: process.env.SCONE_BROWSER_PATH } : { channel: 'chrome' }) });
});
after(async () => { if (browser) await browser.close(); });
const status = { space: 'studio', episodes: 12, chunks: 24, revision: 5,
  semantic_lane: 'paused', pending_distill: 0 };
const item = { episode_id: 7, text: 'Keep the launch local. Café 🥐 <script>untrusted</script>',
  score: 0.72, source: 'https://example.com/notes/launch', created_at: '2026-09-05T12:00:00.000Z' };
const belief = { fact_id: 3, subject: 'Launch', predicate: 'environment', object: 'local',
  confidence: 0.8, valid_from: '2026-09-01T00:00:00.000Z', valid_until: null, status: 'active' };

async function withPage(run, options = {}) {
  const requests = [];
  let closed = false;
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/') {
      res.writeHead(200, { 'content-type': 'text/html' }).end(html); return;
    }
    if (url.pathname === '/favicon.ico') { res.writeHead(204).end(); return; }
    requests.push({ url, method: req.method });
    res.setHeader('content-type', 'application/json');
    if (req.headers.authorization !== 'Bearer ui-test-only' || options.unauthorized) {
      res.writeHead(401).end(JSON.stringify({ error: 'unknown key' })); return;
    }
    if (url.pathname === '/v1/status') {
      if (options.statusDelay) await new Promise(r => setTimeout(r, options.statusDelay));
      if (options.statusError) { res.writeHead(503).end(JSON.stringify({ error: 'Status unavailable' })); return; }
      res.end(JSON.stringify(status)); return;
    }
    if (url.pathname === '/v1/recall') {
      if (options.delay) await new Promise(r => setTimeout(r, 250));
      res.end(JSON.stringify({ items: [{ ...item, ...(options.item || {}) }], facts: [],
        degraded: [], returned_bytes: 80, space_bytes: 800, context_reduction: 0.9 })); return;
    }
    if (url.pathname === '/v1/facts') {
      res.end(JSON.stringify({ facts: [{ ...belief, status: closed ? 'closed' : 'active',
        valid_until: closed ? '2026-09-06T00:00:00.000Z' : null }] })); return;
    }
    if (url.pathname === '/v1/facts/3/close' && req.method === 'POST') {
      let body = ''; for await (const chunk of req) body += chunk;
      requests.at(-1).body = JSON.parse(body); closed = true;
      res.end(JSON.stringify({ closed: 3 })); return;
    }
    if (url.pathname === '/v1/tags') { res.end(JSON.stringify({ tags: [{ name: 'work', count: 3 }] })); return; }
    if (url.pathname === '/v1/profile') { res.end(JSON.stringify({ static_facts: [], dynamic: [] })); return; }
    res.writeHead(404).end(JSON.stringify({ error: 'unsupported' }));
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const page = await browser.newPage({ viewport: options.viewport || { width: 1440, height: 1000 },
    colorScheme: options.colorScheme || 'light' });
  page.setDefaultTimeout(1800);
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  try {
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await run(page, requests);
    assert.deepEqual(errors, [], 'the console must not throw uncaught errors');
  } finally { await page.close(); server.closeAllConnections(); await new Promise(r => server.close(r)); }
}

test('search keeps scope visible, sends supported filters, and reveals evidence on demand', async () => {
  await withPage(async (page, requests) => {
    await page.getByText('studio', { exact: true }).first().waitFor();
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('launch');
    await page.getByLabel('Tags', { exact: true }).fill('work');
    await page.getByLabel('As of (UTC)', { exact: true }).fill('2026-09-05');
    await page.locator('#controls').getByRole('button', { name: 'Search', exact: true }).click();
    await page.getByText(item.text, { exact: true }).waitFor();
    const last = requests.filter(r => r.url.pathname === '/v1/recall').at(-1).url;
    assert.equal(last.searchParams.get('tags'), 'work');
    assert.equal(last.searchParams.get('as_of'), '2026-09-05T00:00:00.000Z');
    assert.equal(last.searchParams.has('where'), false);
    const details = page.locator('details').filter({ hasText: 'Retrieval details' });
    assert.equal(await details.getAttribute('open'), null);
    await details.locator('summary').press('Enter');
    await details.getByText('0.720', { exact: false }).waitFor();
    assert.equal(await page.locator('#out script').count(), 0);
    assert.equal(await page.locator('nav [data-view="live"]').count(), 0);
    if (process.env.SCONE_SCREENSHOT_DIR) await page.screenshot({ path: resolve(process.env.SCONE_SCREENSHOT_DIR, 'rust-search-light.png') });
  });
});

test('closing a belief requires confirmation and a reason, and cancel changes nothing', async () => {
  await withPage(async (page, requests) => {
    await page.getByRole('button', { name: 'Beliefs', exact: true }).click();
    await page.getByRole('button', { name: 'Close belief', exact: true }).click();
    await page.getByRole('dialog').waitFor();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    assert.equal(requests.filter(r => r.method === 'POST').length, 0);
    assert.equal(await page.getByRole('button', { name: 'Close belief', exact: true }).evaluate(e => e === document.activeElement), true);
    await page.getByRole('button', { name: 'Close belief', exact: true }).click();
    await page.getByLabel('Reason', { exact: true }).fill('The launch environment changed.');
    await page.getByRole('dialog').getByRole('button', { name: 'Close belief', exact: true }).click();
    await page.getByText('closed', { exact: true }).waitFor();
    assert.deepEqual(requests.find(r => r.method === 'POST').body, { reason: 'The launch environment changed.' });
    assert.equal(requests.some(r => r.method === 'DELETE'), false);
  });
});

test('a slow search cannot overwrite a newer navigation choice', async () => {
  await withPage(async page => {
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('launch');
    await page.locator('#controls').getByRole('button', { name: 'Search', exact: true }).click();
    await page.getByRole('button', { name: 'Status', exact: true }).click();
    await page.getByText('24', { exact: true }).waitFor();
    await page.waitForTimeout(400); // Delayed response is the controlled race input.
    assert.equal(await page.getByText(item.text, { exact: true }).count(), 0);
    assert.equal(await page.getByRole('heading', { name: 'Status', exact: true }).count(), 1);
  }, { delay: true });
});

test('untrusted source schemes stay inert while excerpts remain readable', async () => {
  await withPage(async page => {
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('launch');
    await page.locator('#controls').getByRole('button', { name: 'Search', exact: true }).click();
    await page.getByText(item.text, { exact: true }).waitFor();
    assert.equal(await page.locator('a[href^="javascript:"]').count(), 0);
    assert.equal(await page.locator('a[href^="data:"]').count(), 0);
  }, { item: { source: 'javascript:alert(document.domain)' } });
});

test('mobile dark layout remains within the viewport and keyboard navigation works', async () => {
  await withPage(async page => {
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('launch');
    await page.getByRole('searchbox', { name: 'Search memory' }).press('Enter');
    await page.getByText(item.text, { exact: true }).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.getByRole('button', { name: 'Status', exact: true }).focus();
    await page.keyboard.press('Enter');
    await page.getByText('24', { exact: true }).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    if (process.env.SCONE_SCREENSHOT_DIR) await page.screenshot({ path: resolve(process.env.SCONE_SCREENSHOT_DIR, 'rust-status-mobile-dark.png') });
  }, { viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
});

test('unauthorized responses do not leave memory or a misleading connected state on screen', async () => {
  await withPage(async page => {
    await page.getByRole('alert').waitFor();
    assert.equal(await page.getByText(item.text, { exact: true }).count(), 0);
    assert.equal(await page.getByText('Connected', { exact: true }).count(), 0);
  }, { unauthorized: true });
});

test('clearing the query while a search is pending keeps the empty state', async () => {
  await withPage(async page => {
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('launch');
    await page.locator('#controls').getByRole('button', { name: 'Search', exact: true }).click();
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('');
    await page.getByRole('button', { name: 'Clear filters', exact: true }).click();
    await page.waitForTimeout(400);
    assert.equal(await page.getByText(item.text, { exact: true }).count(), 0);
    assert.equal(await page.locator('#out').getAttribute('aria-busy'), null);
  }, { delay: true });
});

test('a late status failure does not replace successful search results', async () => {
  await withPage(async page => {
    await page.getByRole('searchbox', { name: 'Search memory' }).fill('launch');
    await page.locator('#controls').getByRole('button', { name: 'Search', exact: true }).click();
    await page.getByText(item.text, { exact: true }).waitFor();
    await page.waitForTimeout(500);
    assert.equal(await page.getByText(item.text, { exact: true }).count(), 1);
  }, { statusDelay: 400, statusError: true });
});
