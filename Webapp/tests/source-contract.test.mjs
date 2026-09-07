import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';

test('the webapp has one React TypeScript entry and no legacy JavaScript application source', () => {
  const index = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(index, /src="\/src\/main\.tsx"/);
  assert.match(index, /id="root"/);
  const sources = readdirSync(new URL('../src', import.meta.url), { recursive: true });
  assert.deepEqual(sources.filter(file => /\.(js|jsx)$/.test(file)), []);
  assert.equal(sources.some(file => file.startsWith('rust-memory')), false);
});
