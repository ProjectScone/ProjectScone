import { test } from 'node:test';
import assert from 'node:assert/strict';
import { consoleAccessKey } from '../src/local-session.ts';

test('local development bootstrap accepts only the explicit native console token', () => {
  assert.equal(consoleAccessKey('<script id="scone-bootstrap" type="application/json">{"key":"local-react-test"}</script>'), 'local-react-test');
  assert.equal(consoleAccessKey('<script id="scone-bootstrap" type="application/json">{"key":"__SCONE_TOKEN__"}</script>'), '');
  assert.equal(consoleAccessKey('<script data-token="local-test">x()</script>'), 'local-test');
  assert.equal(consoleAccessKey('const TOKEN = document.currentScript.dataset.token || "rust-test";'), 'rust-test');
  assert.equal(consoleAccessKey('<p>key=unrelated-text</p>'), '');
  assert.equal(consoleAccessKey('const TOKEN = document.currentScript.dataset.token || "__SCONE_TOKEN__";'), '');
});
