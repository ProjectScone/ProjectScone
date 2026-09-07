import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const layouts = {
  src: ['python/memory', 'python/memory/src/scone_memory'],
  legacy: ['python/scone-memory', 'python/scone-memory/scone_memory'],
};

function fixture(t, names) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'scone-package-layout-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  function write(relative, text) {
    const file = path.join(root, relative);
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, text);
  }
  write('Webapp/scripts/package.mjs', fs.readFileSync(path.join(repository, 'Webapp/scripts/package.mjs')));
  const helper = path.join(repository, 'scripts/python-layout.cjs');
  write('scripts/python-layout.cjs', fs.readFileSync(helper));
  write('Webapp/dist/index.html', '<html><script type="module" src="/assets/app.js"></script><link rel="stylesheet" href="/assets/app.css"></html>');
  write('Webapp/dist/assets/app.js', 'window.sconeFixture = true;');
  write('Webapp/dist/assets/app.css', 'body { color: olive; }');
  write('crates/scone/src/playground.html', 'previous Rust UI');
  for (const name of names) {
    const [project, source] = layouts[name];
    write(`${project}/pyproject.toml`, '[project]\nname = "scone-memory"\n');
    write(`${source}/api/__init__.py`, '');
    write(`${source}/api/console.html`, 'previous Python UI');
    write(`${source}/api/playground.html`, 'previous Python UI');
  }
  const run = (...args) => spawnSync(process.execPath, [path.join(root, 'Webapp/scripts/package.mjs'), ...args], {cwd: root, encoding: 'utf8'});
  const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');
  return {root, run, read, write};
}

for (const name of ['src', 'legacy']) {
  test(`asset packaging embeds matching native UI for the ${name} Python layout`, t => {
    const {run, read, write} = fixture(t, [name]);
    const [, source] = layouts[name];
    const result = run();
    assert.equal(result.status, 0, result.stderr);
    const html = read(`${source}/api/console.html`);
    assert.match(html, /window.sconeFixture = true;/);
    assert.match(html, /body \{ color: olive; \}/);
    assert.doesNotMatch(html, /(?:src|href)="\/assets\//);
    assert.equal(read(`${source}/api/playground.html`), html);
    assert.equal(read('crates/scone/src/playground.html'), html);
    assert.equal(run('--check').status, 0);
    write(`${source}/api/console.html`, 'drift');
    const check = run('--check');
    assert.equal(check.status, 1);
    assert.match(check.stderr, /Console build drift:/);
    assert.equal(read(`${source}/api/console.html`), 'drift', '--check never writes');
  });
}

for (const names of [[], ['src', 'legacy']]) {
  test(`asset packaging rejects ${names.length ? 'ambiguous' : 'missing'} Python projects before publishing anything`, t => {
    const {run, read} = fixture(t, names);
    const result = run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Python project/);
    assert.equal(read('crates/scone/src/playground.html'), 'previous Rust UI');
    for (const name of names) assert.equal(read(`${layouts[name][1]}/api/console.html`), 'previous Python UI');
  });
}

test('an incomplete src move fails before replacing either native UI', t => {
  const {run, read, write} = fixture(t, []);
  write('python/memory/pyproject.toml', '[project]\nname = "scone-memory"\n');
  const result = run();
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Python.*(?:source|package)/);
  assert.equal(read('crates/scone/src/playground.html'), 'previous Rust UI');
});

test('a retained old virtualenv is not mistaken for a second source project', t => {
  const {run, write} = fixture(t, ['src']);
  write('python/scone-memory/.venv/pyvenv.cfg', 'home = /old/python');
  const result = run();
  assert.equal(result.status, 0, result.stderr);
});

test('a standalone UI artifact can be staged without any Python source tree', t => {
  const {run, read, root} = fixture(t, []);
  const result = run('--output', path.join(root, 'preview.html'));
  assert.equal(result.status, 0, result.stderr);
  assert.match(read('preview.html'), /window.sconeFixture/);
  assert.equal(read('crates/scone/src/playground.html'), 'previous Rust UI');
});
