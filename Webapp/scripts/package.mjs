// Bundle once, embed into both native products. Node is build-time only.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(app, '..');
const dist = path.join(app, 'dist');
let html = fs.readFileSync(path.join(dist, 'index.html'), 'utf8');
function asset(url) {
  const file = path.resolve(dist, '.' + url);
  if (!file.startsWith(dist + path.sep)) throw new Error('Invalid bundled asset path');
  return fs.readFileSync(file, 'utf8');
}
html = html.replace(/<script\b[^>]*src="([^"]+)"[^>]*><\/script>/g, (_, url) =>
  '<script type="module">' + asset(url).replaceAll('</script', '<\\/script') + '</script>');
html = html.replace(/<link\b[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g, (_, url) =>
  '<style>' + asset(url) + '</style>');
if (/<(?:script|link)[^>]+(?:src|href)="\/assets\//.test(html)) throw new Error('Unembedded build dependency');
const bytes = Buffer.from(html);
// Stage a standalone browser-test artifact without replacing a running server's UI.
const outputIndex = process.argv.indexOf('--output');
if (outputIndex !== -1) {
  const output = process.argv[outputIndex + 1];
  if (!output || output.startsWith('--') || process.argv.includes('--check')) throw new Error('--output requires a path and cannot be combined with --check');
  fs.writeFileSync(path.resolve(output), bytes);
  process.exit(0);
}
const targets = ['crates/scone/src/playground.html', 'python/scone-memory/scone_memory/api/playground.html', 'python/scone-memory/scone_memory/api/console.html'];
let drift = false;
for (const target of targets) {
  const file = path.join(repo, target);
  if (process.argv.includes('--check')) {
    if (!fs.existsSync(file) || !fs.readFileSync(file).equals(bytes)) {
      process.stderr.write('Console build drift: ' + target + '\n'); drift = true;
    }
  } else fs.writeFileSync(file, bytes);
}
if (drift) process.exitCode = 1;
