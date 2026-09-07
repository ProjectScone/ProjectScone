// The native loopback console already supplies this key to the local browser.
// Read only its explicit bootstrap field; never scrape settings or transcripts.
export function consoleAccessKey(html: string): string {
  const bootstrap = html.match(/<script id="scone-bootstrap" type="application\/json">([^<]*)<\/script>/);
  if (bootstrap) {
    try {
      const key: unknown = JSON.parse(bootstrap[1]).key;
      return typeof key === 'string' && !key.startsWith('__SCONE_') ? key : '';
    } catch { return ''; }
  }
  const attribute = html.match(/<script\s+data-token="([^"<>]+)"/);
  if (attribute) return attribute[1];
  const literal = html.match(/const TOKEN = document\.currentScript\.dataset\.token \|\| ("(?:[^"\\]|\\.)*")/);
  if (!literal) return '';
  try {
    const value = JSON.parse(literal[1]);
    return value.startsWith('__SCONE_') ? '' : value;
  } catch { return ''; }
}
export async function localSession() {
  try {
    const response = await fetch('/__native_console', { cache: 'no-store', signal: AbortSignal.timeout(3000) });
    return response.ok ? consoleAccessKey(await response.text()) : '';
  } catch { return ''; }
}
