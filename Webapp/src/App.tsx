import { useCallback, useEffect, useMemo, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { createApiClient } from './api';
import type { Status } from './types';
import { PlaygroundPage } from './playground/PlaygroundPage';
import { MemoryPage } from './memory/MemoryPage';
import { ConversationPage } from './conversations/ConversationPage';
import { Modal } from './components/Modal';
import { LearnPage } from './learn/LearnPage';
import { conceptPages } from './learn/content';
import { AppFrame } from './components/AppFrame';

export function App({ initialKey }: { initialKey: string }) {
  const [key, setKey] = useState(initialKey.startsWith('__SCONE_') ? '' : initialKey);
  const [epoch, setEpoch] = useState(0), [auth, setAuth] = useState(!key && !conceptPages.some(page=>page.path===location.pathname.replace(/\/$/,''))), [input, setInput] = useState('');
  const [space, setSpace] = useState(key ? 'Connecting…' : 'Not connected');
  const [connected, setConnected] = useState(false);
  const unauthorized = useCallback(() => { setKey(''); setEpoch(value => value + 1); setSpace('Not connected'); setConnected(false); }, []);
  const api = useMemo(() => createApiClient(key, unauthorized), [key, unauthorized]);
  useEffect(() => {
    if (!key) { setSpace('Not connected'); setConnected(false); return; }
    setSpace('Connecting…');
    setConnected(false);
    const controller = new AbortController();
    api.request<Status>('/v1/status', { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(8000)]) }).then(status => { if (!controller.signal.aborted) { setSpace(status.space || status.name || 'Authenticated space'); setConnected(true); } }).catch(() => { if (!controller.signal.aborted) setSpace('Connection failed'); });
    return () => controller.abort();
  }, [api, key]);
  return <BrowserRouter>
    <AppFrame space={space} connected={connected} checking={Boolean(key)&&space==='Connecting…'} onConnect={()=>setAuth(true)}>
    <Routes>
      <Route path="/" element={<Navigate to="/memory" replace />} />
      <Route path="/memory" element={key ? <MemoryPage key={epoch} api={api} /> : <main id="main" className="connection-empty"><div className="eyebrow">Your memory workspace</div><h1>Connect to Scone</h1><p>This app reads memory from <code>{location.origin}</code>. No memory is available until this server accepts your space key.</p><button className="primary" onClick={() => setAuth(true)}>Set up memory connection</button><p className="muted">The local single-key preview connects automatically after reload. For a multi-space server, its administrator supplies the Scone space key.</p></main>} />
      <Route path="/playground" element={<PlaygroundPage key={epoch} api={api} enabled={Boolean(key)} />} />
      <Route path="/conversations/:sid?" element={<ConversationPage key={epoch} api={api} enabled={Boolean(key)} />} />
      {conceptPages.map(page=><Route key={page.id} path={page.path} element={<LearnPage key={page.id} page={page}/>}/>)}
      <Route path="*" element={<Navigate to="/memory" replace />} />
    </Routes>
    </AppFrame>
    {auth && <Modal title="Memory connection" onClose={() => setAuth(false)}>
      <p className="setup-intro">This webapp talks to Scone’s API. Scone stores and retrieves memory from its configured database.</p>
      <dl className="connection-details"><div><dt>Server</dt><dd><code>{location.origin}</code></dd></div><div><dt>Memory space</dt><dd>{connected ? space : 'Determined by your Scone space key'}</dd></div><div><dt>Access</dt><dd>{connected ? 'Verified by the memory API' : 'Not authenticated'}</dd></div></dl>
      <p className="setup-intro">Use a <strong>Scone space key</strong> from this server’s configuration or administrator—not a Claude, Codex or OpenAI API key. A single-key local preview supplies it automatically.</p>
      <form onSubmit={event => { event.preventDefault(); const next = input.trim(); if (!next) return; setKey(next); setInput(''); setEpoch(value => value + 1); setAuth(false); }}><label htmlFor="space-key">{connected ? 'Use a different Scone space key' : 'Scone space key'}</label><input id="space-key" className="key-input" type="password" autoComplete="off" required value={input} onChange={event => setInput(event.target.value)} /><button className="primary">{connected ? 'Switch space' : 'Connect'}</button></form>
      <p className="setup-note">The key stays in page memory, never browser storage. Connecting memory does not automatically connect Claude Code or Codex activity.</p>
    </Modal>}
  </BrowserRouter>;
}
