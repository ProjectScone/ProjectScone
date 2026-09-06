import { useCallback, useEffect, useMemo, useState } from 'react';
import { BrowserRouter, Navigate, NavLink, Route, Routes } from 'react-router-dom';
import { createApiClient } from './api';
import type { Status } from './types';
import { PlaygroundPage } from './playground/PlaygroundPage';
import { MemoryPage } from './memory/MemoryPage';
import { ConversationPage } from './conversations/ConversationPage';
import { Modal } from './components/Modal';
import mark from './assets/scone-mark-small.png';

export function App({ initialKey }: { initialKey: string }) {
  const [key, setKey] = useState(initialKey.startsWith('__SCONE_') ? '' : initialKey);
  const [epoch, setEpoch] = useState(0), [auth, setAuth] = useState(!key), [input, setInput] = useState('');
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
    <a className="skip" href="#main">Skip to workspace</a>
    <header className="topbar"><NavLink className="brand" to="/memory" aria-label="Scone console"><img className="brand-icon" src={mark} alt="" width={32} height={32} />Scone</NavLink><nav aria-label="Workspace"><NavLink to="/memory">Memory</NavLink><NavLink to="/playground">Playground</NavLink><NavLink to="/conversations">Conversations</NavLink></nav><span className="scope"><span className="eyebrow">Space</span> <span id="space">{space}</span></span><button className="subtle connection-button" onClick={() => setAuth(true)}>{connected ? 'Memory connection' : 'Connect memory'}</button></header>
    <div className="server-strip" role="status"><span className={connected ? 'server-state verified' : 'server-state'}>{connected ? 'Authenticated' : key ? 'Checking connection' : 'Not connected'}</span><span>API <code>{location.host}</code></span>{connected && <span>Space <b>{space}</b></span>}<span className="server-caption">{connected ? 'Memory access verified · agent capture is separate' : 'A Scone space key grants access to this server’s memory'}</span></div>
    <Routes>
      <Route path="/" element={<Navigate to="/memory" replace />} />
      <Route path="/memory" element={key ? <MemoryPage key={epoch} api={api} /> : <main id="main" className="connection-empty"><div className="eyebrow">Your memory workspace</div><h1>Connect to Scone</h1><p>This app reads memory from <code>{location.origin}</code>. No memory is available until this server accepts your space key.</p><button className="primary" onClick={() => setAuth(true)}>Set up memory connection</button><p className="muted">The local single-key preview connects automatically after reload. For a multi-space server, its administrator supplies the Scone space key.</p></main>} />
      <Route path="/playground" element={<PlaygroundPage key={epoch} api={api} enabled={Boolean(key)} />} />
      <Route path="/conversations/:sid?" element={<ConversationPage key={epoch} api={api} enabled={Boolean(key)} />} />
      <Route path="*" element={<Navigate to="/memory" replace />} />
    </Routes>
    {auth && <Modal title="Memory connection" onClose={() => setAuth(false)}>
      <p className="setup-intro">This webapp talks to Scone’s API. Scone stores and retrieves memory from its configured database.</p>
      <dl className="connection-details"><div><dt>Server</dt><dd><code>{location.origin}</code></dd></div><div><dt>Memory space</dt><dd>{connected ? space : 'Determined by your Scone space key'}</dd></div><div><dt>Access</dt><dd>{connected ? 'Verified by the memory API' : 'Not authenticated'}</dd></div></dl>
      <p className="setup-intro">Use a <strong>Scone space key</strong> from this server’s configuration or administrator—not a Claude, Codex or OpenAI API key. A single-key local preview supplies it automatically.</p>
      <form onSubmit={event => { event.preventDefault(); const next = input.trim(); if (!next) return; setKey(next); setInput(''); setEpoch(value => value + 1); setAuth(false); }}><label htmlFor="space-key">{connected ? 'Use a different Scone space key' : 'Scone space key'}</label><input id="space-key" className="key-input" type="password" autoComplete="off" required value={input} onChange={event => setInput(event.target.value)} /><button className="primary">{connected ? 'Switch space' : 'Connect'}</button></form>
      <p className="setup-note">The key stays in page memory, never browser storage. Connecting memory does not automatically connect Claude Code or Codex activity.</p>
    </Modal>}
  </BrowserRouter>;
}
