import { createRoot } from 'react-dom/client';
import { App } from './App';
import { consoleAccessKey, localSession } from './local-session';
import './workspace.css';
import './identity.css';
import './app.css';
import './playground/graph.css';

const initialKey = import.meta.env.DEV ? await localSession() : consoleAccessKey(document.documentElement.outerHTML);
createRoot(document.getElementById('root')!).render(<App initialKey={initialKey} />);
