import type {ReactNode} from 'react';
import {NavLink} from 'react-router-dom';
import {WorkspaceIcon, type WorkspaceIconName} from './WorkspaceIcon';
import mark from '../assets/scone-mark-small.png';

const destinations: {path:string; label:string; icon:WorkspaceIconName}[] = [
  {path:'/memory',label:'Memory',icon:'memory'},
  {path:'/playground',label:'Playground',icon:'graph'},
  {path:'/conversations',label:'Conversations',icon:'chat'},
  {path:'/learn',label:'Guide',icon:'guide'},
];

/** Persistent product frame; route components own their tools, not navigation. */
export function AppFrame({children,space,connected,checking,onConnect}:{
  children:ReactNode; space:string; connected:boolean; checking:boolean; onConnect:()=>void;
}) {
  return <div className="app-shell">
    <a className="skip" href="#main" onClick={event=>{
      const main=document.getElementById('main');
      if(!main)return;
      event.preventDefault();
      if(!main.hasAttribute('tabindex'))main.tabIndex=-1;
      main.focus({preventScroll:true});
      main.scrollIntoView({block:'start'});
    }}>Skip to workspace</a>
    <header className="topbar">
      <NavLink className="brand" to="/memory" aria-label="Scone console">
        <img className="brand-icon" src={mark} alt="" width={32} height={32}/><span>Scone</span>
      </NavLink>
      <nav aria-label="Workspace">
        {destinations.map(item=><NavLink key={item.path} to={item.path}><WorkspaceIcon name={item.icon}/><span>{item.label}</span></NavLink>)}
      </nav>
    </header>
    <div className="app-pane">
      <div className="server-strip">
        <div className="space-identity"><span className="space-monogram" aria-hidden="true"><WorkspaceIcon name="scopes"/></span><div><span className="eyebrow">Memory space</span><span id="space" title={space}>{space}</span></div></div>
        <div className="server-summary" role="status"><span className={connected?'server-state verified':'server-state'}>{connected?'Authenticated':checking?'Checking connection':space==='Connection failed'?'Connection failed':'Not connected'}</span><span className="server-endpoint">API <code>{location.host}</code></span></div>
        <span className="server-caption">{connected?'Memory access verified · agent capture is separate':'Connect a space to access your memory'}</span>
        <button className="subtle connection-button" onClick={onConnect}>{connected?'Memory connection':'Connect memory'}</button>
      </div>
      {children}
    </div>
  </div>;
}
