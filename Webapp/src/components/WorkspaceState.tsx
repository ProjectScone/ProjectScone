import {useId, type ReactNode} from 'react';
import {WorkspaceIcon, type WorkspaceIconName} from './WorkspaceIcon';

/** The same quiet, actionable empty state across product workspaces. */
export function WorkspaceState({icon,title,description,actions,className='',role,busy}:{
  icon:WorkspaceIconName;title:string;description:ReactNode;actions?:ReactNode;className?:string;role?:'alert'|'status';busy?:boolean;
}) {
  const heading=useId();
  return <section className={`workspace-state ${className}`} aria-labelledby={heading} role={role} aria-busy={busy}>
    <span className="workspace-state-icon"><WorkspaceIcon name={icon}/></span>
    <h2 id={heading}>{title}</h2>
    <div className="workspace-state-description">{description}</div>
    {actions&&<div className="workspace-state-actions">{actions}</div>}
  </section>;
}
