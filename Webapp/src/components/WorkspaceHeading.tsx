import type {ReactNode, Ref} from 'react';

/** One title and action hierarchy for every product workspace. */
export function WorkspaceHeading({eyebrow,title,description,actions,className='',titleRef}: {
  eyebrow:string;
  title:string;
  description:string;
  actions?:ReactNode;
  className?:string;
  titleRef?:Ref<HTMLHeadingElement>;
}) {
  return <header className={`workspace-heading ${className}`}>
    <div className="workspace-heading-copy">
      <span className="eyebrow">{eyebrow}</span>
      <h1 ref={titleRef} tabIndex={titleRef?-1:undefined}>{title}</h1>
      <p>{description}</p>
    </div>
    {actions&&<div className="workspace-heading-actions">{actions}</div>}
  </header>;
}
