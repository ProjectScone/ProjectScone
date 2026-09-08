import {useEffect, useRef, useState} from 'react';
import {Link} from 'react-router-dom';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {WorkspaceHeading} from '../components/WorkspaceHeading';
import {conceptMarkdown, conceptPages, documentationGroups, sourceExample, type ConceptPage} from './content';
import {DocumentationHeader, DocumentationNav} from './DocumentationNav';
import './learn.css';

function CopyButton({text, label, fallback}: {text: string; label: string; fallback: string}) {
  const [result, setResult] = useState('');
  return <div className="learn-copy"><button onClick={async()=>{
    try {await navigator.clipboard.writeText(text);setResult('Copied.');}
    catch {setResult(fallback);}
  }}>{label}</button><span role="status">{result}</span></div>;
}

function ContextMap() {
  return <figure className="learn-map" aria-label="Source, retrieval and review paths">
    <figcaption>ONE ORIGINAL. TWO PATHS TO CONTEXT.</figcaption>
    <div className="learn-map-root"><span>01 / Original</span><strong>A retained source</strong><small>Episode ID · original content</small></div>
    <div className="learn-map-branches">
      <div><span>02 / Retrieval</span><strong>Chunks → passages</strong><small>Find and open relevant material</small></div>
      <div><span>03 / Consolidation</span><strong>Proposals → review</strong><small>Inspect claims before accepting them</small></div>
    </div>
    <p>Retrieving a source does not approve a claim.</p>
  </figure>;
}

function RelationshipMap() {
  return <figure className="learn-map learn-relations" aria-label="Illustrative memory relationships, not live records">
    <figcaption>ILLUSTRATIVE RELATIONSHIPS · NOT YOUR RECORDS</figcaption>
    <div><span>Mira works at Cedar</span><b>Updates →</b><strong>Mira now works at Rowan</strong></div>
    <div><span>Mira works at Rowan</span><b>Extends →</b><strong>Mira leads payments</strong></div>
    <div><span>Named supporting claims</span><b>Derives →</b><strong>A proposal to review</strong></div>
    <p>A relationship needs stored evidence—not proximity on a canvas.</p>
  </figure>;
}

export function LearnPage({page}: {page: ConceptPage}) {
  const title=useRef<HTMLHeadingElement>(null);
  const ordered= documentationGroups.flatMap(group=>group.ids).map(id=>conceptPages.find(item=>item.id===id)).filter(item=>item!==undefined);
  const index=ordered.findIndex(item=>item.id===page.id);
  const group=documentationGroups.find(item=>item.ids.some(id=>id===page.id));
  useEffect(()=>{
    const before=document.title;document.title=`${page.label} · Scone`;
    const section=page.sections.find(item=>`#${item.id}`===location.hash);
    if(section) document.getElementById(section.id)?.scrollIntoView({block:'start'});
    else {window.scrollTo(0,0);title.current?.focus({preventScroll:true});}
    return ()=>{document.title=before;};
  },[page]);
  return <div className="documentation">
    <a className="skip" href="#main">Skip to article</a>
    <DocumentationHeader/>
    <div className="learn-shell">
    <DocumentationNav/>
    <main id="main" className="learn-article">
      <WorkspaceHeading className="learn-heading" eyebrow={group?.label??'Documentation'} title={page.title} description={page.intro} titleRef={title} actions={<CopyButton text={conceptMarkdown(page)} label="Copy page" fallback="Copy unavailable. Select the page text instead."/>}/>
      {page.id==='overview'&&<div className="docs-start">
        <Link className="docs-start-primary" to="/learn/quickstart"><span className="eyebrow">START BUILDING</span><h2>Your first memory <span aria-hidden="true">→</span></h2><p>Save a source. Retrieve it. Open the original.</p><span className="docs-start-link">Follow the quickstart <span aria-hidden="true">↗</span></span></Link>
        <Link to="/learn/graph-memory"><span className="eyebrow">UNDERSTAND THE SYSTEM</span><h2>Connected knowledge <span aria-hidden="true">→</span></h2><p>Sources, claims and the relationships between them.</p><span className="docs-start-link">Explore graph memory <span aria-hidden="true">↗</span></span></Link>
      </div>}
      {page.id==='how-it-works'&&<ContextMap/>}
      {page.id==='graph-memory'&&<RelationshipMap/>}
      {page.sections.map(section=><section className="learn-section" key={section.id} id={section.id}>
        <h2><a href={`#${section.id}`}>{section.title}<span aria-hidden="true"> #</span></a></h2><Markdown remarkPlugins={[remarkGfm]} components={{table:({children})=><div className="docs-table-scroll" tabIndex={0}><table>{children}</table></div>}}>{section.body}</Markdown>
        {(page.id==='how-it-works'||page.id==='quickstart')&&section.id==='source'&&<section className="learn-example" aria-label="Source and recall example">
          <header><span>HTTP · Source and recall</span><CopyButton label="Copy example" text={sourceExample} fallback="Select and copy the text below."/></header>
          <pre tabIndex={0}><code>{sourceExample}</code></pre>
          <p>Inspect the returned episode_id, then the recall items. Claims require separate consolidation; this example does not run an extraction model.</p>
        </section>}
      </section>)}
      <footer className="learn-next"><div className="eyebrow">From understanding to action</div><h2>Try it in your workspace.</h2><div>{page.actions.map(action=><Link key={action.to} to={action.to}>{action.label}<span aria-hidden="true">↗</span></Link>)}</div><p>These links open the real workspace. They do not create example data or approve anything.</p></footer>
      <nav className="docs-pagination" aria-label="Continue reading">{[ordered[index-1],ordered[index+1]].map((item,position)=>item?<Link key={item.id} to={item.path}><small>{position===0?'← Previous':'Next →'}</small><strong>{item.label}</strong></Link>:<span key={position}/>)}</nav>
    </main>
    <aside className="docs-outline"><nav className="learn-contents" aria-label="On this page"><span>On this page</span>{page.sections.map(section=><a key={section.id} href={`#${section.id}`}>{section.title}</a>)}</nav><div className="docs-outline-note">Reading is public.<br/>Your memory stays private.</div></aside>
    </div>
  </div>;
}
