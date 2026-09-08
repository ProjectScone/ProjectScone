import {useEffect, useRef, useState} from 'react';
import {Link, NavLink} from 'react-router-dom';
import Markdown from 'react-markdown';
import {WorkspaceHeading} from '../components/WorkspaceHeading';
import {conceptMarkdown, conceptPages, sourceExample, type ConceptPage} from './content';
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
  useEffect(()=>{
    const before=document.title;document.title=`${page.label} · Scone`;
    const section=page.sections.find(item=>`#${item.id}`===location.hash);
    if(section) document.getElementById(section.id)?.scrollIntoView({block:'start'});
    else {window.scrollTo(0,0);title.current?.focus({preventScroll:true});}
    return ()=>{document.title=before;};
  },[page]);
  return <main id="main" className="learn-shell">
    <aside className="learn-sidebar">
      <Link to="/learn" className="learn-wordmark">The Scone guide <span aria-hidden="true">↗</span></Link>
      <p>Understand the system.<br/>Put it to work.</p>
      <nav aria-label="Scone concepts">{conceptPages.map(item=><NavLink key={item.id} to={item.path} end><span aria-hidden="true">{String(conceptPages.indexOf(item)+1).padStart(2,'0')}</span>{item.label}</NavLink>)}</nav>
      <div className="learn-sidebar-note">No connection needed to read.<br/>Your workspace stays private.</div>
    </aside>
    <article className="learn-article">
      <WorkspaceHeading className="learn-heading" eyebrow={`Concepts / ${page.label}`} title={page.title} description={page.intro} titleRef={title} actions={<CopyButton text={conceptMarkdown(page)} label="Copy page" fallback="Copy unavailable. Select the page text instead."/>}/>
      {page.id==='graph-memory'?<RelationshipMap/>:<ContextMap/>}
      <nav className="learn-contents" aria-label="On this page"><span className="eyebrow">On this page</span>{page.sections.map(section=><a key={section.id} href={`#${section.id}`}>{section.title}</a>)}</nav>
      {page.sections.map(section=><section className="learn-section" key={section.id} id={section.id}>
        <h2>{section.title}</h2><Markdown>{section.body}</Markdown>
        {page.id==='how-it-works'&&section.id==='source'&&<section className="learn-example" aria-label="Source and recall example">
          <header><span>HTTP · Source and recall</span><CopyButton label="Copy example" text={sourceExample} fallback="Select and copy the text below."/></header>
          <pre tabIndex={0}><code>{sourceExample}</code></pre>
          <p>Inspect the returned episode_id, then the recall items. Claims require separate consolidation; this example does not run an extraction model.</p>
        </section>}
      </section>)}
      <footer className="learn-next"><div className="eyebrow">From understanding to action</div><h2>Try it in your workspace.</h2><div>{page.actions.map(action=><Link key={action.to} to={action.to}>{action.label}<span aria-hidden="true">↗</span></Link>)}</div><p>These links open the real workspace. They do not create example data or approve anything.</p></footer>
    </article>
  </main>;
}
