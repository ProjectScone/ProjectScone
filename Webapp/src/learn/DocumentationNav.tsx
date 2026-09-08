import { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { conceptPages, documentationGroups } from './content';
import mark from '../assets/scone-mark-small.png';

export function DocumentationHeader() {
  const [query, setQuery] = useState('');
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const matches = terms.length ? conceptPages.filter(page => {
    const text = [page.label, page.title, page.intro, ...page.sections.map(section => section.title + ' ' + section.body)].join(' ').toLocaleLowerCase();
    return terms.every(term => text.includes(term));
  }) : [];
  return <header className="docs-header" aria-label="Scone documentation">
    <Link to="/learn" className="docs-brand"><img src={mark} width="28" height="28" alt="" /><strong>Scone</strong><span>Documentation</span></Link>
    <div className="docs-search" onKeyDown={event => { if (event.key === 'Escape') setQuery(''); }}>
      <span className="docs-search-icon" aria-hidden="true">⌕</span>
      <input type="search" aria-label="Search documentation" placeholder="Search documentation…" value={query} onChange={event => setQuery(event.target.value)} aria-controls={terms.length ? 'docs-results' : undefined} />
      {terms.length > 0 && <nav id="docs-results" className="docs-search-results" aria-label="Documentation search results">
        <div className="docs-search-caption">{matches.length} matching {matches.length === 1 ? 'topic' : 'topics'}<button onClick={() => setQuery('')} aria-label="Close documentation search">×</button></div>
        {matches.length ? matches.map(page => <Link key={page.id} to={page.path} onClick={() => setQuery('')}><strong>{page.label}<span aria-hidden="true">↗</span></strong><small>{page.intro}</small></Link>) : <p>No matching documentation. Try a feature or API name.</p>}
      </nav>}
    </div>
    <Link className="docs-workspace" to="/memory">Open workspace <span aria-hidden="true">↗</span></Link>
  </header>;
}

export function DocumentationNav() {
  const [open, setOpen] = useState(false);
  return <aside className="learn-sidebar">
    <button className="docs-browse" aria-expanded={open} aria-controls="docs-navigation" onClick={() => setOpen(value => !value)}>Browse documentation <span aria-hidden="true">{open ? '−' : '+'}</span></button>
    <nav id="docs-navigation" className={open ? 'docs-navigation is-open' : 'docs-navigation'} aria-label="Scone concepts">
      {documentationGroups.map(group => <div className="docs-nav-group" key={group.label}><h2>{group.label}</h2>{group.ids.map(id => {
        const item = conceptPages.find(page => page.id === id);
        return item ? <NavLink key={id} to={item.path} end onClick={() => setOpen(false)}>{item.label}</NavLink> : null;
      })}</div>)}
      <div className="learn-sidebar-note"><strong>Built for your stack.</strong><p>Native Rust and Python.<br />One connected workspace.</p><Link to="/learn/api">Explore the API <span aria-hidden="true">→</span></Link></div>
    </nav>
  </aside>;
}
