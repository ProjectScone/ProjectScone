/** Small, shared line icons. Decorative: the adjacent label names the control. */
export type WorkspaceIconName = 'memory' | 'graph' | 'chat' | 'guide' | 'search' | 'documents' | 'profile' | 'beliefs' | 'review' | 'live' | 'analytics' | 'scopes' | 'status';
const paths: Record<WorkspaceIconName,string> = {
  memory:'M12 3 3 8l9 5 9-5-9-5ZM3 12l9 5 9-5M3 16l9 5 9-5',
  profile:'M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0ZM4 21v-2a8 8 0 0 1 16 0v2',
  graph:'M6 7h.01M18 6h.01M12 18h.01M8 7l8-1M7 9l4 7M17 8l-4 8M8 7a2 2 0 1 1-4 0 2 2 0 0 1 4 0ZM20 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0ZM14 18a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z',
  chat:'M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2ZM7 9h10M7 13h6',
  guide:'M12 5v15M3 4h5a4 4 0 0 1 4 3 4 4 0 0 1 4-3h5v15h-5a4 4 0 0 0-4 2 4 4 0 0 0-4-2H3V4Z',
  search:'M17 17l4 4M19 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z',
  documents:'M14 3H5v18h14V8l-5-5ZM14 3v5h5M8 12h8M8 16h6',
  beliefs:'M12 3 3 8v8l9 5 9-5V8l-9-5ZM3 8l9 5 9-5M12 13v8',
  review:'M9 4H5v17h14V4h-4M9 3h6v4H9V3ZM8 13l3 3 5-6',
  live:'M2 12h5l3-8 4 16 3-8h5',
  analytics:'M4 3v18h17M8 17v-5M13 17V7M18 17v-8',
  scopes:'M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M8 8h8v8H8V8Z',
  status:'M12 8v5M12 16h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
};
export function WorkspaceIcon({name}:{name:WorkspaceIconName}) {
  return <svg className="workspace-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d={paths[name]}/></svg>;
}
