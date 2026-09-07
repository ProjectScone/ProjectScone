import Markdown, {type Components} from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './source-content.css';

// Source documents are data. Never resolve their relative URLs against the app,
// load their images automatically, or interpret embedded HTML as DOM.
function sourceUrl(value: string): string | undefined {
  if (!/^(https?:\/\/|mailto:)/i.test(value) || /[\u0000-\u0020\u007f]/.test(value)) return;
  try {
    const url = new URL(value);
    if (url.username || url.password) return;
    return url.href;
  } catch { return; }
}

const components: Components = {
  a: ({href, children}) => href ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}</span>,
  img: ({alt}) => <span className="source-image-reference">Image reference{alt ? `: ${alt}` : ''} · not loaded</span>,
  h1: ({children}) => <h3>{children}</h3>,
  h2: ({children}) => <h3>{children}</h3>,
  table: ({children}) => <div className="source-table"><table>{children}</table></div>,
};
const inlineElements = ['strong', 'em', 'code', 'del', 'br', 'img'];
const FORMAT_LIMIT = 80000;

export function MarkdownText({text, inline = false}: {text: string; inline?: boolean}) {
  const content = <Markdown remarkPlugins={[remarkGfm]} components={components}
    urlTransform={url => sourceUrl(url)} allowedElements={inline ? inlineElements : undefined}
    unwrapDisallowed={inline}>{text.slice(0, FORMAT_LIMIT)}</Markdown>;
  return inline ? <span className="source-markdown-inline">{content}</span>
    : <div className="source-markdown">{content}</div>;
}

/** Presentation only. The original string remains available without normalization. */
export function SourceContent({text}: {text: string}) {
  return <div className="formatted-source">
    <MarkdownText text={text}/>
    {text.length > FORMAT_LIMIT && <p className="source-format-limit">Formatted preview is limited to {FORMAT_LIMIT.toLocaleString()} characters. The complete stored text is below.</p>}
    <details className="source-original"><summary>View original Markdown</summary>
      <pre tabIndex={0} aria-label="Original source text">{text}</pre>
    </details>
  </div>;
}
