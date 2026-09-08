import {Link} from 'react-router-dom';
import type {Capabilities} from './contracts';

export type ServiceState='checking'|'ready'|'missing'|'incompatible'|'unavailable'|'unauthorized';

export function ConversationReadiness({enabled,state,cap,invalidAddress,onRetry}:{
  enabled:boolean;state:ServiceState;cap:Capabilities|null;invalidAddress:boolean;onRetry:()=>void;
}){
  const ready=enabled&&state==='ready'&&cap!==null;
  const canStart=ready&&(cap.text_configured||cap.voice);
  const title=!enabled?'Connect your memory first':invalidAddress?'Invalid session address':
    state==='checking'?'Checking conversation service':state==='missing'?'Connect a conversation service':
    state==='incompatible'?'Conversation service needs an update':state==='unauthorized'?'Reconnect your memory space':
    state==='unavailable'?'Connection check failed':!canStart?'Text runtime not configured':'Your knowledge, ready for a conversation.';
  const description=!enabled?'Use the memory connection control to authorize this workspace.':invalidAddress?'Choose a saved session or return to Conversations. This address is not a valid session ID.':
    state==='checking'?'Checking this server before loading your saved sessions. No conversation has started.':
    state==='missing'?'This server does not expose the conversation service. Memory and conversations need separate service configuration.':
    state==='incompatible'?'The service responded, but its conversation contract is not supported by this webapp. Session reads and new conversations are paused.':
    state==='unauthorized'?'The service rejected this connection. Use the memory connection control to authorize the correct space.':
    state==='unavailable'?'We could not check the conversation service. Your saved sessions have not been confirmed empty. Retry when the connection is available.':
    !canStart?'Saved conversations remain available. Configure a text model on this server to start or send messages. No provider key belongs in this page.':
    cap.voice?'Start a text or voice session with a configured persona. Choose the memory it can recall, and inspect the original sources alongside each reply. Your microphone stays off until you enable it.':'Start a text session or reopen a saved conversation. Choose the memory it can recall, then inspect the original sources alongside each reply.';
  const retry=enabled&&!ready&&state!=='checking';
  return <section className="conversation-readiness" aria-labelledby="conversation-readiness-title" aria-busy={enabled&&state==='checking'}>
    <span className="eyebrow">{canStart?'A connected workspace':'Workspace setup'}</span>
    <h2 id="conversation-readiness-title">{title}</h2>
    <p className="conversation-readiness-intro">{description}</p>
    <dl className="conversation-readiness-stages">
      <div data-available={ready}><dt><span aria-hidden="true">01</span>Conversation service</dt><dd>{!enabled?'Connect memory first':ready?'Connected':state==='checking'?'Checking…':state==='missing'?'Not available on this server':state==='incompatible'?'Incompatible response':'Connection not verified'}</dd></div>
      <div data-available={ready&&cap.text_configured}><dt><span aria-hidden="true">02</span>Text model</dt><dd>{ready?cap.text_configured?'Configured':'Needs configuration':'Not checked'}</dd></div>
      <div data-available={ready&&cap.voice}><dt><span aria-hidden="true">03</span>Browser voice</dt><dd>{ready&&cap.voice?'Available · microphone off':'Not connected'}</dd></div>
    </dl>
    <div className="conversation-readiness-actions">
      {retry&&<button className="primary" onClick={onRetry}>Retry service connection</button>}
      <Link to="/memory">Open memory <span aria-hidden="true">↗</span></Link>
      {invalidAddress&&<Link to="/conversations">Back to conversations</Link>}
    </div>
    <details className="conversation-service-setup">
      <summary>Service setup details</summary>
      <div>
        <h3>Connect the service to this workspace</h3>
        <p>The Python service can serve memory, conversations and this webapp on one origin. Set <code>SCONE_CONVERSATIONS_JOURNAL</code> to a separate persistent conversation database to compose conversations into <code>scone-memory serve</code>. Never use the memory database itself as the journal.</p>
        <pre><code>scone-memory serve-conversations --help</code></pre>
        <p>For a standalone conversation service, use <code>serve-conversations --journal</code> with <code>--history-only</code> for saved sessions or <code>--model-factory module:callable</code> for a trusted text adapter. Add <code>--console</code> for the webapp. Preserve the intended memory store and scoped keys; do not start two listeners on the same port.</p>
        <h3>Enable a voice persona</h3>
        <p>A text model alone does not enable spoken conversations. Configure <code>SCONE_CONVERSATIONS_PERSONAS</code> with your persona files and <code>SCONE_CONVERSATIONS_REGISTRY</code> with your trusted provider registry. A voice persona needs transcription, a reply model and speech synthesis; speech activity can come from its recognizer or a configured detector.</p>
        <p>The workspace checks the server’s voice protocol before offering Voice. Starting a session leaves the microphone off; you enable audio separately. Leaving that session releases the microphone and disconnects playback. Audio connections are never resumed automatically.</p>
        <p>Provider credentials stay on the server. This page never selects or calls a paid provider for you.</p>
      </div>
    </details>
    {canStart&&<p className="conversation-readiness-footnote">{cap.streaming?'Live public-text previews are supported.':'Replies appear when complete.'} Sources remain inspectable; a retrieved passage is not proof the model used it correctly.</p>}
  </section>;
}
