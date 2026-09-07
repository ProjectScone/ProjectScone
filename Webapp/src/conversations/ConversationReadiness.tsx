import {Link} from 'react-router-dom';
import type {Capabilities} from './contracts';

export type ServiceState='checking'|'ready'|'missing'|'incompatible'|'unavailable'|'unauthorized';

export function ConversationReadiness({enabled,state,cap,invalidAddress,onRetry}:{
  enabled:boolean;state:ServiceState;cap:Capabilities|null;invalidAddress:boolean;onRetry:()=>void;
}){
  const ready=enabled&&state==='ready'&&cap!==null;
  const canStart=ready&&cap.text_configured;
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
    'Start a text session or reopen a saved conversation. Choose the memory it can recall, then inspect the original sources alongside each reply.';
  const retry=enabled&&!ready&&state!=='checking';
  return <section className="conversation-readiness" aria-labelledby="conversation-readiness-title" aria-busy={enabled&&state==='checking'}>
    <span className="eyebrow">{canStart?'A connected workspace':'Workspace setup'}</span>
    <h2 id="conversation-readiness-title">{title}</h2>
    <p className="conversation-readiness-intro">{description}</p>
    <dl className="conversation-readiness-stages">
      <div data-available={ready}><dt><span aria-hidden="true">01</span>Conversation service</dt><dd>{!enabled?'Connect memory first':ready?'Connected':state==='checking'?'Checking…':state==='missing'?'Not available on this server':state==='incompatible'?'Incompatible response':'Connection not verified'}</dd></div>
      <div data-available={canStart}><dt><span aria-hidden="true">02</span>Text model</dt><dd>{ready?canStart?'Configured':'Needs configuration':'Not checked'}</dd></div>
      <div data-available="false"><dt><span aria-hidden="true">03</span>Browser voice</dt><dd>Not connected</dd></div>
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
        <p>The memory-only <code>serve</code> command does not enable conversations. The Python conversation service can serve the webapp and memory routes together. An operator must configure it on the origin you open.</p>
        <pre><code>scone-memory serve-conversations --help</code></pre>
        <p>Use the intended memory-store configuration and scoped API keys, plus a separate persistent conversation journal. Select <code>--history-only</code> for saved sessions, or <code>--model-factory module:callable</code> for a trusted text adapter. Add <code>--console</code> to serve this webapp. Do not start two listeners on the same port or use the memory database as the journal.</p>
        <h3>What voice still needs</h3>
        <p>Browser audio transport and provider wiring are not connected in this workspace. A text model alone does not enable a microphone or spoken replies. Voice needs transcription, a reply model, speech synthesis and activity detection, with device consent and working playback.</p>
        <p>Provider credentials stay on the server. This page never selects or calls a paid provider for you.</p>
      </div>
    </details>
    {canStart&&<p className="conversation-readiness-footnote">{cap.streaming?'Live public-text previews are supported.':'Replies appear when complete.'} Sources remain inspectable; a retrieved passage is not proof the model used it correctly.</p>}
  </section>;
}
