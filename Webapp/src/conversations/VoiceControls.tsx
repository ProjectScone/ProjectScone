import {useEffect,useRef,useState} from 'react';
import type {ApiClient} from '../api';
import type {ConversationSession} from './contracts';
import {startVoiceDevice,type VoiceDeviceState} from './voice/device';

/** A saved session is an inspection view until the person explicitly enables audio. */
export function VoiceControls({api,session,verified,supported,stopRequested}:{api:ApiClient;session:ConversationSession;verified:boolean;supported:boolean;stopRequested:boolean}){
  const [state,setState]=useState<VoiceDeviceState|'idle'>('idle'),[consent,setConsent]=useState(false);
  const device=useRef<ReturnType<typeof startVoiceDevice>|null>(null),controller=useRef<AbortController|null>(null);
  const mounted=useRef(false),attached=useRef(false);
  const terminal=['ended','failed','interrupted'].includes(session.state);
  useEffect(()=>{
    mounted.current=true;
    return()=>{mounted.current=false;controller.current?.abort();device.current?.stop();};
  },[api,session.session_id]);
  useEffect(()=>{
    if(terminal||session.state==='stopping'||!verified||!supported||stopRequested){controller.current?.abort();device.current?.stop();}
  },[terminal,session.state,verified,supported,stopRequested]);
  const active=['permission','connecting','listening','muted'].includes(state);
  const canJoin=supported&&verified&&session.state==='created'&&Boolean(session.persona?.fingerprint)
    &&session.persona?.current===true&&!attached.current&&!active&&!stopRequested;
  function start(){
    if(!canJoin||!consent)return;
    controller.current?.abort();
    const abort=new AbortController();controller.current=abort;
    device.current=startVoiceDevice((format,events,signal)=>{
      attached.current=true; // A sent hello is never automatically retried or reattached.
      return api.voiceConnection(session.session_id,format,events,signal);
    },next=>{if(mounted.current)setState(next);},abort.signal);
  }
  const label=state==='permission'?'Waiting for microphone permission':state==='connecting'?'Connecting audio':state==='listening'?'Microphone on':state==='muted'?'Microphone muted':state==='failed'?'Audio connection unavailable':'Microphone off in this tab';
  const description=terminal?'This voice session is closed. Its retained transcript and sources remain available.':active?
    state==='muted'?'Your microphone is muted. You can still hear the reply.':state==='listening'?'Live audio is connected to your selected persona. Speak naturally; interrupt to change direction.':'You can cancel at any time. Audio starts only after the server accepts this session.':
    attached.current?'Audio has stopped in this tab. Checking the recorded session outcome; this connection will not be resumed automatically.':
    state==='failed'?'Check your browser’s microphone permission and audio device, then enable it again. No audio connection was opened.':
    session.state==='running'?'The server reports this voice session as running. This tab is inspecting it, not connected to its audio.':
    !supported?'This server has not advertised a compatible browser voice transport. You can still inspect the retained transcript.':
    !session.persona?.fingerprint||session.persona.current!==true?'The recorded persona configuration cannot be verified. Start a new conversation with a current voice persona.':
    'This voice session is waiting for an audio connection. Opening a saved session never activates your microphone.';
  return <section className={`voice-controls${active?' is-active':''}`} aria-label="Voice session">
    <div className="voice-controls-status"><span className="voice-device-symbol" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M6 11v1a6 6 0 0 0 12 0v-1M12 18v3m-3 0h6"/></svg></span><div><span className="eyebrow">Live voice · {session.persona?.name??session.persona?.id??'Saved session'}</span><strong role="status">{label}</strong><p>{description}</p></div></div>
    {canJoin&&<label className="conversation-consent"><input type="checkbox" checked={consent} onChange={event=>setConsent(event.target.checked)}/>Allow live audio with this persona<span>Microphone audio goes to its configured speech providers. Public transcripts and replies are saved to this memory space.</span></label>}
    <div className="voice-controls-actions">
      {canJoin&&<button className="primary" disabled={!consent} onClick={start}>Enable microphone</button>}
      {state==='listening'||state==='muted'?<button onClick={()=>device.current?.mute(state!=='muted')}>{state==='muted'?'Unmute microphone':'Mute microphone'}</button>:null}
      {active&&<button onClick={()=>controller.current?.abort()}>{state==='permission'||state==='connecting'?'Cancel audio connection':'Stop audio'}</button>}
      <small>{active?'Leaving this view disconnects your microphone.':'No microphone access or automatic reconnection.'}</small>
    </div>
  </section>;
}
