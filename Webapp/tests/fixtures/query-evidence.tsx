import {useEffect,useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {BrowserRouter,Route,Routes} from 'react-router-dom';
import {createApiClient,type ApiClient} from '../../src/api';
import {RecallPanel} from '../../src/playground/RecallPanel';
import {MemoryPage} from '../../src/memory/MemoryPage';
import {SourcePage} from '../../src/memory/SourcePage';
import {SourceNavigation} from '../../src/memory/SourceNavigation';
import {ConversationEvidence} from '../../src/conversations/ConversationEvidence';
import {ConversationSession} from '../../src/conversations/ConversationSession';
import {turnReceipt,type TurnResult} from '../../src/conversations/contracts';
import '../../src/workspace.css';
import '../../src/app.css';
import '../../src/conversations/conversations.css';
import '../../src/identity.css';
import '../../src/shell.css';

function ReplyFixture({api}:{api:ApiClient}){
  const [result,setResult]=useState<TurnResult>(),[selected,setSelected]=useState<number|null>(null),[attempt,setAttempt]=useState(0),[loaded,setLoaded]=useState(-1);
  useEffect(()=>{
    const controller=new AbortController();
    api.request('/v1/conversations/fixture/turns/fixture',{signal:controller.signal}).then(turnReceipt).then(receipt=>{if(!controller.signal.aborted){setResult(receipt.result);setLoaded(attempt);}});
    return()=>controller.abort();
  },[api,attempt]);
  return <><button type="button" onClick={()=>setAttempt(value=>value+1)}>Poll identical reply receipt</button><output style={{display:'block',fontSize:11}}>Receipt poll {loaded+1}</output><div style={{maxWidth:280,marginTop:20}}><ConversationEvidence api={api} result={result} selected={selected} onSelect={setSelected}/></div></>;
}
function Fixture(){
  const [key,setKey]=useState('fixture-a');
  const api=useMemo(()=>createApiClient(key,()=>{}),[key]);
  const mode=new URLSearchParams(location.search).get('mode');
  return <BrowserRouter><SourceNavigation api={api} enabled><main style={{padding:16,maxWidth:1120,margin:'0 auto',boxSizing:'border-box'}}>
    <button type="button" onClick={()=>setKey(previous=>previous==='fixture-a'?'fixture-b':'fixture-a')}>Switch fixture connection</button>
    <Routes><Route path="/memory/sources/:id" element={<SourcePage api={api} enabled/>}/><Route path="*" element={mode==='memory'?<MemoryPage api={api}/>:mode==='reply'?<ReplyFixture api={api}/>:mode==='chat'?<div className="conversation-page" style={{display:'block'}}><div className="conversation-workspace"><h1 style={{margin:'50px 0 35px'}}>Conversation fixture</h1><div className="conversation-layout"><ConversationSession api={api} sid="fixture" onSession={()=>{}} textConfigured voiceSupported={false} deletionSupported={false} cancellationSupported={false} paginationSupported streamingSupported={false} onRemoved={()=>{}}/></div></div></div>:<RecallPanel api={api} enabled onRecalled={()=>{}}/>}/></Routes>
  </main></SourceNavigation></BrowserRouter>;
}
createRoot(document.getElementById('root')!).render(<Fixture/>);
