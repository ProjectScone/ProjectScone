import {createContext,useContext,useEffect,useState,type ReactNode} from 'react';
import type {ApiClient} from '../api';
import {parseCapabilities} from '../capabilities';
import {verifiedSpace} from './source-address';

type Discovery={state:'ready';space:string}|{state:'pending'|'unsupported'|'failed'|'disconnected'};
interface Navigation {api:ApiClient;discovery:Discovery;retry:()=>void}
const SourceContext=createContext<Navigation|null>(null);
export const useSourceNavigation=()=>useContext(SourceContext);

/** One short-lived connection check for the mounted workspace, not one per card. */
export function SourceNavigation({api,enabled,children}:{api:ApiClient;enabled:boolean;children:ReactNode}){
  const [attempt,setAttempt]=useState(0);
  const [snapshot,setSnapshot]=useState<{api:ApiClient;attempt:number;discovery:Discovery}|null>(null);
  useEffect(()=>{
    if(!enabled)return;
    const controller=new AbortController();
    const signal=AbortSignal.any([controller.signal,AbortSignal.timeout(10000)]);
    const save=(discovery:Discovery)=>{if(!controller.signal.aborted)setSnapshot({api,attempt,discovery});};
    void Promise.all([
      api.request<unknown>('/v1/status',{cache:'no-store',signal}).then(verifiedSpace),
      api.request<unknown>('/v1/capabilities',{cache:'no-store',signal}).then(parseCapabilities),
    ]).then(([space,caps])=>save(caps.features['episodes.read']?{state:'ready',space}:{state:'unsupported'}))
      .catch(()=>save({state:'failed'}));
    return()=>controller.abort();
  },[api,enabled,attempt]);
  const discovery:Discovery=!enabled?{state:'disconnected'}:snapshot?.api===api&&snapshot.attempt===attempt?snapshot.discovery:{state:'pending'};
  return <SourceContext.Provider value={{api,discovery,retry:()=>setAttempt(value=>value+1)}}>{children}</SourceContext.Provider>;
}
