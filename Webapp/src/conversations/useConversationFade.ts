import {useEffect,useSyncExternalStore} from 'react';
import {useAnimate} from 'motion/react-mini';

const reducedMotionQuery='(prefers-reduced-motion: reduce)';
const subscribe=(notify:()=>void)=>{
  const query=window.matchMedia(reducedMotionQuery);
  query.addEventListener('change',notify);
  return()=>query.removeEventListener('change',notify);
};
const prefersReducedMotion=()=>window.matchMedia(reducedMotionQuery).matches;

/** Fade the current content only; never retain an outgoing source or alter layout. */
export function useConversationFade<T extends HTMLElement>(state:string){
  const reduced=useSyncExternalStore(subscribe,prefersReducedMotion,()=>true);
  const [scope,animate]=useAnimate<T>();
  useEffect(()=>{
    const element=scope.current;
    if(!element)return;
    if(reduced){element.style.opacity='1';return;}
    const animation=animate(element,{opacity:[0.72,1]},{duration:0.16,ease:'easeOut'});
    return()=>{animation.stop();element.style.opacity='1';};
  },[animate,reduced,scope,state]);
  return scope;
}
