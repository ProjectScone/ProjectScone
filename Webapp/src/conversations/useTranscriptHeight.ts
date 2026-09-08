import {useLayoutEffect,useRef} from 'react';

/** Reserve the measured controls, then give the remaining viewport to messages. */
export function useTranscriptHeight(){
  const ref=useRef<HTMLElement>(null);
  useLayoutEffect(()=>{
    const thread=ref.current;
    if(!thread)return;
    const workspace=thread.closest('.conversation-workspace');
    let frame=0;
    const measure=()=>{
      const messages=thread.querySelector<HTMLElement>('.conversation-messages');
      if(!messages)return;
      const occupied=Array.from(thread.children).reduce((height,child)=>{
        if(child===messages)return height;
        const style=getComputedStyle(child);
        if(style.display==='none')return height;
        return height+child.getBoundingClientRect().height+parseFloat(style.marginTop)+parseFloat(style.marginBottom);
      },0);
      const bottom=workspace?parseFloat(getComputedStyle(workspace).paddingBottom):0;
      // Document coordinates keep ordinary page scrolling from resizing the transcript.
      const top=thread.getBoundingClientRect().top+window.scrollY;
      const viewport=window.visualViewport?.height??window.innerHeight;
      const height=`${Math.max(0,viewport-top-occupied-bottom)}px`;
      if(thread.style.getPropertyValue('--transcript-available-height')!==height)
        thread.style.setProperty('--transcript-available-height',height);
      thread.parentElement?.style.setProperty('--conversation-panel-height',`${thread.getBoundingClientRect().height}px`);
    };
    const schedule=()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(measure);};
    const observer=new ResizeObserver(schedule);
    const observe=()=>{
      observer.disconnect();
      for(const child of thread.children)if(!child.classList.contains('conversation-messages'))observer.observe(child);
      for(let parent=thread.parentElement;parent;parent=parent.parentElement)observer.observe(parent);
      document.querySelectorAll('.topbar,.server-strip,.conversation-heading').forEach(el=>observer.observe(el));
      schedule();
    };
    const mutations=new MutationObserver(observe);
    mutations.observe(thread,{childList:true});
    if(workspace)mutations.observe(workspace,{childList:true});
    observe();measure();
    window.addEventListener('resize',schedule);
    window.visualViewport?.addEventListener('resize',schedule);
    return()=>{
      cancelAnimationFrame(frame);observer.disconnect();mutations.disconnect();
      window.removeEventListener('resize',schedule);
      window.visualViewport?.removeEventListener('resize',schedule);
    };
  },[]);
  return ref;
}
