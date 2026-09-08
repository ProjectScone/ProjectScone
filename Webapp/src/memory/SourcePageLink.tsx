import {Link} from 'react-router-dom';
import type {ApiClient} from '../api';
import {sourceAddress} from './source-address';
import {SourceNavigation,useSourceNavigation} from './SourceNavigation';

export function SourcePageLink({api,episodeId}:{api:ApiClient;episodeId:number|string}){
  const context=useSourceNavigation();
  const id=Number(episodeId);
  if(!/^[1-9]\d*$/.test(String(episodeId))||!Number.isSafeInteger(id))return <span className="muted">No valid source reference.</span>;
  if(!context||context.api!==api)return <SourceNavigation api={api} enabled><SourceAnchor episodeId={id}/></SourceNavigation>;
  return <SourceAnchor episodeId={id}/>;
}

function SourceAnchor({episodeId}:{episodeId:number}){
  const context=useSourceNavigation();
  if(!context)return null;
  const {discovery,retry}=context;
  if(discovery.state==='ready')return <Link className="btn quiet small" to={sourceAddress(episodeId,discovery.space)}>Open source page</Link>;
  if(discovery.state==='failed')return <button className="btn quiet small" onClick={retry}>Retry source link</button>;
  if(discovery.state==='unsupported')return <span className="muted">Original pages are not supported by this server.</span>;
  return <span className="muted">{discovery.state==='disconnected'?'Connect memory to open the original.':'Preparing source link…'}</span>;
}
