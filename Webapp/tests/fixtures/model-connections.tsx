import {useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {createApiClient} from '../../src/api';
import {ModelConnectionsView} from '../../src/memory/ModelConnectionsView';
import {MemoryPage} from '../../src/memory/MemoryPage';
import {SourcePage} from '../../src/memory/SourcePage';
import {BrowserRouter,MemoryRouter,Route,Routes} from 'react-router-dom';
import '../../src/identity.css';
import '../../src/memory/memory.css';
function Fixture(){
  const [key,setKey]=useState('fixture-a');
  const api=useMemo(()=>createApiClient(key,()=>{}),[key]);
  if(new URLSearchParams(location.search).has('source'))return <MemoryRouter initialEntries={['/memory/sources/7?space=alpha']}><Routes><Route path="/memory/sources/:id" element={<SourcePage api={api} enabled/>}/></Routes></MemoryRouter>;
  return <main style={{padding:24}}><button onClick={()=>setKey('fixture-b')}>Switch fixture connection</button>{new URLSearchParams(location.search).has('workspace')?<BrowserRouter><MemoryPage api={api}/></BrowserRouter>:<ModelConnectionsView api={api}/>}</main>;
}
createRoot(document.getElementById('root')!).render(<Fixture/>);
