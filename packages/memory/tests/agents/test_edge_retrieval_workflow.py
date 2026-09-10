"""An actual two-framework workflow runs and resumes with all sockets blocked."""
import os
import socket

import pytest
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.agents import WorkflowError
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex


@pytest.mark.parametrize('mutation',['delete','revision'])
async def test_framework_run_resume_and_deletion_work_offline(tmp_path,monkeypatch,mutation):
    pytest.importorskip("langchain_core")
    pytest.importorskip("llama_index.core")
    from scone_memory.agents.retrieval import build_edge_retrieval_runner
    def forbidden(*args,**kwargs):
        raise AssertionError('network must not be used')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(socket,'create_connection',forbidden)
    monkeypatch.setenv('LANGSMITH_TRACING','true')
    path=tmp_path/'memory.db'
    memory=await MemoryEngine(SqliteDocumentStore(path),SqliteVectorIndex(path),HashEmbedder()).open()
    key=os.urandom(32)
    try:
        source=await memory.remember('team','Beacon stores checkpoint progress in an encrypted local journal.',metadata={'project':'edge'})
        await memory.remember('team','Beacon secret belongs to another project.',metadata={'project':'other'})
        await memory.remember('foreign','Beacon private foreign space content.',metadata={'project':'edge'})
        runner=build_edge_retrieval_runner(memory,tmp_path/'workflow.db',key=key,space='team',where={'project':'edge'})
        first=await runner.run('lookup',space='team',scope={'where':{'project':'edge'}},inputs={'query':'Beacon checkpoint'})
        runner.close()
        assert {item['episode_id'] for item in first.results['evidence']}=={source.episode_id}
        assert b'Beacon stores' not in (tmp_path/'workflow.db').read_bytes()
        resumed=build_edge_retrieval_runner(memory,tmp_path/'workflow.db',key=key,space='team',where={'project':'edge'})
        try:
            second=await resumed.run('lookup',space='team',scope={'where':{'project':'edge'}},inputs={'query':'Beacon checkpoint'})
            assert second.reused_steps==('retrieve','evidence')
            assert second.results==first.results
            if mutation == 'delete':
                await memory.forget('team',source.episode_id)
            else:
                # Simulate an adapter revising source bytes while retaining its identity.
                memory.documents.conn.execute('UPDATE episodes SET content = content || ? WHERE id = ?',
                                              (' New revision.',source.episode_id))
                memory.documents.conn.commit()
            with pytest.raises(WorkflowError):
                await resumed.run('lookup',space='team',scope={'where':{'project':'edge'}},inputs={'query':'Beacon checkpoint'})
        finally:
            resumed.close()
    finally:
        await memory.close()


async def test_package_runner_keeps_authorized_scope_outside_query_control(tmp_path):
    pytest.importorskip("langchain_core")
    pytest.importorskip("llama_index.core")
    from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex
    from scone_memory.agents.retrieval import build_edge_retrieval_runner

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        source = await memory.remember("team", "Synthetic calibration evidence.", metadata={"project": "allowed"})
        where = {"project": "allowed"}
        runner = build_edge_retrieval_runner(memory, tmp_path / "scope.db", key=os.urandom(32), space="team", where=where)
        where["project"] = "other"
        try:
            for space, scope in (("foreign", {"where": {"project": "allowed"}}),
                                 ("team", {"where": {"project": "other"}})):
                with pytest.raises(WorkflowError, match="sources_invalid"):
                    await runner.run(f"denied-{space}", space=space, scope=scope, inputs={"query": "calibration"})
            result = await runner.run("allowed", space="team", scope={"where": {"project": "allowed"}},
                                      inputs={"query": "calibration"})
            assert {record["episode_id"] for record in result.results["evidence"]} == {source.episode_id}
        finally:
            runner.close()
    finally:
        await memory.close()
