"""Native retrieval runs and resumes with all sockets blocked."""
import os
import socket

import pytest
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.agents import WorkflowError
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex


@pytest.mark.parametrize('mutation',['delete','revision'])
async def test_native_run_resume_and_deletion_work_offline(tmp_path,monkeypatch,mutation):
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


async def test_native_runner_preserves_long_queries_limits_and_source_bytes(tmp_path):
    from scone_memory.agents.retrieval import build_edge_retrieval_runner
    from scone_memory.core.validation import MAX_QUERY

    database = tmp_path / 'memory.db'
    memory = await MemoryEngine(SqliteDocumentStore(database), SqliteVectorIndex(database), HashEmbedder()).open()
    text = 'The café calibration guide is in the observatory wiki.'
    try:
        source = await memory.remember('team', text, source='docs/calibration', metadata={'project': 'edge'})
        await memory.remember('team', 'The observatory also has a calibration checklist.', metadata={'project': 'edge'})
        runner = build_edge_retrieval_runner(memory, tmp_path / 'workflow.db', key=os.urandom(32),
                                             space='team', where={'project': 'edge'}, limit=1)
        try:
            result = await runner.run('long', space='team', scope={'where': {'project': 'edge'}},
                inputs={'query': 'Unrelated background. ' * MAX_QUERY + '\nWhere is the café calibration guide?'})
            assert len(result.results['evidence']) == 1
            record = result.results['evidence'][0]
            assert record['episode_id'] == source.episode_id
            assert record['text'] == text
            assert record['source'] == 'docs/calibration'
        finally:
            runner.close()
    finally:
        await memory.close()


async def test_native_runner_replays_an_empty_snapshot(tmp_path):
    from scone_memory.agents.retrieval import build_edge_retrieval_runner

    database = tmp_path / 'memory.db'
    memory = await MemoryEngine(SqliteDocumentStore(database), SqliteVectorIndex(database), HashEmbedder()).open()
    runner = build_edge_retrieval_runner(memory, tmp_path / 'workflow.db', key=os.urandom(32),
                                         space='team', where={'project': 'edge'})
    try:
        invocation = {'space': 'team', 'scope': {'where': {'project': 'edge'}}, 'inputs': {'query': 'calibration'}}
        first = await runner.run('empty', **invocation)
        assert first.results == {'retrieve': [], 'evidence': []}
        await memory.remember('team', 'A new calibration guide.', metadata={'project': 'edge'})
        replay = await runner.run('empty', **invocation)
        assert replay.results == first.results
        assert replay.reused_steps == ('retrieve', 'evidence')
        fresh = await runner.run('fresh', **invocation)
        assert len(fresh.results['evidence']) == 1
    finally:
        runner.close()
        await memory.close()
