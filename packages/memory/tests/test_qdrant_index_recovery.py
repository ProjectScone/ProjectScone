"""Qdrant setup resumes missing payload indexes without replacing a collection."""
from types import SimpleNamespace

import pytest

models = pytest.importorskip('qdrant_client.models')

from scone_memory.backends.qdrant import QdrantVectorIndex

EXPECTED = {'space': models.PayloadSchemaType.KEYWORD,
    'created_ts': models.PayloadSchemaType.FLOAT, 'tags': models.PayloadSchemaType.KEYWORD}


class Client:
    def __init__(self):
        self.exists = False
        self.dim = 3
        self.distance = models.Distance.COSINE
        self.schemas = {}
        self.created = 0
        self.index_calls = []
        self.fail_field = None
        self.persist_before_failure = False

    async def collection_exists(self, name):
        return self.exists

    async def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=models.VectorParams(size=self.dim, distance=self.distance))),
            payload_schema={field: SimpleNamespace(data_type=schema) for field,schema in self.schemas.items()})

    async def create_collection(self, name, vectors_config):
        self.exists = True
        self.created += 1
        self.dim = vectors_config.size

    async def create_payload_index(self, collection, field, schema):
        self.index_calls.append(field)
        if field == self.fail_field:
            if self.persist_before_failure:
                self.schemas[field] = schema
            self.fail_field = None
            raise TimeoutError('index creation response lost')
        self.schemas[field] = schema


@pytest.mark.parametrize('persist_before_failure', [False, True])
async def test_reopen_finishes_interrupted_index_creation(persist_before_failure):
    client = Client()
    client.fail_field = 'created_ts'
    client.persist_before_failure = persist_before_failure
    first = QdrantVectorIndex(client=client)
    with pytest.raises(TimeoutError):
        await first.ensure(3)
    assert first.dim is None and client.created == 1
    before = dict(client.schemas)
    second = QdrantVectorIndex(client=client)
    await second.ensure(3)
    assert client.schemas == EXPECTED
    assert second.dim == 3 and client.created == 1
    assert client.index_calls[2:] == [field for field in EXPECTED if field not in before]
    calls = list(client.index_calls)
    await second.ensure(3)
    assert client.index_calls == calls


async def test_existing_collection_repairs_only_missing_indexes():
    client = Client()
    client.exists = True
    client.schemas = {'space': EXPECTED['space'], 'custom': models.PayloadSchemaType.INTEGER}
    index = QdrantVectorIndex(client=client)
    await index.ensure(3)
    assert client.created == 0
    assert client.index_calls == ['created_ts', 'tags']
    assert client.schemas == {**EXPECTED, 'custom': models.PayloadSchemaType.INTEGER}


async def test_existing_mismatched_schema_is_rejected_before_any_write():
    client = Client()
    client.exists = True
    client.schemas = {'tags': models.PayloadSchemaType.INTEGER}
    with pytest.raises(ValueError, match='payload index.*tags'):
        await QdrantVectorIndex(client=client).ensure(3)
    assert client.index_calls == [] and client.created == 0


async def test_dimension_mismatch_never_mutates_indexes():
    client = Client()
    client.exists = True
    with pytest.raises(ValueError, match='holds 3-d'):
        await QdrantVectorIndex(client=client).ensure(4)
    assert client.index_calls == [] and client.created == 0


@pytest.mark.parametrize('distance', [models.Distance.DOT, models.Distance.EUCLID])
async def test_existing_metric_mismatch_never_mutates_indexes(distance):
    client = Client()
    client.exists = True
    client.distance = distance
    with pytest.raises(ValueError, match='cosine'):
        await QdrantVectorIndex(client=client).ensure(3)
    assert client.index_calls == [] and client.created == 0


@pytest.mark.parametrize('persist_before_failure', [False, True])
async def test_configured_metadata_indexes_resume_without_replacing_collection(persist_before_failure):
    client = Client()
    client.exists = True
    client.schemas = dict(EXPECTED)
    client.fail_field = 'meta.document_format'
    client.persist_before_failure = persist_before_failure
    with pytest.raises(TimeoutError):
        await QdrantVectorIndex(client=client, metadata_indexes=('document_format', 'entity_id')).ensure(3)
    before = dict(client.schemas)
    index = QdrantVectorIndex(client=client, metadata_indexes=('document_format', 'entity_id', 'entity_id'))
    await index.ensure(3)
    assert client.schemas == {**EXPECTED, 'meta.document_format': models.PayloadSchemaType.KEYWORD,
                             'meta.entity_id': models.PayloadSchemaType.KEYWORD}
    assert client.created == 0
    assert client.index_calls[1:] == [name for name in ('meta.document_format', 'meta.entity_id') if name not in before]
    calls = list(client.index_calls)
    await index.ensure(3)
    assert client.index_calls == calls


async def test_metadata_schema_conflict_rejects_before_creating_other_indexes():
    client = Client()
    client.exists = True
    client.schemas = {'meta.entity_id': models.PayloadSchemaType.INTEGER}
    with pytest.raises(ValueError, match='payload index.*meta.entity_id'):
        await QdrantVectorIndex(client=client, metadata_indexes=('document_format', 'entity_id')).ensure(3)
    assert client.index_calls == [] and client.created == 0


@pytest.mark.parametrize('keys', ['entity_id', ('meta.entity_id',), ('Bad-Key',), ('',), tuple(f'k{i}' for i in range(17))])
def test_invalid_metadata_index_configuration_fails_without_client_requests(keys):
    client = Client()
    with pytest.raises(ValueError, match='metadata_indexes'):
        QdrantVectorIndex(client=client, metadata_indexes=keys)
    assert client.index_calls == [] and client.created == 0


@pytest.mark.qdrant
@pytest.mark.parametrize('metadata_indexes', [(), ('document_format', 'entity_id')])
async def test_real_server_repairs_partial_schema_and_preserves_points(metadata_indexes):
    import os
    from uuid import uuid4
    from qdrant_client import AsyncQdrantClient
    url = os.environ.get('SCONE_TEST_QDRANT_URL')
    if not url:
        pytest.skip('SCONE_TEST_QDRANT_URL is not configured')
    client = AsyncQdrantClient(url=url)
    collection = 'scone_index_recovery_' + uuid4().hex
    try:
        await client.create_collection(collection,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
        await client.create_payload_index(collection, 'space', models.PayloadSchemaType.KEYWORD)
        await client.upsert(collection, points=[models.PointStruct(id=7, vector=[1.,0.,0.],
            payload={'space':'alpha', 'tags':['public'], 'created_ts':0.,
                     'meta': {'document_format': 'image', 'entity_id': 'entity7'}})])
        before = await client.get_collection(collection)
        assert set(before.payload_schema) == {'space'}
        index = QdrantVectorIndex(collection=collection, client=client, metadata_indexes=metadata_indexes,
                                  hnsw_ef=128 if metadata_indexes else None)
        await index.ensure(3)
        after = await client.get_collection(collection)
        assert after.config.hnsw_config == before.config.hnsw_config
        assert after.config.optimizer_config == before.config.optimizer_config
        assert {name: info.data_type for name,info in after.payload_schema.items()} == {
            **EXPECTED, **{f'meta.{key}': models.PayloadSchemaType.KEYWORD for key in metadata_indexes}}
        assert (await client.count(collection, exact=True)).count == 1
        result = await index.search('alpha', [1.,0.,0.], 1, tags=('public',),
                                    where={'document_format': 'image', 'entity_id': 'entity7'})
        assert len(result) == 1 and result[0][0] == 7
        assert await index.search('alpha', [1.,0.,0.], 1, where={'entity_id': 'other'}) == []
        assert await index.search('beta', [1.,0.,0.], 1) == []
        await index.ensure(3)
        assert (await client.count(collection, exact=True)).count == 1
    finally:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
        await client.close()
