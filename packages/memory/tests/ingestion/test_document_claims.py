"""A source file or manifest stored as a document says what it defines,
imports and depends on, like one remembered through `map`; the durable
directory sync keeps those claims true as files change and go."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import Gone
from scone_memory.ingestion.files import DocumentIngested, ingest_document
from scone_memory.memory.file_claims import Retired

MODULE = "import os\nimport json\n\n\nclass Store:\n    def read(self, path):\n        return json.load(open(path))\n"
SHORTER = "import json\n\n\ndef read(path):\n    return json.load(open(path))\n"


async def facts_of(memory, space, episode_id):
    rows = await memory.documents.facts_for_graph(space, episode_id, 500)
    return {(f.subject, f.predicate, f.object): f for f in rows}


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


async def test_a_stored_source_file_says_what_it_defines_and_imports_and_every_quote_is_in_the_episode(memory):
    ingested = await ingest_document(memory, "code", MODULE.encode(), filename="pkg/store.py")
    assert isinstance(ingested, DocumentIngested) and ingested.claims >= 4
    episode = await memory.episode("code", ingested.added.episode_id)
    facts = await facts_of(memory, "code", episode.episode_id)
    said = {(s, p, o) for (s, p, o), f in facts.items() if f.status == "active"}
    assert ("pkg/store.py", "defines", "pkg/store.py:Store") in said and ("pkg/store.py", "imports", "json") in said
    assert ("pkg/store.py:store", "defines", "pkg/store.py:Store.read") in said, "a subject is held as a normalised term"
    assert all(f.origin == "extracted" for f in facts.values())
    assert all(f.quote and f.quote in episode.content for f in facts.values()), "every claim quotes a line the episode holds"
    again = await ingest_document(memory, "code", MODULE.encode(), filename="pkg/store.py")
    assert again.added.outcome == "duplicate" and again.claims == ingested.claims, "the same file again restates, and the ledger does not grow"
    assert len(await facts_of(memory, "code", episode.episode_id)) == len(facts)


async def test_a_manifest_stored_as_a_document_declares_its_dependencies_and_prose_declares_nothing(memory):
    manifest = await ingest_document(memory, "code", b"requests>=2.31\npytest\n", filename="requirements.txt")
    said = {k for k, f in (await facts_of(memory, "code", manifest.added.episode_id)).items()}
    assert ("requirements.txt", "depends_on", "requests") in said and manifest.claims == 2
    prose = await ingest_document(memory, "code", b"def not_code():\n    pass\n", filename="notes.md")
    assert prose.claims == 0 and not await facts_of(memory, "code", prose.added.episode_id), "a .md holding code is prose that quotes code"
