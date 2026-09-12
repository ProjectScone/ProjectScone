"""An admitted import belongs to the local service rather than its HTTP caller."""
import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.workflow import WorkflowError
from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
from scone_memory.ingestion.import_service import DocumentImportService, ImportParserBinding


class HeldParser:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def parse(self, data, filename, limits):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return await BuiltinDocumentParser().parse(data, filename, limits)


async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    original = await engine.attach('alpha', b'Ada maintains the observatory.', 'text/plain', filename='note.md')
    return engine, original.attachment_id


def service(tmp_path, engine, parser, **options):
    return DocumentImportService(tmp_path / 'imports', key=b'k' * 32, memory=engine,
        parser_for=lambda selection: ImportParserBinding('parser-v1', parser), **options)


@pytest.mark.asyncio
async def test_admission_survives_waiter_disconnect_and_repeated_start(tmp_path):
    engine, original = await memory(); parser = HeldParser(); owner = service(tmp_path, engine, parser)
    try:
        first = await owner.start('alpha', 'one', attachment_id=original, filename='note.md')
        assert first.active_local and first.attempt == 1
        await asyncio.wait_for(parser.entered.wait(), 3)
        repeated = await owner.start('alpha', 'one', attachment_id=original, filename='note.md')
        assert repeated.attempt == 1 and parser.calls == 1
        waiter = asyncio.create_task(owner.wait('alpha', 'one'))
        await asyncio.sleep(0); waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert (await owner.status('alpha', 'one')).active_local
        parser.release.set()
        finished = await owner.wait('alpha', 'one')
        assert finished.status == 'completed' and not finished.active_local
        result = await owner.result('alpha', 'one')
        assert result.added.episode_id > 0 and result.original.attachment_id == original
        assert (await engine.episodes('alpha', {'document_original': original}))[0].content == 'Ada maintains the observatory.'
    finally:
        await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_restart_is_passive_and_resume_is_explicit(tmp_path):
    engine, original = await memory(); parser = HeldParser(); owner = service(tmp_path, engine, parser)
    await owner.start('alpha', 'one', attachment_id=original, filename='note.md')
    await asyncio.wait_for(parser.entered.wait(), 3); await owner.aclose()
    replacement = HeldParser(); reopened = service(tmp_path, engine, replacement)
    try:
        status = await reopened.status('alpha', 'one')
        assert not status.active_local and replacement.calls == 0
        assert (await reopened.request('alpha', 'one')).spec.attachment_id == original
        repeated = await reopened.start('alpha', 'one', attachment_id=original, filename='note.md')
        assert not repeated.active_local and replacement.calls == 0
        with pytest.raises(WorkflowError, match='not_completed'):
            await reopened.result('alpha', 'one')
        await reopened.resume('alpha', 'one', expected_revision=status.revision)
        await asyncio.wait_for(replacement.entered.wait(), 3); replacement.release.set()
        assert (await reopened.wait('alpha', 'one')).status == 'completed'
        assert len(await engine.episodes('alpha', {'document_original': original})) == 1
    finally:
        await reopened.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_other_owner_cannot_start_or_cancel_active_import(tmp_path):
    engine, original = await memory(); parser = HeldParser()
    owner = service(tmp_path, engine, parser); other = service(tmp_path, engine, BuiltinDocumentParser())
    try:
        await owner.start('alpha', 'one', attachment_id=original, filename='note.md'); await asyncio.wait_for(parser.entered.wait(), 3)
        status = await other.status('alpha', 'one')
        with pytest.raises(WorkflowError, match='not_owned|busy'):
            await other.resume('alpha', 'one', expected_revision=status.revision)
        with pytest.raises(WorkflowError, match='not_owned|busy'):
            await other.cancel('alpha', 'one', expected_revision=status.revision)
        assert (await owner.request('alpha', 'one')).cancel_requested_at is None
        assert await other.status('beta', 'one') is None
        assert await other.cancel('beta', 'one', expected_revision=0) is None
        parser.release.set(); await owner.wait('alpha', 'one')
    finally:
        await other.aclose(); await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_capacity_refusal_and_cancel_have_no_duplicate_work(tmp_path):
    engine, original = await memory(); parser = HeldParser(); owner = service(tmp_path, engine, parser, max_active=1)
    try:
        await owner.start('alpha', 'one', attachment_id=original, filename='note.md'); await asyncio.wait_for(parser.entered.wait(), 3)
        with pytest.raises(WorkflowError, match='busy'):
            await owner.start('alpha', 'two', attachment_id=original, filename='note.md')
        assert await owner.request('alpha', 'two') is None
        status = await owner.status('alpha', 'one')
        cancelled = await owner.cancel('alpha', 'one', expected_revision=status.revision)
        assert not cancelled.active_local and cancelled.status == 'cancelled'
        assert len(await engine.episodes('alpha', {'document_original': original})) == 0
        assert parser.calls == 1
    finally:
        await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_result_reads_verify_retained_source_without_reexecuting(tmp_path):
    engine, original = await memory(); parser = HeldParser(); parser.release.set(); owner = service(tmp_path, engine, parser)
    try:
        await owner.start('alpha', 'one', attachment_id=original, filename='note.md'); await owner.wait('alpha', 'one')
        result = await owner.result('alpha', 'one'); calls = parser.calls
        await engine.forget('alpha', result.added.episode_id)
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await owner.result('alpha', 'one')
        assert parser.calls == calls and len(await engine.episodes('alpha', {'document_original': original})) == 0
    finally:
        await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_failed_parser_waits_for_explicit_resume_and_budget_is_durable(tmp_path):
    class Broken:
        calls = 0
        async def parse(self, data, filename, limits):
            self.calls += 1
            raise RuntimeError('unavailable')
    engine, original = await memory(); parser = Broken(); owner = service(tmp_path, engine, parser, max_attempts=2)
    try:
        await owner.start('alpha', 'one', attachment_id=original, filename='note.md')
        failed = await owner.wait('alpha', 'one')
        assert failed.status == 'failed' and parser.calls == 1
        await owner.aclose()
        owner = service(tmp_path, engine, parser, max_attempts=4)
        await owner.resume('alpha', 'one', expected_revision=failed.revision)
        failed = await owner.wait('alpha', 'one')
        assert parser.calls == 2
        with pytest.raises(WorkflowError, match='retries_exhausted'):
            await owner.resume('alpha', 'one', expected_revision=failed.revision)
        assert parser.calls == 2 and (await owner.request('alpha', 'one')).attempt == 2
    finally:
        await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_admission_guard_refuses_changed_authorization_without_registering(tmp_path):
    engine, original = await memory(); owner = service(tmp_path, engine, BuiltinDocumentParser())
    def revoked():
        raise PermissionError('revoked')
    try:
        with pytest.raises(PermissionError):
            await owner.start('alpha', 'one', attachment_id=original, filename='note.md', admission_guard=revoked)
        assert await owner.request('alpha', 'one') is None
    finally:
        await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_stale_or_coerced_control_cannot_cancel_a_resumed_attempt(tmp_path):
    engine, original = await memory(); parser = HeldParser(); owner = service(tmp_path, engine, parser)
    try:
        first = await owner.start('alpha', 'one', attachment_id=original, filename='note.md')
        await asyncio.wait_for(parser.entered.wait(), 3)
        cancelled = await owner.cancel('alpha', 'one', expected_revision=first.revision)
        await owner.resume('alpha', 'one', expected_revision=cancelled.revision)
        for revision in (first.revision, True, '3', 3.0):
            with pytest.raises(WorkflowError):
                await owner.cancel('alpha', 'one', expected_revision=revision)
        assert (await owner.request('alpha', 'one')).cancel_requested_at is None
        parser.release.set()
        assert (await owner.wait('alpha', 'one')).status == 'completed'
    finally:
        await owner.aclose(); await engine.close()


@pytest.mark.asyncio
async def test_cancel_stops_owned_work_even_when_intent_storage_fails(tmp_path, monkeypatch):
    engine, original = await memory(); parser = HeldParser(); owner = service(tmp_path, engine, parser)
    try:
        first = await owner.start('alpha', 'one', attachment_id=original, filename='note.md')
        await asyncio.wait_for(parser.entered.wait(), 3)
        def unavailable(*args, **kwargs):
            raise WorkflowError('import_store_unavailable')
        monkeypatch.setattr(owner._imports, 'request_cancel', unavailable)
        with pytest.raises(WorkflowError, match='import_store_unavailable'):
            await owner.cancel('alpha', 'one', expected_revision=first.revision)
        assert not (await owner.status('alpha', 'one')).active_local
        parser.release.set()
        assert not await engine.episodes('alpha', {'document_original': original})
    finally:
        await owner.aclose(); await engine.close()
