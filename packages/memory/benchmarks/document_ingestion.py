"""Reproducible mixed-document operational workload using real local stores.

Run with Python 3.14 and PYTHONPATH=packages/memory/src. No model downloads,
hosted services, pytest transport, or user documents are used. Hash embeddings
exercise storage/retrieval plumbing, not semantic or generated-answer quality.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import TypeVar
import uuid
from zipfile import ZipFile, ZipInfo

import httpx
import scone_memory
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.agents import WorkflowError
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.backends.qdrant import QdrantVectorIndex
from scone_memory.ingestion import (BuiltinDocumentParser,
    DocumentLimits, ParsedDocument, document_provenance, ingest_document)
from scone_memory.ingestion.files import FILE_MEDIA_TYPES
from scone_memory.ingestion.file_workflow import DocumentIngestionWorkflow

from knowledge_lifecycle import loopback

T = TypeVar('T')


@dataclass(frozen=True)
class Fixture:
    name: str
    data: bytes
    query: str
    origin: str = 'generated synthetic fixture'


def archive(parts: dict[str, str]) -> bytes:
    output = BytesIO()
    with ZipFile(output, 'w') as bundle:
        for name, body in parts.items():
            bundle.writestr(ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), body)
    return output.getvalue()


def fixtures(repo: Path) -> list[Fixture]:
    """Small OOXML packages include content types and package relationships."""
    w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    s = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    p = 'http://schemas.openxmlformats.org/presentationml/2006/main'
    a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    rel = 'http://schemas.openxmlformats.org/package/2006/relationships'

    def office(parts: dict[str, str], main: str, mime: str) -> bytes:
        parts['[Content_Types].xml'] = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/{main}" ContentType="{mime}"/></Types>')
        parts['_rels/.rels'] = f'<Relationships xmlns="{rel}"><Relationship Id="root" Type="{r}/officeDocument" Target="{main}"/></Relationships>'
        return archive(parts)

    docs = [Fixture('calibration.docx', office({'word/document.xml':
        f'<w:document xmlns:w="{w}"><w:body><w:p><w:r><w:t>Juniper calibration uses Polaris.</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>Ada</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'},
        'word/document.xml', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'), 'Polaris'),
        Fixture('budget.xlsx', office({
            'xl/workbook.xml': f'<workbook xmlns="{s}" xmlns:r="{r}"><sheets><sheet name="Budget" sheetId="1" r:id="s1"/></sheets></workbook>',
            'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{rel}"><Relationship Id="s1" Target="worksheets/sheet1.xml" Type="{r}/worksheet"/></Relationships>',
            'xl/worksheets/sheet1.xml': f'<worksheet xmlns="{s}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Saffron budget</t></is></c><c r="B1"><v>42</v></c><c r="C1"><f>B1*2</f><v>84</v></c></row></sheetData></worksheet>'},
            'xl/workbook.xml', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'), 'Saffron'),
        Fixture('briefing.pptx', office({
            'ppt/presentation.xml': f'<p:presentation xmlns:p="{p}" xmlns:r="{r}"><p:sldIdLst><p:sldId id="256" r:id="s1"/></p:sldIdLst></p:presentation>',
            'ppt/_rels/presentation.xml.rels': f'<Relationships xmlns="{rel}"><Relationship Id="s1" Target="slides/slide1.xml" Type="{r}/slide"/></Relationships>',
            'ppt/slides/slide1.xml': f'<p:sld xmlns:p="{p}" xmlns:a="{a}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Orchid launch is Friday.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
            'ppt/slides/_rels/slide1.xml.rels': f'<Relationships xmlns="{rel}"><Relationship Id="n1" Target="../notesSlides/notesSlide1.xml" Type="{r}/notesSlide"/></Relationships>',
            'ppt/notesSlides/notesSlide1.xml': f'<p:notes xmlns:p="{p}" xmlns:a="{a}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Speaker note: verify Orchid rollback.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>'},
            'ppt/presentation.xml', 'application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'), 'Orchid'),
        Fixture('release.json', b'{"release":{"codename":"Marigold","replicas":3}}', 'Marigold'),
        Fixture('schedule.csv', b'project,day\nCobalt,Tuesday\nAmber,Thursday\n', 'Cobalt'),
        Fixture('notice.eml', b'From: sender@example.invalid\r\nTo: reader@example.invalid\r\nSubject: Copper maintenance\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nCopper maintenance begins at noon.\r\n', 'Copper')]
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    for phrase in ('Indigo readiness review.', 'Rollback is scheduled for Sunday.'):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 72 720 Td ({phrase}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    docs.append(Fixture('readiness.pdf', output.getvalue(), 'Indigo'))
    for name in ('README.md', 'ARCHITECTURE.md'):
        path = f'packages/memory/{name}'
        raw = subprocess.run(['git', 'show', f'HEAD:{path}'], cwd=repo, check=True, capture_output=True).stdout
        docs.append(Fixture(name, raw, 'memory', f'public ProjectScone repository HEAD:{path}'))
    return docs


class CountingParser:
    def __init__(self) -> None:
        self.calls = 0
        self.elapsed_ms = 0.0
        self.parser = BuiltinDocumentParser()

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
        self.calls += 1
        start = time.perf_counter()
        try:
            return await self.parser.parse(data, filename, limits)
        finally:
            self.elapsed_ms += (time.perf_counter() - start) * 1000


class PausedHashEmbedder(HashEmbedder):
    """Controlled cancellation point before real hashing; no storage is mocked."""
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.entered.set()
        await asyncio.Event().wait()
        return await super().embed(texts)


class Workload:
    def __init__(self, directory: Path, url: str | None) -> None:
        self.directory = directory
        self.url = url
        self.collection = 'scone_document_workload_' + uuid.uuid4().hex
        self.key = secrets.token_bytes(32)
        self.checks: list[str] = []
        self.errors: list[dict[str, str]] = []
        self.measurements: list[dict[str, object]] = []
        self.sources: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self.open_engines: list[MemoryEngine] = []
        self.open_jobs: list[DocumentIngestionWorkflow] = []

    def require(self, condition: bool, label: str) -> None:
        if not condition:
            raise RuntimeError(label)
        self.checks.append(label)

    async def measured(self, label: str, parser: CountingParser, action: Callable[[], Awaitable[T]]) -> T:
        calls, parse_ms, start = parser.calls, parser.elapsed_ms, time.perf_counter()
        passed = False
        try:
            result = await action()
            passed = True
            return result
        finally:
            self.measurements.append({'operation': label, 'parser_calls': parser.calls - calls,
                'parse_ms': parser.elapsed_ms - parse_ms, 'elapsed_ms': (time.perf_counter() - start) * 1000,
                'passed': passed})

    async def memory(self, embedder: HashEmbedder | None = None) -> MemoryEngine:
        vectors = (QdrantVectorIndex(self.url, self.collection) if self.url
                   else SqliteVectorIndex(self.directory / 'memory.db'))
        memory = MemoryEngine(SqliteDocumentStore(self.directory / 'memory.db'), vectors,
            embedder or HashEmbedder(), blobs=FileBlobStore(self.directory / 'blobs'))
        self.open_engines.append(memory)
        return await memory.open()

    async def close_memory(self, memory: MemoryEngine) -> None:
        await memory.close()
        self.open_engines.remove(memory)

    def job(self, memory: MemoryEngine, filename: str, parser: CountingParser, *, retries: int = 1) -> DocumentIngestionWorkflow:
        job = DocumentIngestionWorkflow(memory, self.directory / f'{filename}.db', key=self.key,
            parser=parser, parser_revision='builtin-document-v1', max_retries=retries)
        self.open_jobs.append(job)
        return job

    def close_job(self, job: DocumentIngestionWorkflow) -> None:
        job.close()
        self.open_jobs.remove(job)

    async def evidence(self, memory: MemoryEngine, fixture: Fixture, episode_id: int, phase: str) -> None:
        # Explicit identity filter tests retrieval/citation plumbing, not ranking quality.
        digest = hashlib.sha256(fixture.data).hexdigest()
        recalled = await memory.recall('mixed', fixture.query, limit=5, where={'document_original': digest})
        self.require(bool(recalled.items), f'{phase}:{fixture.name}: recall source')
        citations: list[dict[str, object]] = []
        query_located = False
        for item in recalled.items:
            self.require(item.episode_id == episode_id, f'{phase}:{fixture.name}: source identity {item.chunk_id}')
            evidence = await document_provenance(memory, 'mixed', item.episode_id, chunk_id=item.chunk_id)
            _, retained = await memory.attachment('mixed', evidence.original.attachment_id)
            self.require(retained == fixture.data, f'{phase}:{fixture.name}: original bytes {item.chunk_id}')
            self.require(bool(evidence.segments) and all(s.locator for s in evidence.segments),
                         f'{phase}:{fixture.name}: locators {item.chunk_id}')
            query_located = query_located or any(fixture.query.casefold() in s.text.casefold() for s in evidence.segments)
            citations.append({'chunk_id': item.chunk_id, 'episode_id': item.episode_id,
                'locators': [s.locator for s in evidence.segments]})
        self.require(query_located, f'{phase}:{fixture.name}: query appears in cited segment')
        self.sources.append({'phase': phase, 'filename': fixture.name, 'query': fixture.query,
            'where': {'document_original': digest}, 'recall': json.loads(recalled.model_dump_json()), 'citations': citations})

    async def run(self, docs: list[Fixture]) -> None:
        memory = await self.memory()
        episodes: dict[str, int] = {}
        for index, fixture in enumerate(docs):
            direct, durable = CountingParser(), CountingParser()
            job = self.job(memory, f'pair-{index}', durable)
            original = await memory.attach('mixed', fixture.data,
                FILE_MEDIA_TYPES.get(Path(fixture.name).suffix, 'application/octet-stream'), filename=fixture.name)

            async def baseline() -> None:
                first = await self.measured(f'direct.initial:{fixture.name}', direct,
                    lambda: ingest_document(memory, 'mixed', fixture.data, filename=fixture.name, parser=direct))
                replay = await self.measured(f'direct.replay:{fixture.name}', direct,
                    lambda: ingest_document(memory, 'mixed', fixture.data, filename=fixture.name, parser=direct))
                self.require(replay.added.deduplicated and replay.added.episode_id == first.added.episode_id,
                    f'{fixture.name}: direct duplicate preserves identity')
                episodes[fixture.name] = first.added.episode_id

            async def checkpointed() -> None:
                await self.measured(f'workflow.initial:{fixture.name}', durable,
                    lambda: job.run('pair', space='mixed', attachment_id=original.attachment_id))
                replay = await self.measured(f'workflow.replay:{fixture.name}', durable,
                    lambda: job.run('pair', space='mixed', attachment_id=original.attachment_id))
                self.require(replay.reused_steps == ('extract', 'index') and durable.calls == 1,
                    f'{fixture.name}: workflow replay skips parser')

            # Alternate order; both paths use the same retained bytes and warm stores.
            if index % 2:
                await checkpointed()
                await baseline()
            else:
                await baseline()
                await checkpointed()
            self.close_job(job)
            await self.evidence(memory, fixture, episodes[fixture.name], 'initial')
        self.require((await memory.documents.counts('mixed')).episodes == len(docs), 'one episode per source after paired replays')
        await self.close_memory(memory)
        memory = await self.memory()
        for fixture in docs:
            await self.evidence(memory, fixture, episodes[fixture.name], 'reopened')
        await self.close_memory(memory)
        await self.cancel_resume(docs[0])
        await self.failure()

    async def cancel_resume(self, fixture: Fixture) -> None:
        paused, parser = PausedHashEmbedder(), CountingParser()
        memory = await self.memory(paused)
        original = await memory.attach('cancel', fixture.data, FILE_MEDIA_TYPES['.docx'], filename=fixture.name)
        job = self.job(memory, 'cancel', parser)
        task = asyncio.create_task(job.run('cancel', space='cancel', attachment_id=original.attachment_id))
        try:
            await asyncio.wait_for(paused.entered.wait(), 30)
            state = job.status('cancel', space='cancel', attachment_id=original.attachment_id)
            self.require(state is not None and state.completed_steps == ('extract',) and state.inflight == 'index',
                'cancellation occurs after durable extraction')
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                self.events.append({'expected': 'CancelledError', 'operation': 'cancel.index'})
        self.close_job(job)
        await self.close_memory(memory)
        memory = await self.memory()
        job = self.job(memory, 'cancel', parser)
        state = job.status('cancel', space='cancel', attachment_id=original.attachment_id)
        self.require(state is not None and state.status == 'cancelled', 'cancelled journal survives close/reopen')
        self.events.append({'operation': 'cancel.reopened', 'state': asdict(state) if state else None})
        result = await self.measured('cancel.resume', parser,
            lambda: job.run('cancel', space='cancel', attachment_id=original.attachment_id))
        self.require(result.reused_steps == ('extract',) and parser.calls == 1, 'resume indexes without repeated extraction')
        self.require((await memory.documents.counts('cancel')).episodes == 1, 'cancel/resume creates one episode')
        self.events.append({'operation': 'cancel.resumed', 'result': asdict(result), 'total_parser_calls': parser.calls})
        self.close_job(job)
        await self.close_memory(memory)

    async def failure(self) -> None:
        memory, parser = await self.memory(), CountingParser()
        original = await memory.attach('failure', b'invalid OOXML archive', FILE_MEDIA_TYPES['.docx'], filename='broken.docx')
        job = self.job(memory, 'failure', parser, retries=0)
        try:
            await job.run('failure', space='failure', attachment_id=original.attachment_id)
        except WorkflowError as error:
            self.require(error.code == 'step_failed', 'malformed source fails extraction')
            self.events.append({'operation': 'malformed', 'expected_error': error.code})
        else:
            raise RuntimeError('malformed source unexpectedly succeeded')
        self.close_job(job)
        await self.close_memory(memory)
        memory = await self.memory()
        job = self.job(memory, 'failure', parser, retries=0)
        state = job.status('failure', space='failure', attachment_id=original.attachment_id)
        self.require(state is not None and state.status == 'failed' and state.error_class == 'InvalidInput'
            and state.completed_steps == () and state.attempts == {'extract': 1}, 'failed stage survives close/reopen')
        self.require((await memory.documents.counts('failure')).episodes == 0, 'malformed source creates no searchable episode')
        self.events.append({'operation': 'malformed.reopened', 'state': asdict(state) if state else None})
        self.close_job(job)
        await self.close_memory(memory)

    async def cleanup(self) -> None:
        for job in list(self.open_jobs):
            try:
                self.close_job(job)
            except Exception as error:
                self.errors.append({'stage': 'cleanup.journal', 'class': type(error).__name__, 'message': str(error)})
        for memory in list(self.open_engines):
            try:
                await self.close_memory(memory)
            except Exception as error:
                self.errors.append({'stage': 'cleanup.memory', 'class': type(error).__name__, 'message': str(error)})


async def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True, help='New directory for source fixtures and raw results')
    cli.add_argument('--qdrant-url', type=loopback, help='Existing loopback server; otherwise use real SQLite vectors')
    args = cli.parse_args()
    directory = args.output.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    workload = Workload(directory, args.qdrant_url)
    report: dict[str, object] = {'schema_version': 1, 'kind': 'mixed_document_operational',
        'python': sys.version, 'embedder': 'HashEmbedder', 'semantic_quality_tested': False,
        'generation_tested': False, 'restart_kind': 'close/reopen engine and journal in one process',
        'vector_backend': 'Qdrant server' if args.qdrant_url else 'SQLite',
        'collection': workload.collection if args.qdrant_url else None, 'checks': workload.checks,
        'errors': workload.errors, 'measurements': workload.measurements, 'sources': workload.sources,
        'events': workload.events, 'passed': False}
    creation_attempted = False
    started = time.perf_counter()
    try:
        root = Path(__file__).resolve().parents[3]
        docs = fixtures(root)
        report['versions'] = {name: version(name) for name in ('scone-memory', 'pypdf', 'defusedxml', 'qdrant-client', 'cryptography')}
        report['git_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        source_root = Path(scone_memory.__file__).parent
        source_hash = hashlib.sha256()
        for path in sorted(source_root.rglob('*.py')):
            source_hash.update(str(path.relative_to(source_root)).encode() + b'\0' + path.read_bytes() + b'\0')
        report['source_sha256'] = source_hash.hexdigest()
        report['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['fixtures'] = [{'filename': f.name, 'bytes': len(f.data), 'sha256': hashlib.sha256(f.data).hexdigest(), 'origin': f.origin} for f in docs]
        (directory / 'fixtures').mkdir()
        for fixture in docs:
            (directory / 'fixtures' / fixture.name).write_bytes(fixture.data)
        if args.qdrant_url:
            async with httpx.AsyncClient(base_url=args.qdrant_url, trust_env=False, timeout=30) as client:
                response = await client.get(f'/collections/{workload.collection}')
                workload.require(response.status_code == 404, 'unique owned collection initially absent')
                creation_attempted = True
        await workload.run(docs)
    except BaseException as error:
        workload.errors.append({'stage': 'workload', 'class': type(error).__name__, 'message': str(error)})
    finally:
        await workload.cleanup()
        if args.qdrant_url and creation_attempted:
            try:
                async with httpx.AsyncClient(base_url=args.qdrant_url, trust_env=False, timeout=30) as client:
                    response = await client.delete(f'/collections/{workload.collection}')
                    workload.require(response.status_code in (200, 404), 'owned collection cleanup accepted')
                    workload.require((await client.get(f'/collections/{workload.collection}')).status_code == 404,
                        'owned collection absent after cleanup')
            except Exception as error:
                workload.errors.append({'stage': 'cleanup.collection', 'class': type(error).__name__, 'message': str(error)})
        report['elapsed_ms'] = (time.perf_counter() - started) * 1000
        report['passed'] = not workload.errors
        (directory / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(workload.checks), 'errors': workload.errors,
        'results': str(directory / 'results.json')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', ORT_DISABLE_TELEMETRY='1', NO_PROXY='*', no_proxy='*')
    raise SystemExit(asyncio.run(main()))
