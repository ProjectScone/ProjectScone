"""Run every question in the public integration fixture; not a research benchmark."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field

from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.embedders.remote import RemoteEmbedder
from scone_memory.ingestion.embedding_cache import InMemoryEmbeddingCache
from scone_memory.providers.jev_sections import JevSectionChooser, MODEL
from scone_memory.retrieval.section_routing import SectionRouter, SectionSnapshot
from scone_memory.retrieval.structured_document import Mode, StructuredDocumentIndex

ROOT = Path(__file__).parent
EMBED_MODEL = 'nvidia/nemotron-3-embed-1b:free'
MODES: tuple[Mode, ...] = ('flat_vector', 'auto', 'section_vector', 'original')


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_contains: list[str] = Field(min_length=1)


async def run(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh output directory')
    questions = [Question.model_validate_json(line)
                 for line in (ROOT / 'questions.jsonl').read_text().splitlines()]
    if len({q.id for q in questions}) != len(questions):
        raise ValueError('duplicate question IDs')
    key = os.environ.get('SCONE_EMBED_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    jev_key = os.environ.get('SCONE_CHAT_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    if not key or not jev_key:
        raise ValueError('embedding and Jev server credentials required')
    snapshot = SectionSnapshot.from_markdown('public-integration-fixture', 'observatory',
                                            (ROOT / 'handbook.md').read_text())
    output.mkdir(parents=True, exist_ok=True)
    source_root = ROOT.parents[1] / 'src/scone_memory'
    files = [Path(__file__), ROOT / 'handbook.md', ROOT / 'questions.jsonl',
        source_root / 'retrieval/section_routing.py', source_root / 'retrieval/structured_document.py',
        source_root / 'providers/jev_sections.py']
    manifest = {'purpose': 'live API integration smoke check; NOT SearchTome or LlamaIndex comparison',
        'source_version': snapshot.version, 'questions': len(questions),
        'embedding_model': EMBED_MODEL, 'dimensions': 2048,
        'query_prefix': 'query: ', 'document_prefix': 'passage: ', 'jev_model': MODEL,
        'limit': 5, 'max_bytes': 1800, 'beam_width': 3, 'max_rounds': 8,
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'timing': 'flat_vector encodes the cold query; later modes reuse its vector. auto uses cold routing; subsequent forced modes reuse that route. Timings are actual calls, not comparable end-to-end arm latencies.',
        'grading': 'case-insensitive literal evidence coverage, not answer accuracy or general retrieval quality'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    remote = RemoteEmbedder('https://openrouter.ai/api/v1', EMBED_MODEL,
        api_key=key, dim=2048, query_prefix='query: ', document_prefix='passage: ', trust_env=False)
    cached = CachedEmbedder(remote, InMemoryEmbeddingCache())
    try:
        started = time.perf_counter()
        index = await StructuredDocumentIndex.build(snapshot, cached)
        build_ms = (time.perf_counter() - started) * 1000
        observations: list[dict[str, object]] = []
        async with JevSectionChooser(api_key=jev_key) as chooser:
            router = SectionRouter(chooser)
            for question in questions:
                # Free embedding route is limited to 20 requests/minute.
                await asyncio.sleep(3.2)
                for mode in MODES:
                    result = await index.retrieve(question.question, snapshot, router, chooser,
                                                  mode=mode, max_bytes=1800)
                    evidence = '\n'.join(item.text for item in result.evidence).casefold()
                    covered = [value.casefold() in evidence for value in question.evidence_contains]
                    row: dict[str, object] = {'id': question.id, 'requested_mode': mode,
                        'all_fixture_evidence_present': all(covered), 'coverage': covered, **asdict(result)}
                    observations.append(row)
                    with (output / 'observations.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    print(json.dumps({'id': question.id, 'requested': mode, 'effective': result.mode,
                        'reason': result.reason, 'covered': all(covered),
                        'total_ms': round(result.total_ms, 2)}), flush=True)
        summary = {'completed_questions': len(questions), 'completed_mode_observations': len(observations),
                   'index_build_ms': build_ms, 'cache': cached.record(),
                   'coverage': {mode: sum(row['all_fixture_evidence_present'] is True
                     for row in observations if row['requested_mode'] == mode) for mode in MODES},
                   'limitation': manifest['purpose']}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    finally:
        await remote.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output))


if __name__ == '__main__':
    main()
