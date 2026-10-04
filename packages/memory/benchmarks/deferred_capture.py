"""Controlled local A/B: embedding delay on inline versus deferred capture.

Run: PYTHONPATH=src python benchmarks/deferred_capture.py --runs 5 --delay 0.25
SQLite, deterministic hash vectors and a scripted public reply exercise the real
native conversation path. This is a scheduling benchmark, not model quality or
provider latency. Both modes finish indexing before their next sample begins.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Sequence

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.realtime.deferred_capture import DeferredTextCapture
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation


class DelayedDocuments(HashEmbedder):
    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay
        self.document_calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        await asyncio.sleep(self.delay)
        return await super().embed(texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await super().embed(texts)


class ScriptedReply:
    async def respond(self, messages):
        yield TextDelta('The retained calibration note names Vega.')
        yield ReplyCompleted()

    async def aclose(self):
        pass


async def sample(path: Path, mode: str, delay: float) -> dict[str, float | int]:
    embedder = DelayedDocuments(delay)
    memory = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), embedder).open()
    service = DeferredTextCapture(memory) if mode == 'deferred' else None
    if service is not None:
        service.start()
    conversation = TextConversation(memory, 'bench', 'calibration', ScriptedReply, deferred_capture=service)
    first_text_ms = 0.0
    try:
        await memory.remember('bench', 'Juniper uses Vega for calibration.', kind='file', source='calibration.txt')
        embedder.document_calls = 0
        started = time.perf_counter()

        async def on_text(text):
            nonlocal first_text_ms
            if not first_text_ms:
                first_text_ms = (time.perf_counter() - started) * 1000

        result = await conversation.reply('Which star calibrates Juniper?', on_text=on_text)
        completed_ms = (time.perf_counter() - started) * 1000
        pending_at_completion = len(await memory.documents.inflight())
        if service is not None:
            await service.wait_idle()
        fully_indexed_ms = (time.perf_counter() - started) * 1000
        assert not await memory.documents.inflight()
        for role in ('user', 'assistant'):
            assert await memory.episode('bench', result[role + '_episode_id'])
        found = await memory.recall('bench', 'retained calibration note', lanes=('vector',), kind='conversation')
        assert found.items
        return {'first_text_ms': first_text_ms, 'completed_ms': completed_ms,
                'fully_indexed_ms': fully_indexed_ms, 'pending_at_completion': pending_at_completion,
                'document_embedding_calls': embedder.document_calls}
    finally:
        await conversation.close()
        if service is not None:
            await service.aclose()
        await memory.close()


async def run(runs: int, delay: float) -> dict[str, object]:
    samples: dict[str, list[dict[str, float | int]]] = {'inline': [], 'deferred': []}
    with tempfile.TemporaryDirectory(prefix='scone-capture-benchmark-') as folder:
        for index in range(runs):
            for mode in ('inline', 'deferred'):
                samples[mode].append(await sample(Path(folder) / f'{mode}-{index}.db', mode, delay))
    metrics = ('first_text_ms', 'completed_ms', 'fully_indexed_ms')
    return {'kind': 'controlled-local-scheduling', 'runs_per_mode': runs,
            'document_embedding_delay_ms': delay * 1000,
            'median_ms': {mode: {metric: round(statistics.median(float(row[metric]) for row in rows), 3)
                                for metric in metrics} for mode, rows in samples.items()},
            'samples': samples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--delay', type=float, default=0.25)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or not 0 <= args.delay <= 10:
        parser.error('runs must be 1..100 and delay finite seconds in 0..10')
    print(json.dumps(asyncio.run(run(args.runs, args.delay)), indent=2))
