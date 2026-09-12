"""Reproducible local fixture comparison; no general retrieval-quality claim."""
from __future__ import annotations

import asyncio
import json

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.files import document_provenance, ingest_document

METRICS = ('Revenue', 'Expenses', 'Headcount', 'Inventory', 'Refunds', 'Shipments')


def source(metric: str, long_cell: bool) -> bytes:
    if long_cell:
        rows = '<tr><td>' + 'Shared long description with source detail. ' * 80 + 'violet launch milestone</td></tr>'
        header = f'<th>{metric}</th>'
    else:
        header = f'<th>Region</th><th>{metric}</th>'
        rows = ''.join(f'<tr><th scope="row">District {i} Café</th><td>{100+i}</td></tr>' for i in range(30))
    return f'<table><tr>{header}</tr>{rows}</table>'.encode()


async def measure(*, enabled: bool, long_cell: bool) -> dict[str, object]:
    model = HashEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), model,
        chunk_target=120, table_context_embeddings=enabled).open()
    targets: dict[str, tuple[int, int, int]] = {}
    ranks: list[dict[str, object]] = []
    reciprocal = correct = 0.0
    try:
        for metric in METRICS:
            saved = await ingest_document(memory, 'alpha', source(metric, long_cell), filename=metric+'.html')
            evidence = await document_provenance(memory, 'alpha', saved.added.episode_id)
            offset = 0
            for segment in evidence.segments:
                for cell in segment.table_cells:
                    if (long_cell and not cell.is_header) or (not long_cell and cell.row == 21 and not cell.is_header):
                        start = offset + cell.start
                        end = offset + cell.end
                        if long_cell:
                            start = end - len('violet launch milestone'.encode())
                        targets[metric] = (saved.added.episode_id, start, end)
                offset += len(segment.text.encode()) + 2
        for metric in METRICS:
            query = metric + (' violet launch milestone' if long_cell else ' District 20 Café')
            vector = (await model.embed([query]))[0]
            hits = await memory.vectors.search('alpha', vector, limit=1000)
            chunks = {c.chunk_id: c for c in await memory.documents.get_chunks('alpha', [i for i, _ in hits])}
            episode_id, start, end = targets[metric]
            rank = next((n for n, (identifier, _) in enumerate(hits, 1)
                if chunks[identifier].episode_id == episode_id and chunks[identifier].start <= start
                and chunks[identifier].end >= end), None)
            reciprocal += 1 / rank if rank else 0
            correct += rank == 1
            ranks.append({'query': query, 'rank': rank, 'target_span': [start, end]})
    finally:
        await memory.close()
    return {'ranks': ranks, 'mrr': reciprocal / len(METRICS), 'hit_at_1': correct / len(METRICS)}


async def main() -> None:
    report: dict[str, object] = {'scope': 'synthetic hash-256 vector-lane fixtures; not held-out or general quality evidence'}
    for name, long_cell in (('table_values', False), ('long_cells', True)):
        report[name] = {'plain': await measure(enabled=False, long_cell=long_cell),
                        'restored_headers': await measure(enabled=True, long_cell=long_cell)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
