"""Does an in-process reranker help Scone on LongMemEval-M? Rankings with each local cross-encoder, with every
question's rerank trace kept so a silent fallback cannot pass for "no effect".

Scone falls back to its fused order when the reranker fails or runs past ``SCONE_RERANK_TIMEOUT`` (1 s by
default), and a CPU cross-encoder reading 32 long chat passages can take longer than that. So each variant runs
with the maximum timeout (10 s), and each row records the trace's status, ordering and duration. The report counts
how many questions were actually reranked before comparing evidence coverage with the saved no-reranker rankings.

Vectors come from the run's cache. Every item ranks at its question's date, as the main run does.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .prefill import HostedTwin
from .run import _bench_item, _local, engine_at, load, question_clock, read_jsonl
from .vector_store import Float32Cache

RERANKERS = {
    'minilm': ('ms-marco-MiniLM-L-6-v2', 'Xenova/ms-marco-MiniLM-L-6-v2'),
    'bge': ('bge-reranker-base', 'BAAI/bge-reranker-base'),
}


class _CacheOnly:
    async def embed(self, texts: Any) -> list[list[float]]:
        raise RuntimeError('a vector missing from the cache; run the prefill first')


async def run(dataset: Path, run_dir: Path, models_dir: Path, embed_cache: Path, name: str) -> None:
    from scone_memory.bench.comparative import CachedEmbedder, OneThreadCache, distinct_sessions
    from scone_memory.ingestion.records import Record
    from scone_memory.retrieval.fusion import PER_EPISODE_CAP
    from scone_memory.runtime.config import Settings

    folder, model_id = RERANKERS[name]
    out = run_dir / f'rankings-rerank-{name}.jsonl'
    done = {str(r['question_id']) for r in read_jsonl(out)}
    items = [i for i in load(dataset, 0, 42) if i.question_id not in done]
    embedder = CachedEmbedder(HostedTwin(_local('bge-base-en-v1.5'), _CacheOnly()),
                              OneThreadCache(lambda: Float32Cache(str(embed_cache))))
    settings = Settings.from_env({**os.environ, 'SCONE_EMBEDDER': 'local', 'SCONE_EMBED_MODEL': 'bge-base-en-v1.5',
                                  'SCONE_RERANKER_CROSS_ENCODER_DIR': str(models_dir / folder),
                                  'SCONE_RERANKER_CROSS_ENCODER_MODEL': model_id, 'SCONE_RERANK_TIMEOUT': '10'})
    depth = 10
    for number, item in enumerate(items, 1):
        bench = _bench_item(item.raw, with_sessions=True)
        engine: Any = await engine_at(settings, embedder, question_clock(bench.question_date))
        records = [Record(content='\n'.join(s), kind='conversation', source=bench.session_ids[i],
                          created_at=bench.session_dates[i] or None)
                   for i, s in enumerate(bench.sessions) if '\n'.join(s).strip()]
        await engine.remember_many('item', records)
        started = time.perf_counter()
        found = await engine.recall('item', bench.question, limit=depth * PER_EPISODE_CAP)
        ms = (time.perf_counter() - started) * 1000
        trace = found.rerank.model_dump() if found.rerank is not None else None
        with out.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'question_id': item.question_id, 'reranker': model_id,
                                     name: list(distinct_sessions([i.source or '' for i in found.items], depth)),
                                     'recall_ms': ms, 'rerank': trace}) + '\n')
        if number % 25 == 0:
            print(f'{name}: {number}/{len(items)}', flush=True)


def report(dataset: Path, run_dir: Path) -> dict[str, Any]:
    from math import comb

    base: dict[str, dict[str, Any]] = {str(r['question_id']): r for r in read_jsonl(run_dir / 'rankings.jsonl')}
    items = [i for i in load(dataset, 0, 42) if i.evidence and '_abs' not in i.question_id]
    out: dict[str, Any] = {}
    for name in RERANKERS:
        rows: dict[str, dict[str, Any]] = {str(r['question_id']): r
                                           for r in read_jsonl(run_dir / f'rankings-rerank-{name}.jsonl')}
        if not rows:
            continue
        both = [i for i in items if i.question_id in rows and i.question_id in base]
        status = Counter((rows[i.question_id]['rerank'] or {}).get('status', 'none') for i in both)
        entry: dict[str, Any] = {'items': len(both), 'rerank_status': dict(status)}
        for k in (5, 10):
            plain = [set(i.evidence) <= set(base[i.question_id]['scone'][:k]) for i in both]
            ranked = [set(i.evidence) <= set(rows[i.question_id][name][:k]) for i in both]
            wins = sum(r and not p for p, r in zip(plain, ranked))
            losses = sum(p and not r for p, r in zip(plain, ranked))
            n = wins + losses
            p_value = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(min(wins, losses) + 1)) / 2 ** n)
            entry[f'all@{k}'] = {'no_reranker': sum(plain) / len(both), 'reranked': sum(ranked) / len(both),
                                 'reranked_only': wins, 'no_reranker_only': losses, 'sign_p': p_value}
        ms = sorted(float(rows[i.question_id]['recall_ms']) for i in both)
        entry['recall_ms_p50'] = ms[len(ms) // 2]
        entry['recall_ms_p95'] = ms[int(0.95 * (len(ms) - 1))]
        out[name] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['rank', 'report'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--embed-cache', type=Path)
    parser.add_argument('--reranker', choices=sorted(RERANKERS))
    args = parser.parse_args()
    if args.stage == 'rank':
        if not (args.models_dir and args.embed_cache and args.reranker):
            raise SystemExit('rank needs --models-dir, --embed-cache and --reranker')
        asyncio.run(run(args.dataset, args.run_dir, args.models_dir, args.embed_cache, args.reranker))
    result = report(args.dataset, args.run_dir)
    (args.run_dir / 'report-rerank.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
