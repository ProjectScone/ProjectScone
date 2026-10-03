"""The engine's second hop (``retrieval.second_hop``) end to end, on the held-out half of HotpotQA.

``hop_rule.py`` chose the rule over saved document lists. This runs the engine's own implementation, over passages
(seeded with the leading passage and keeping the leading episodes), on the test half only. It writes
``rankings-hotpotqa-enginehop.jsonl`` and compares it with Scone's and LlamaIndex's saved lists for the same
questions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hop_rule import is_test
from .run import FIXED_NOW, distinct, load, read_jsonl, score, sign_test


async def run(data_dir: Path, run_dir: Path, depth: int, embed_model: str, embed_cache: Path) -> None:
    from longmemeval_matched.run import _embedder
    from scone_memory.ingestion.records import Record
    from scone_memory.retrieval.fusion import PER_EPISODE_CAP
    from scone_memory.retrieval.second_hop import recall_with_hop
    from scone_memory.runtime.config import Settings, build_in_process_engine

    from .fast_index import install

    docs, questions = load('hotpotqa', data_dir)
    out = run_dir / 'rankings-hotpotqa-enginehop.jsonl'
    done = {r['id'] for r in read_jsonl(out)}
    todo = [q for q in questions if is_test(q.id) and q.id not in done]
    print(f'{len(todo)} held-out questions to rank', flush=True)
    if not todo:
        return
    embedder, _ = _embedder(embed_model, embed_cache)
    install()
    settings = Settings.from_env({**os.environ, 'SCONE_EMBEDDER': 'local', 'SCONE_EMBED_MODEL': embed_model})
    engine = await build_in_process_engine(settings, embedder)
    engine.clock = lambda: FIXED_NOW
    for start in range(0, len(docs), 2000):
        await engine.remember_many('bench', [Record(content=d.text, source=d.id, created_at=FIXED_NOW, dedup_key=d.id)
                                             for d in docs[start:start + 2000]])
    limit = depth * PER_EPISODE_CAP

    async def recall(query: str) -> Any:
        return await engine.recall('bench', query, limit=limit)

    with out.open('a', encoding='utf-8') as handle:
        for number, question in enumerate(todo, 1):
            started = time.perf_counter()
            result, trace = await recall_with_hop(recall, question.question, limit=limit)
            handle.write(json.dumps({'id': question.id, 'enginehop': distinct([i.source or '' for i in result.items], depth),
                                     'added': trace.added if trace else 0,
                                     'ms': (time.perf_counter() - started) * 1000}) + '\n')
            if number % 500 == 0:
                handle.flush()
                print(f'  ranked {number}/{len(todo)}', flush=True)


def report(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    _, questions = load('hotpotqa', data_dir)
    base = {r['id']: r for r in read_jsonl(run_dir / 'rankings-hotpotqa.jsonl')}
    hop = {r['id']: r for r in read_jsonl(run_dir / 'rankings-hotpotqa-enginehop.jsonl')}
    test = [q for q in questions if is_test(q.id) and q.id in hop and q.id in base]
    result: dict[str, Any] = {'held_out': len(test), 'by_kind': {}}
    groups: dict[str, list[Any]] = defaultdict(list)
    for q in test:
        groups['all'].append(q)
        groups[q.kind].append(q)
    for kind, group in groups.items():
        entry: dict[str, Any] = {'n': len(group)}
        for name, rows, key in (('scone', base, 'scone'), ('scone_hop', hop, 'enginehop'), ('llamaindex', base, 'llamaindex')):
            metrics = [score(rows[q.id][key], q.gold) for q in group]
            entry[name] = {m: sum(x[m] for x in metrics) / len(metrics) for m in ('hit@1', 'all@2', 'all@5', 'all@10')}
        for other in ('scone', 'llamaindex'):
            for k in (2, 5, 10):
                a = [score(hop[q.id]['enginehop'], q.gold)[f'all@{k}'] for q in group]
                b = [score(base[q.id][other], q.gold)[f'all@{k}'] for q in group]
                wins, losses = sum(x > y for x, y in zip(a, b)), sum(y > x for x, y in zip(a, b))
                entry[f'hop_vs_{other}_all@{k}'] = {'hop_only': wins, 'other_only': losses, 'sign_p': sign_test(wins, losses)}
        result['by_kind'][kind] = entry
    ms = sorted(r['ms'] for r in hop.values())
    if ms:
        result['latency_ms'] = {'p50': ms[len(ms) // 2], 'p95': ms[int(0.95 * (len(ms) - 1))]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['rank', 'report'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--embed-model', default='bge-base-en-v1.5')
    parser.add_argument('--embed-cache', type=Path)
    args = parser.parse_args()
    if args.stage == 'rank':
        if args.embed_cache is None:
            raise SystemExit('rank needs --embed-cache')
        asyncio.run(run(args.data_dir, args.run_dir, args.depth, args.embed_model, args.embed_cache))
    result = report(args.data_dir, args.run_dir)
    (args.run_dir / 'report-hotpotqa-enginehop.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
