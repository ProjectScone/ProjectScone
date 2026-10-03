"""A second hop for Scone without a model: search again from what the first search found.

A HotpotQA bridge question names one thing while its second supporting
paragraph is about another, which the first paragraph names ("the director
of film X was born where?"). Both systems find the first paragraph at rank 1
for 86% of bridge questions, yet both gold paragraphs reach the top 5 for only
59%. Multi-hop dense retrieval (MDR) searches again with the question joined
to a retrieved passage, so the bridge entity is in the query. This runs that
without training or a model:

1. Scone's recall for the question, folded to documents (the first pass);
2. for each of the first ``SEEDS`` documents, a recall for the question, a
   newline and as much of that document as fits in the engine's query limit (``MAX_QUERY``, 1,000
   characters; the question always whole);
3. the lists fused by reciprocal rank (k = 60, equal weights), first pass
   included, so a first-pass document that also leads a hop stays on top.

The parameters were fixed before any result was seen and are not tuned on
the data this reports. Writes ``rankings-<dataset>-twohop.jsonl``, which
``report_twohop`` scores against the first pass and LlamaIndex from the
main run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .run import FIXED_NOW, distinct, load, read_jsonl, score, sign_test

SEEDS = 2
RRF_K = 60


def fuse(lists: Sequence[Sequence[str]], depth: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    for list_index, ranked in enumerate(lists):
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] += 1.0 / (RRF_K + rank)
            first_seen.setdefault(doc_id, (rank, list_index))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))[:depth]


async def run(dataset: str, data_dir: Path, run_dir: Path, depth: int, embed_model: str, embed_cache: Path) -> None:
    from longmemeval_matched.run import _embedder
    from scone_memory.core.validation import MAX_QUERY
    from scone_memory.ingestion.records import Record
    from scone_memory.retrieval.fusion import PER_EPISODE_CAP
    from scone_memory.runtime.config import Settings, build_in_process_engine

    from .fast_index import install

    docs, questions = load(dataset, data_dir)
    text_of = {d.id: d.text for d in docs}
    out = run_dir / f'rankings-{dataset}-twohop.jsonl'
    done = {r['id'] for r in read_jsonl(out)}
    todo = [q for q in questions if q.id not in done]
    print(f'{dataset}: {len(todo)} questions to rank with a second hop', flush=True)
    if not todo:
        return
    embedder, cached = _embedder(embed_model, embed_cache)
    install()
    settings = Settings.from_env({**os.environ, 'SCONE_EMBEDDER': 'local', 'SCONE_EMBED_MODEL': embed_model})
    engine = await build_in_process_engine(settings, embedder)
    engine.clock = lambda: FIXED_NOW
    started = time.perf_counter()
    for start in range(0, len(docs), 2000):
        await engine.remember_many('bench', [Record(content=d.text, source=d.id, created_at=FIXED_NOW, dedup_key=d.id)
                                             for d in docs[start:start + 2000]])
    print(f'  ingested in {time.perf_counter() - started:.0f}s' + (f' cache {cached.record()}' if cached else ''),
          flush=True)
    limit = depth * PER_EPISODE_CAP

    async def recall(query: str) -> list[str]:
        pack = await engine.recall('bench', query, limit=limit)
        return distinct([i.source or '' for i in pack.items], depth)

    with out.open('a', encoding='utf-8') as handle:
        for number, question in enumerate(todo, 1):
            started = time.perf_counter()
            first = await recall(question.question)
            room = MAX_QUERY - len(question.question) - 1
            hops = [await recall(question.question + '\n' + text_of[seed][:room]) if room > 0 else []
                    for seed in first[:SEEDS]]
            handle.write(json.dumps({'id': question.id, 'first': first, 'hops': hops,
                                     'twohop': fuse([first, *hops], depth),
                                     'ms': (time.perf_counter() - started) * 1000}) + '\n')
            if number % 500 == 0:
                handle.flush()
                print(f'  ranked {number}/{len(todo)}', flush=True)


def report_twohop(dataset: str, data_dir: Path, run_dir: Path) -> dict[str, Any]:
    _, questions = load(dataset, data_dir)
    base = {r['id']: r for r in read_jsonl(run_dir / f'rankings-{dataset}.jsonl')}
    hop = {r['id']: r for r in read_jsonl(run_dir / f'rankings-{dataset}-twohop.jsonl')}
    scored = [q for q in questions if q.id in base and q.id in hop]
    same_first = sum(hop[q.id]['first'] == base[q.id]['scone'] for q in scored)
    groups: dict[str, list[Any]] = defaultdict(list)
    for q in scored:
        groups['all'].append(q)
        groups[q.kind].append(q)
    result: dict[str, Any] = {'dataset': dataset, 'scored': len(scored), 'first_pass_matches_main_run': same_first,
                              'seeds': SEEDS, 'seed_text': 'up to the engine query limit', 'by_kind': {}}
    for kind, group in groups.items():
        entry: dict[str, Any] = {'n': len(group)}
        lists: dict[str, Callable[[Any], list[str]]] = {
            'scone': lambda q: base[q.id]['scone'], 'scone_twohop': lambda q: hop[q.id]['twohop'],
            'llamaindex': lambda q: base[q.id]['llamaindex']}
        for name, get in lists.items():
            metrics = [score(get(q), q.gold) for q in group]
            entry[name] = {m: sum(x[m] for x in metrics) / len(metrics) for m in ('hit@1', 'all@2', 'all@5', 'all@10', 'mrr@10')}
        for other in ('scone', 'llamaindex'):
            for k in (2, 5, 10):
                a = [score(hop[q.id]['twohop'], q.gold)[f'all@{k}'] for q in group]
                b = [score(lists[other](q), q.gold)[f'all@{k}'] for q in group]
                wins, losses = sum(x > y for x, y in zip(a, b)), sum(y > x for x, y in zip(a, b))
                entry[f'twohop_vs_{other}_all@{k}'] = {'twohop_only': wins, 'other_only': losses,
                                                       'sign_p': sign_test(wins, losses)}
        result['by_kind'][kind] = entry
    ms = sorted(hop[q.id]['ms'] for q in scored)
    if ms:
        result['latency_ms'] = {'p50': ms[len(ms) // 2], 'p95': ms[int(0.95 * (len(ms) - 1))]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['rank', 'report'])
    parser.add_argument('--dataset', default='hotpotqa')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--embed-model', default='bge-base-en-v1.5')
    parser.add_argument('--embed-cache', type=Path)
    args = parser.parse_args()
    if args.stage == 'rank':
        if args.embed_cache is None:
            raise SystemExit('rank needs --embed-cache')
        asyncio.run(run(args.dataset, args.data_dir, args.run_dir, args.depth, args.embed_model, args.embed_cache))
    result = report_twohop(args.dataset, args.data_dir, args.run_dir)
    (args.run_dir / f'report-{args.dataset}-twohop.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
