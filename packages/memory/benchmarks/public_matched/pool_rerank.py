"""Hops find the candidates; a small cross-encoder orders them. Chosen on HotpotQA's development half.

On the development half, the first pass's top ten, the text hop's top ten and the entity-name searches together hold
both gold paragraphs for 90.5% of questions. Merging them by rule puts both in the top five for only 71%, so the
ordering is now the loss. This reranks that pool, about 20 to 25 short paragraphs, with a local ONNX cross-encoder
(``OfflineCrossEncoderReranker``), and records the time each question's reranking takes.

It reads the saved lists (``rankings-hotpotqa-twohop.jsonl`` and the entity-hop rows), so no search runs again.
Rows go to ``rankings-hotpotqa-poolrerank-<model>-<half>.jsonl``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .hop_rule import is_test
from .run import load, read_jsonl, score

MODELS = {
    'minilm': ('ms-marco-MiniLM-L-6-v2', 'Xenova/ms-marco-MiniLM-L-6-v2'),
    'bge': ('bge-reranker-base', 'BAAI/bge-reranker-base'),
}


def pool(first: list[str], text_hop: list[str], entity_bests: list[str]) -> list[str]:
    out: list[str] = []
    for doc in [*first[:10], *text_hop[:10], *entity_bests]:
        if doc and doc not in out:
            out.append(doc)
    return out


async def scored(reranker: Any, query: str, ids: list[str], text: dict[str, str],
                 cuts: tuple[int, ...]) -> tuple[dict[int, float], int]:
    """Scores by candidate number (1-based: the reranker refuses chunk id 0). A query and a paragraph must fit the
    model's 512-token pair window or the whole request is refused, so paragraphs are cut, shorter on each refusal."""
    from scone_memory.retrieval.reranking import RerankCandidate

    for cut in cuts:
        candidates = tuple(RerankCandidate(chunk_id=i + 1, episode_id=i + 1, text=text[doc][:cut], source=doc,
                                           created_at='2026-01-01T00:00:00Z', baseline_score=0.0, similarity=None,
                                           lanes=())
                           for i, doc in enumerate(ids))
        try:
            found = await reranker.rerank(query, candidates)
        except ValueError:
            continue
        by = {s.chunk_id: s.score for s in found}
        if len(by) != len(ids):
            raise RuntimeError(f'the reranker scored {len(by)} of {len(ids)} candidates')
        return by, cut
    raise RuntimeError('no cut fit the reranker')


async def run(data_dir: Path, run_dir: Path, models_dir: Path, model: str, half: str, threads: int,
              mode: str = 'plain') -> None:
    from scone_memory.providers.offline_reranker import OfflineCrossEncoderReranker

    docs, questions = load('hotpotqa', data_dir)
    text = {d.id: d.text for d in docs}
    twohop = {r['id']: r for r in read_jsonl(run_dir / 'rankings-hotpotqa-twohop.jsonl')}
    entity = {r['id']: r for r in read_jsonl(run_dir / f'rankings-hotpotqa-entityhop-s2n3-{half}.jsonl')}
    out = run_dir / f"rankings-hotpotqa-poolrerank-{model}-{half}{'' if mode == 'plain' else '-' + mode}.jsonl"
    done = {r['id'] for r in read_jsonl(out)}
    todo = [q for q in questions if (is_test(q.id) == (half == 'test')) and q.id in entity and q.id not in done]
    folder, name = MODELS[model]
    reranker = OfflineCrossEncoderReranker(models_dir / folder, model_name=name, threads=threads, batch_size=32)
    print(f'{len(todo)} {half} questions, {model}, {threads} threads', flush=True)
    with out.open('a', encoding='utf-8') as handle:
        for number, question in enumerate(todo, 1):
            row = twohop[question.id]
            candidates_ids = pool(row['first'], row['hops'][0], [s['best'] for s in entity[question.id]['searched']])
            started = time.perf_counter()
            by, cut_used = await scored(reranker, question.question, candidates_ids, text, (1500, 1000, 600))
            ranked = [candidates_ids[i] for i in sorted(range(len(candidates_ids)), key=lambda i: -by[i + 1])]
            extra: dict[str, Any] = {}
            if mode == 'bridge':
                # The leader answers the question's first hop; the bridge paragraph is relevant to the question
                # *with* the leader's text, which names it. Rescore the rest against that.
                lead = ranked[0]
                rest = [d for d in candidates_ids if d != lead]
                hop_query = question.question + '\n' + text[lead][:700]
                by2, cut2 = await scored(reranker, hop_query, rest, text, (800, 500, 300))
                bridge = [rest[i] for i in sorted(range(len(rest)), key=lambda i: -by2[i + 1])]
                extra = {'bridge': [lead, *bridge], 'bridge_cut': cut2}
            ms = (time.perf_counter() - started) * 1000
            handle.write(json.dumps({'id': question.id, 'pool': candidates_ids, 'reranked': ranked, 'ms': ms,
                                     'cut': cut_used, **extra}) + '\n')
            if number % 500 == 0:
                handle.flush()
                print(f'  {number}/{len(todo)}', flush=True)


def report(data_dir: Path, run_dir: Path, half: str) -> dict[str, Any]:
    from .hop_rule import keep_then_fill

    _, questions = load('hotpotqa', data_dir)
    twohop = {r['id']: r for r in read_jsonl(run_dir / 'rankings-hotpotqa-twohop.jsonl')}
    result: dict[str, Any] = {}
    for model, suffix in [(m, x) for m in MODELS for x in ('', '-bridge')]:
        rows = {r['id']: r for r in read_jsonl(run_dir / f'rankings-hotpotqa-poolrerank-{model}-{half}{suffix}.jsonl')}
        group = [q for q in questions if q.id in rows]
        if not group:
            continue
        arms: dict[str, Any] = {
            'first pass': lambda q: twohop[q.id]['first'],
            'text hop (current)': lambda q: keep_then_fill(twohop[q.id]['first'], twohop[q.id]['hops'][:1]),
            'pool reranked': lambda q, rows=rows: rows[q.id]['reranked'],
            'pool reranked, first-pass leader kept': lambda q, rows=rows: (
                [twohop[q.id]['first'][0]] + [d for d in rows[q.id]['reranked'] if d != twohop[q.id]['first'][0]]),
        }
        if suffix:
            arms['bridge-aware rerank'] = lambda q, rows=rows: rows[q.id]['bridge']
        ms = sorted(rows[q.id]['ms'] for q in group)
        entry: dict[str, Any] = {'n': len(group), 'pool_size_mean': sum(len(rows[q.id]['pool']) for q in group) / len(group),
                                 'pool_holds_both': round(100 * sum(set(q.gold) <= set(rows[q.id]['pool']) for q in group) / len(group), 1),
                                 'rerank_ms_p50': ms[len(ms) // 2], 'rerank_ms_p95': ms[int(0.95 * (len(ms) - 1))]}
        for name, get in arms.items():
            metrics = [score(get(q), q.gold) for q in group]
            entry[name] = {m: round(100 * sum(x[m] for x in metrics) / len(group), 1) for m in ('hit@1', 'all@2', 'all@5', 'all@10')}
        result[model + suffix] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['run', 'report'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--model', choices=sorted(MODELS), default='minilm')
    parser.add_argument('--half', choices=['dev', 'test'], default='dev')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--mode', choices=['plain', 'bridge'], default='plain')
    args = parser.parse_args()
    if args.stage == 'run':
        if args.models_dir is None:
            raise SystemExit('run needs --models-dir')
        asyncio.run(run(args.data_dir, args.run_dir, args.models_dir, args.model, args.half, args.threads, args.mode))
    print(json.dumps(report(args.data_dir, args.run_dir, args.half), indent=1))


if __name__ == '__main__':
    main()
