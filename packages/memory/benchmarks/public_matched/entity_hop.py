"""A second hop seeded by the entities the first hop's documents name, chosen on HotpotQA's development half.

Where single-pass Scone had only one of a bridge question's two gold paragraphs in its top five (34% of the
held-out half), the reader refused 60% of the time. The missing paragraph is about an entity the first one names
("...directed by Christopher Nolan"). The text-seeded hop (question plus the leading passage) recovers some of these.
This hop searches for the names themselves:

1. take the first pass's top ``seeds`` documents;
2. from each, take up to ``names`` named entities (spaCy, precomputed into ``hotpot-doc-entities.jsonl``) whose
   words the question does not already contain;
3. search each name alone, and take its best document not already placed;
4. keep the first pass's top three documents whole, then the entity documents in order, then the rest of the first
   pass.

Every variant is scored on the development half only (``hop_rule.is_test`` false). The chosen one is then run once
on the test half. Rows keep each question's lists, so rules can be compared without searching again.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .hop_rule import is_test
from .run import FIXED_NOW, distinct, load, read_jsonl, score

KEEP = 3
WORD = re.compile(r"[a-z0-9]+")


def new_names(question: str, names: Sequence[str], limit: int) -> list[str]:
    """Names that add something to the question: at least one word of the name is not in it."""
    asked = set(WORD.findall(question.lower()))
    out = []
    for name in names:
        words = set(WORD.findall(name.lower()))
        if words and not words <= asked and name not in out:
            out.append(name)
        if len(out) == limit:
            break
    return out


def fill(first: Sequence[str], extra: Sequence[str], depth: int = 10, keep: int = KEEP) -> list[str]:
    out = list(first[:keep])
    for doc in [*extra, *first[keep:]]:
        if doc not in out:
            out.append(doc)
        if len(out) == depth:
            break
    return out


async def run(data_dir: Path, run_dir: Path, embed_model: str, embed_cache: Path, half: str, seeds: int,
              names_per_seed: int) -> None:
    from longmemeval_matched.run import _embedder
    from scone_memory.ingestion.records import Record
    from scone_memory.retrieval.fusion import PER_EPISODE_CAP
    from scone_memory.runtime.config import Settings, build_in_process_engine

    from .fast_index import install

    docs, questions = load('hotpotqa', data_dir)
    entities = {r['id']: r['entities'] for r in read_jsonl(run_dir / 'hotpot-doc-entities.jsonl')}
    first_pass = {r['id']: r['first'] for r in read_jsonl(run_dir / 'rankings-hotpotqa-twohop.jsonl')}
    out = run_dir / f'rankings-hotpotqa-entityhop-s{seeds}n{names_per_seed}-{half}.jsonl'
    done = {r['id'] for r in read_jsonl(out)}
    todo = [q for q in questions if (is_test(q.id) == (half == 'test')) and q.id not in done]
    print(f'{len(todo)} {half} questions; seeds {seeds}, names per seed {names_per_seed}', flush=True)
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
    limit = 3 * PER_EPISODE_CAP
    with out.open('a', encoding='utf-8') as handle:
        for number, question in enumerate(todo, 1):
            started = time.perf_counter()
            first = first_pass[question.id]
            searched: list[dict[str, Any]] = []
            extra: list[str] = []
            for seed in first[:seeds]:
                for name in new_names(question.question, entities.get(seed, []), names_per_seed):
                    pack = await engine.recall('bench', name, limit=limit)
                    found = distinct([i.source or '' for i in pack.items], 3)
                    best = next((d for d in found if d not in first[:KEEP] and d not in extra), None)
                    searched.append({'seed': seed, 'name': name, 'best': best})
                    if best:
                        extra.append(best)
            handle.write(json.dumps({'id': question.id, 'first': first, 'searched': searched,
                                     'entityhop': fill(first, extra),
                                     'ms': (time.perf_counter() - started) * 1000}) + '\n')
            if number % 500 == 0:
                handle.flush()
                print(f'  {number}/{len(todo)}', flush=True)


def report(data_dir: Path, run_dir: Path, half: str) -> dict[str, Any]:
    from .hop_rule import keep_then_fill

    _, questions = load('hotpotqa', data_dir)

    twohop_rows = {r['id']: r for r in read_jsonl(run_dir / 'rankings-hotpotqa-twohop.jsonl')}
    result: dict[str, Any] = {}
    for path in sorted(run_dir.glob(f'rankings-hotpotqa-entityhop-*-{half}.jsonl')):
        rows = {r['id']: r for r in read_jsonl(path)}
        group = [q for q in questions if q.id in rows]
        if not group:
            continue

        def mean(get: Any, metric: str) -> float:
            return sum(score(get(q), q.gold)[metric] for q in group) / len(group)

        arms = {
            'first pass': lambda q: rows[q.id]['first'],
            'text hop (current)': lambda q: keep_then_fill(twohop_rows[q.id]['first'], twohop_rows[q.id]['hops'][:1]),
            'entity hop': lambda q: rows[q.id]['entityhop'],
        }
        ms = sorted(rows[q.id]['ms'] for q in group)
        result[path.stem] = {
            'n': len(group),
            **{name: {m: round(100 * mean(get, m), 1) for m in ('hit@1', 'all@2', 'all@5', 'all@10')}
               for name, get in arms.items()},
            'searches_mean': sum(len(rows[q.id]['searched']) for q in group) / len(group),
            'entity_hop_ms_p50': ms[len(ms) // 2],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['run', 'report'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--embed-model', default='bge-base-en-v1.5')
    parser.add_argument('--embed-cache', type=Path)
    parser.add_argument('--half', choices=['dev', 'test'], default='dev')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--names', type=int, default=3)
    args = parser.parse_args()
    if args.stage == 'run':
        if args.embed_cache is None:
            raise SystemExit('run needs --embed-cache')
        asyncio.run(run(args.data_dir, args.run_dir, args.embed_model, args.embed_cache, args.half, args.seeds,
                        args.names))
    print(json.dumps(report(args.data_dir, args.run_dir, args.half), indent=1))


if __name__ == '__main__':
    main()
