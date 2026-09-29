"""Scone and LlamaIndex over the complete SQuAD v1.1 and HotpotQA (distractor) development sets.

Every question of a dataset searches one pooled corpus: all 2,067 SQuAD dev
paragraphs, or every paragraph any HotpotQA dev question offers (its gold
and distractor paragraphs, deduplicated). The retrieval is open over the
whole set, not over one question's own context, where SQuAD recall is
trivially complete.

Both systems see the same documents (title, a newline, the text) through the
same embedder and its cache. Scone runs its engine defaults. LlamaIndex runs
``VectorStoreIndex`` (``SentenceSplitter`` at 512 model tokens) fused with its
``BM25Retriever`` by reciprocal rank, its best configuration; each of its
retrievers returns 64 nodes, folded to documents. Scone's recall asks for
``depth * PER_EPISODE_CAP`` passages, folded the same way.

``rank`` writes ``rankings-<dataset>.jsonl`` and resumes from it (the indexes
are rebuilt from cached vectors). ``report`` scores supporting-document
recall: ``hit@k`` (any gold document in the top k), ``all@k`` (every gold
document; HotpotQA has two) and MRR@10, with paired counts and sign tests.
Gold is read only for scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, cast

DATASETS = ('squad', 'hotpotqa')
FIXED_NOW = '2026-01-01T00:00:00Z'  # every document is equally old, so recency orders nothing
LLAMA_NODES = 64


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answers: tuple[str, ...]
    gold: tuple[str, ...]  # supporting document ids
    kind: str  # 'squad', or HotpotQA's 'bridge' / 'comparison'


def _doc(title: str, body: str) -> Doc:
    text = f'{title}\n{body}'
    return Doc(hashlib.sha256(text.encode()).hexdigest()[:24], text)


def load_squad(path: Path) -> tuple[list[Doc], list[Question]]:
    docs: dict[str, Doc] = {}
    questions = []
    for article in json.loads(path.read_text(encoding='utf-8'))['data']:
        for paragraph in article['paragraphs']:
            doc = _doc(article['title'], paragraph['context'])
            docs[doc.id] = doc
            for qa in paragraph['qas']:
                answers = tuple(dict.fromkeys(a['text'] for a in qa['answers']))
                questions.append(Question('squad:' + qa['id'], qa['question'], answers, (doc.id,), 'squad'))
    return list(docs.values()), questions


def load_hotpot(path: Path) -> tuple[list[Doc], list[Question]]:
    docs: dict[str, Doc] = {}
    questions = []
    for row in json.loads(path.read_text(encoding='utf-8')):
        by_title = {}
        for title, sentences in row['context']:
            doc = _doc(title, '\n'.join(sentences))
            docs[doc.id] = doc
            by_title[title] = doc.id
        gold = tuple(sorted({by_title[title] for title, _ in row['supporting_facts']}))
        questions.append(Question('hotpotqa:' + row['_id'], row['question'], (row['answer'],), gold, row['type']))
    return list(docs.values()), questions


def load(dataset: str, data_dir: Path) -> tuple[list[Doc], list[Question]]:
    if dataset == 'squad':
        return load_squad(data_dir / 'squad_dev_v1.1.json')
    return load_hotpot(data_dir / 'hotpot_dev_distractor_v1.json')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def distinct(ranked: Sequence[str], k: int) -> list[str]:
    seen: list[str] = []
    for doc_id in ranked:
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
            if len(seen) == k:
                break
    return seen


# ---------------------------------------------------------------- rank

async def rank(dataset: str, data_dir: Path, run_dir: Path, depth: int, embed_model: str, embed_cache: Path,
               limit: int | None) -> None:
    from longmemeval_matched.run import _embedder
    from scone_memory.bench.comparative import SconeEmbedding, reference_splitter
    from scone_memory.ingestion.records import Record
    from scone_memory.retrieval.fusion import PER_EPISODE_CAP
    from scone_memory.runtime.config import Settings, build_in_process_engine

    from .fast_index import install

    docs, questions = load(dataset, data_dir)
    questions = questions[:limit] if limit else questions
    out = run_dir / f'rankings-{dataset}.jsonl'
    done = {r['id'] for r in read_jsonl(out)}
    todo = [q for q in questions if q.id not in done]
    print(f'{dataset}: {len(docs)} documents, {len(questions)} questions, {len(todo)} to rank', flush=True)
    if not todo:
        return
    embedder, cached = _embedder(embed_model, embed_cache)

    # Scone: its engine defaults over every document, with the exact fast index.
    install()
    settings = Settings.from_env({**os.environ, 'SCONE_EMBEDDER': 'local', 'SCONE_EMBED_MODEL': embed_model})
    engine = await build_in_process_engine(settings, embedder)
    engine.clock = lambda: FIXED_NOW
    started = time.perf_counter()
    for start in range(0, len(docs), 1000):
        await engine.remember_many('bench', [Record(content=d.text, source=d.id, created_at=FIXED_NOW, dedup_key=d.id)
                                             for d in docs[start:start + 1000]])
        print(f'  scone ingested {min(start + 1000, len(docs))}/{len(docs)}'
              + (f' cache {cached.record()}' if cached else ''), flush=True)
    scone_ingest_s = time.perf_counter() - started

    # LlamaIndex: one index over the same documents, vector + BM25 fused by reciprocal rank.
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
    from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import-untyped]

    started = time.perf_counter()
    documents = [Document(text=d.text, metadata={'doc_id': d.id}, excluded_embed_metadata_keys=['doc_id'],
                          excluded_llm_metadata_keys=['doc_id']) for d in docs]
    index = await asyncio.to_thread(lambda: VectorStoreIndex.from_documents(
        documents, embed_model=SconeEmbedding(embedder), show_progress=False,
        transformations=[reference_splitter(embedder, chunk_size=512, chunk_overlap=0)]))
    vector = index.as_retriever(similarity_top_k=LLAMA_NODES)
    lexical = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=LLAMA_NODES)
    fused = QueryFusionRetriever([vector, lexical], llm=MockLLM(), similarity_top_k=LLAMA_NODES, num_queries=1,
                                 mode=FUSION_MODES.RECIPROCAL_RANK, use_async=False, verbose=False)
    llama_build_s = time.perf_counter() - started
    print(f'  llamaindex built over {len(index.docstore.docs)} nodes in {llama_build_s:.0f}s', flush=True)

    with out.open('a', encoding='utf-8') as handle:
        for number, question in enumerate(todo, 1):
            started = time.perf_counter()
            pack = await engine.recall('bench', question.question, limit=depth * PER_EPISODE_CAP)
            scone_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            nodes = await asyncio.to_thread(fused.retrieve, question.question)
            llama_ms = (time.perf_counter() - started) * 1000
            handle.write(json.dumps({
                'id': question.id, 'embedder': embed_model,
                'scone': distinct([i.source or '' for i in pack.items], depth), 'scone_ms': scone_ms,
                'llamaindex': distinct([str(n.node.metadata.get('doc_id', '')) for n in nodes], depth),
                'llamaindex_ms': llama_ms, 'scone_ingest_s': scone_ingest_s, 'llamaindex_build_s': llama_build_s,
            }) + '\n')
            if number % 500 == 0:
                handle.flush()
                print(f'  ranked {number}/{len(todo)}', flush=True)


# ---------------------------------------------------------------- report

def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(min(wins, losses) + 1)) / 2 ** n)


def score(ranked: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in (1, 2, 5, 10):
        top = set(ranked[:k])
        out[f'hit@{k}'] = float(any(g in top for g in gold))
        out[f'all@{k}'] = float(all(g in top for g in gold))
    first = next((i for i, doc_id in enumerate(ranked[:10], 1) if doc_id in gold), None)
    out['mrr@10'] = 1.0 / first if first else 0.0
    return out


def report(dataset: str, data_dir: Path, run_dir: Path) -> dict[str, Any]:
    _, questions = load(dataset, data_dir)
    rows = {r['id']: r for r in read_jsonl(run_dir / f'rankings-{dataset}.jsonl')}
    scored = [q for q in questions if q.id in rows]
    result: dict[str, Any] = {'dataset': dataset, 'questions': len(questions), 'ranked': len(scored), 'by_kind': {}}
    groups: dict[str, list[Question]] = defaultdict(list)
    for q in scored:
        groups['all'].append(q)
        if dataset == 'hotpotqa':
            groups[q.kind].append(q)
    for kind, group in groups.items():
        entry: dict[str, Any] = {'n': len(group)}
        for system in ('scone', 'llamaindex'):
            metrics = [score(rows[q.id][system], q.gold) for q in group]
            entry[system] = {m: sum(x[m] for x in metrics) / len(metrics) for m in metrics[0]}
        for k in (2, 5, 10):
            a = [score(rows[q.id]['scone'], q.gold)[f'all@{k}'] for q in group]
            b = [score(rows[q.id]['llamaindex'], q.gold)[f'all@{k}'] for q in group]
            wins = sum(1 for x, y in zip(a, b) if x > y)
            losses = sum(1 for x, y in zip(a, b) if y > x)
            entry[f'paired_all@{k}'] = {'scone_only': wins, 'llamaindex_only': losses, 'sign_p': sign_test(wins, losses)}
        result['by_kind'][kind] = entry
    ms = sorted(r['scone_ms'] for r in rows.values())
    lms = sorted(r['llamaindex_ms'] for r in rows.values())
    if ms:
        result['latency_ms'] = {'scone_p50': statistics.median(ms), 'scone_p95': ms[int(0.95 * (len(ms) - 1))],
                                'llamaindex_p50': statistics.median(lms), 'llamaindex_p95': lms[int(0.95 * (len(lms) - 1))]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['rank', 'report'])
    parser.add_argument('--dataset', choices=DATASETS, action='append', required=True)
    parser.add_argument('--data-dir', type=Path, required=True, help='holds squad_dev_v1.1.json, hotpot_dev_distractor_v1.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--embed-model', default='bge-base-en-v1.5')
    parser.add_argument('--embed-cache', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='first N questions only (smoke tests)')
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset:
        if args.stage == 'rank':
            asyncio.run(rank(dataset, args.data_dir, args.run_dir, args.depth, args.embed_model, args.embed_cache,
                             args.limit))
        result = report(dataset, args.data_dir, args.run_dir)
        (args.run_dir / f'report-{dataset}.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
        print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
