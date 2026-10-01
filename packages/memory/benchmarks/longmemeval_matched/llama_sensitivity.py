"""Was LlamaIndex handicapped? Its ranking under its own defaults and under overlap, against the saved runs.

The matched runs split LlamaIndex's documents with ``SentenceSplitter`` at 512
model tokens and no overlap (the comparison harness's choice, so no node runs
past bge's 512-token window). LlamaIndex's own default is 1,024 tokens with
200 of overlap, and zero overlap can cut an answer in two across long chat
sessions. This re-ranks the items of a saved run's ``rankings.jsonl`` with:

- ``llama_512_0_repro``: the saved configuration again, which must reproduce the saved lists;
- ``llama_512_200``: the saved splitter with LlamaIndex's default overlap;
- ``llama_default_hybrid``: LlamaIndex's own ``SentenceSplitter()`` (1,024 tiktoken tokens, 200 overlap), still fused
  with BM25;
- ``llama_default_vector``: its default splitter and plain vector retrieval, the out-of-the-box setup.

Every variant uses the saved run's embedder, and every node is ranked before folding to sessions, as in the saved
runs. The report sets each variant beside Scone's and LlamaIndex's saved lists (all evidence sessions in the top k).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from math import comb
from pathlib import Path
from collections.abc import Callable
from typing import Any

from .run import load, read_jsonl

VARIANTS = ('llama_512_0_repro', 'llama_512_200', 'llama_default_hybrid', 'llama_default_vector')


async def ranking(bench_item: Any, embedder: Any, variant: str, depth: int) -> list[str]:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.llms import MockLLM
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
    from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import-untyped]
    from scone_memory.bench.comparative import SconeEmbedding, _documents, reference_splitter

    documents = [Document(text=text, metadata={'session_id': sid}, excluded_embed_metadata_keys=['session_id'],
                          excluded_llm_metadata_keys=['session_id']) for sid, text in _documents(bench_item)]
    if variant == 'llama_512_0_repro':  # the saved configuration again: proves the re-ranking reproduces it
        splitter = reference_splitter(embedder, chunk_size=512, chunk_overlap=0)
    elif variant == 'llama_512_200':
        splitter = reference_splitter(embedder, chunk_size=512, chunk_overlap=200)
    else:
        splitter = SentenceSplitter()

    def build_and_retrieve() -> list[Any]:
        index = VectorStoreIndex.from_documents(documents, embed_model=SconeEmbedding(embedder), show_progress=False,
                                                transformations=[splitter])
        every = max(1, len(index.docstore.docs))
        vector = index.as_retriever(similarity_top_k=every)
        if variant == 'llama_default_vector':
            return list(vector.retrieve(bench_item.question))
        lexical = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=every)
        fused = QueryFusionRetriever([vector, lexical], llm=MockLLM(), similarity_top_k=every, num_queries=1,
                                     mode=FUSION_MODES.RECIPROCAL_RANK, use_async=False, verbose=False)
        return list(fused.retrieve(bench_item.question))

    ranked: list[str] = []
    for node in await asyncio.to_thread(build_and_retrieve):
        sid = node.node.metadata.get('session_id')
        if isinstance(sid, str) and sid not in ranked:
            ranked.append(sid)
            if len(ranked) == depth:
                break
    return ranked


async def rank(dataset: Path, run_dir: Path, embed_model: str, embed_cache: Path | None, depth: int) -> None:
    from .run import _bench_item, _embedder

    saved = {r['question_id'] for r in read_jsonl(run_dir / 'rankings.jsonl')}
    items = [i for i in load(dataset, 0, 42) if i.question_id in saved]
    out = run_dir / 'rankings-llama-sensitivity.jsonl'
    done = {r['question_id'] for r in read_jsonl(out)}
    embedder, cached = _embedder(embed_model, embed_cache)
    for number, item in enumerate((i for i in items if i.question_id not in done), 1):
        bench_item = _bench_item(item.raw, with_sessions=True)
        row: dict[str, Any] = {'question_id': item.question_id}
        for variant in VARIANTS:
            row[variant] = await ranking(bench_item, embedder, variant, depth)
        with out.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(f'ranked {number}: {item.question_id}' + (f' cache {cached.record()}' if cached else ''), flush=True)


def _getter(rows: dict[str, dict[str, Any]], key: str) -> Callable[[str], list[str]]:
    return lambda q: list(rows[q][key])


def report(dataset: Path, run_dir: Path) -> dict[str, Any]:
    saved: dict[str, dict[str, Any]] = {str(r['question_id']): r for r in read_jsonl(run_dir / 'rankings.jsonl')}
    extra: dict[str, dict[str, Any]] = {str(r['question_id']): r
                                        for r in read_jsonl(run_dir / 'rankings-llama-sensitivity.jsonl')}
    items = [i for i in load(dataset, 0, 42) if i.question_id in saved and i.question_id in extra
             and i.evidence and '_abs' not in i.question_id]
    lists: dict[str, Callable[[str], list[str]]] = {'scone (saved)': _getter(saved, 'scone'), 'llamaindex 512/0 hybrid (saved)': _getter(saved, 'llamaindex'),
             **{v: _getter(extra, v) for v in VARIANTS}}
    repro = sum(extra[i.question_id]['llama_512_0_repro'] == saved[i.question_id]['llamaindex'] for i in items)
    out: dict[str, Any] = {'items': len(items), 'repro_identical_to_saved': repro, 'arms': {}}
    for k in (5, 10):
        for name, get in lists.items():
            hits = [set(i.evidence) <= set(get(i.question_id)[:k]) for i in items]
            out['arms'].setdefault(name, {})[f'all@{k}'] = sum(hits) / len(hits)
        scone = [set(i.evidence) <= set(saved[i.question_id]['scone'][:k]) for i in items]
        for name, get in lists.items():
            if name.startswith('scone'):
                continue
            other = [set(i.evidence) <= set(get(i.question_id)[:k]) for i in items]
            wins = sum(a and not b for a, b in zip(scone, other))
            losses = sum(b and not a for a, b in zip(scone, other))
            n = wins + losses
            p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(min(wins, losses) + 1)) / 2 ** n)
            out['arms'][name][f'scone_vs_this_all@{k}'] = {'scone_only': wins, 'this_only': losses, 'sign_p': p}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['rank', 'report'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--embed-model', required=True)
    parser.add_argument('--embed-cache', type=Path)
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()
    if args.stage == 'rank':
        asyncio.run(rank(args.dataset, args.run_dir, args.embed_model, args.embed_cache, args.depth))
    result = report(args.dataset, args.run_dir)
    (args.run_dir / 'report-llama-sensitivity.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
