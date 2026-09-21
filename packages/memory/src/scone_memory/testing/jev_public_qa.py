"""Frozen Jev comparison on downloaded public QA, using Scone's own engine.

Run and score are separate commands. Only score loads gold annotations.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .public_qa import Document, FrozenRecord, Question
from ..core.models import RecallResult
from ..providers.jev import JevRelevanceResult, JevReranker, create_reranker
from ..retrieval.reranking import RerankCandidate, RerankScore

ARMS = ('hybrid', 'hybrid_jev', 'text', 'text_jev')
SPACE = 'jev-public-qa-v1'
FILES = ('corpus.jsonl', 'reserved-queries.jsonl')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_digest() -> str:
    package = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(package)):digest(path) for path in sorted(package.rglob('*.py'))}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


class Trial(FrozenRecord):
    id: str
    dataset: str
    arm: str
    elapsed_ms: float
    error: str | None = None
    document_ids: tuple[str, ...] = ()
    recall: RecallResult | None = None
    resolved_model: str | None = None


@dataclass
class Observer:
    adapter: JevReranker
    last: JevRelevanceResult | None = None

    async def rerank(self, query: str, candidates: tuple[RerankCandidate, ...]) -> tuple[RerankScore, ...]:
        self.last = await self.adapter.evaluate(query,candidates)
        return self.last.scores


def rank_metrics(document_ids: list[str], required: set[str]) -> dict[str, float]:
    if not required:
        raise ValueError('gold must contain supporting documents')
    metrics = {}
    for k in (5,10):
        found = required.intersection(document_ids[:k])
        metrics[f'recall_at_{k}'] = len(found)/len(required)
        metrics[f'all_at_{k}'] = float(required <= found)
    metrics['mrr_at_10'] = next((1/rank for rank, doc in enumerate(document_ids[:10],1)
                               if doc in required),0.0)
    return metrics


async def run(dataset: Path, output: Path) -> None:
    from .. import HashEmbedder, MemoryEngine
    from ..backends import SqliteDocumentStore, SqliteVectorIndex
    from ..memory.engine import Record

    dataset_manifest = json.loads((dataset/'dataset.json').read_text())
    inputs = {name:digest(dataset/name) for name in FILES}
    if any(inputs[name] != dataset_manifest['files_sha256'][name] for name in FILES):
        raise ValueError('downloaded corpus or reserved questions changed')
    documents = [Document.model_validate_json(line) for line in (dataset/FILES[0]).read_text().splitlines()]
    questions = [Question.model_validate_json(line) for line in (dataset/FILES[1]).read_text().splitlines()]
    if (len(documents) != 2176 or len({q.id for q in questions}) != 200
            or Counter(q.dataset for q in questions) != {'hotpotqa':100,'squad':100}):
        raise ValueError('unexpected frozen evaluation sample')
    ranker = Observer(create_reranker())
    output.mkdir(parents=True,exist_ok=False)
    code = code_digest()
    save(output/'manifest.json',{'protocol':'jev-public-qa-v1','code_sha256':code,
        'inputs':inputs,'gold_sha256':dataset_manifest['files_sha256']['gold.jsonl'],
        'questions':[q.model_dump() for q in questions],'arms':ARMS,
        'requested_model':ranker.adapter.model,'embedder':'HashEmbedder','chunk_target':700,
        'candidate_limit':64,'rerank_limit':32,'rerank_max_bytes':64000,'limit':10})
    database = output/'memory.db'
    engine = await MemoryEngine(SqliteDocumentStore(database),SqliteVectorIndex(database),HashEmbedder(),
        chunk_target=700,reranker=ranker,rerank_limit=32,rerank_max_bytes=64000,rerank_timeout=10).open()
    episode_documents: dict[int,str] = {}
    try:
        for offset in range(0,len(documents),100):
            batch = documents[offset:offset+100]
            episodes = await engine.remember_many(SPACE,[Record(doc.content,kind='file',source=doc.source_url,
                metadata={'document_id':doc.id},created_at='2026-09-08') for doc in batch])
            for episode, doc in zip(episodes,batch,strict=True):
                if episode.episode_id in episode_documents:
                    raise ValueError('unexpected corpus deduplication')
                episode_documents[episode.episode_id] = doc.id
            print(f'Indexed {len(episode_documents)}/{len(documents)}',flush=True)
        with (output/'observations.jsonl').open('x') as stream:
            for index, question in enumerate(questions):
                arms = ARMS[index%4:]+ARMS[:index%4]
                for arm in arms:
                    ranker.last = None
                    started = time.monotonic()
                    error = None
                    result = None
                    try:
                        async with asyncio.timeout(30):
                            result = await engine.recall(SPACE,question.question,limit=10,candidate_limit=64,
                                rerank=arm.endswith('_jev'),lanes=('text',) if arm.startswith('text') else ('vector','text'))
                    except Exception as failure:
                        error = type(failure).__name__
                    row = Trial(id=question.id,dataset=question.dataset,arm=arm,
                        elapsed_ms=(time.monotonic()-started)*1000,error=error,recall=result,
                        document_ids=tuple(episode_documents[item.episode_id] for item in result.items) if result else (),
                        resolved_model=ranker.last.model if ranker.last else None)
                    stream.write(row.model_dump_json()+'\n')
                    stream.flush()
                print(f'Evaluated {index+1}/{len(questions)} questions across four Scone arms',flush=True)
    finally:
        await engine.close()
    unchanged = code_digest()==code and all(digest(dataset/name)==value for name,value in inputs.items())
    save(output/'completion.json',{'terminal':True,'code_and_inputs_unchanged':unchanged,
        'observations_sha256':digest(output/'observations.jsonl'),
        'manifest_sha256':digest(output/'manifest.json')})
    if not unchanged:
        raise ValueError('code or source data changed during evaluation')


def score(dataset: Path, output: Path) -> dict[str, object]:
    from .public_qa import Gold

    manifest = json.loads((output/'manifest.json').read_text())
    completed = json.loads((output/'completion.json').read_text())
    if (not completed['terminal'] or not completed['code_and_inputs_unchanged']
            or completed['observations_sha256'] != digest(output/'observations.jsonl')
            or completed['manifest_sha256'] != digest(output/'manifest.json')
            or manifest['gold_sha256'] != digest(dataset/'gold.jsonl')
            or any(digest(dataset/name) != sha for name,sha in manifest['inputs'].items())):
        raise ValueError('evaluation integrity check failed')
    rows = [Trial.model_validate_json(line) for line in (output/'observations.jsonl').read_text().splitlines()]
    expected = {(q['id'],arm) for q in manifest['questions'] for arm in ARMS}
    if len(rows) != 800 or {(row.id,row.arm) for row in rows} != expected:
        raise ValueError('incomplete or duplicated evaluation schedule')
    gold = {item.id:item for line in (dataset/'gold.jsonl').read_text().splitlines()
            for item in (Gold.model_validate_json(line),)}
    measurements = {(row.id,row.arm):rank_metrics(list(row.document_ids),set(gold[row.id].support_documents))
                    for row in rows}
    groups = []
    for group in ('all','hotpotqa','squad'):
        for arm in ARMS:
            selected = [row for row in rows if row.arm==arm and (group=='all' or row.dataset==group)]
            latencies = sorted(row.elapsed_ms for row in selected)
            groups.append({'dataset':group,'arm':arm,'n':len(selected),
                **{metric:statistics.mean(measurements[row.id,row.arm][metric] for row in selected)
                   for metric in next(iter(measurements.values()))},
                'p50_ms':statistics.median(latencies),'p95_ms':latencies[math.ceil(.95*len(latencies))-1],
                'failures':sum(row.error is not None for row in selected),
                'rerank_applied':sum(row.recall is not None and row.recall.rerank is not None
                                     and row.recall.rerank.status=='applied' for row in selected),
                'rerank_failed':sum(row.recall is not None and row.recall.rerank is not None
                                    and row.recall.rerank.status=='failed' for row in selected)})
    pairs = []
    for baseline in ('hybrid','text'):
        for metric in next(iter(measurements.values())):
            deltas = [measurements[q['id'],baseline+'_jev'][metric]-measurements[q['id'],baseline][metric]
                      for q in manifest['questions']]
            pairs.append({'baseline':baseline,'metric':metric,'wins':sum(d>0 for d in deltas),
                          'losses':sum(d<0 for d in deltas),'ties':sum(d==0 for d in deltas),
                          'mean_delta':statistics.mean(deltas)})
    report = {'groups':groups,'paired':pairs,'resolved_models':sorted({row.resolved_model for row in rows if row.resolved_model}),
              'per_question':[{'id':row.id,'dataset':row.dataset,'arm':row.arm,
                               **measurements[row.id,row.arm]} for row in rows]}
    save(output/'scores.json',report)
    print(json.dumps({'groups':groups,'paired':pairs,'resolved_models':report['resolved_models']},indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','score'))
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='run':
        asyncio.run(run(args.dataset,args.output))
    else:
        score(args.dataset,args.output)


if __name__=='__main__':
    main()
