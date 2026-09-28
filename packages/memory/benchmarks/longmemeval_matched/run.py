"""Matched-reader LongMemEval: every arm hands its sessions to the same reader,
and the same judge scores every answer with LongMemEval's own prompts.

Stages, each resumable from its own JSONL file in the run directory:
  rank    Scone and LlamaIndex rank each item's sessions with one shared embedder
  answer  each arm's sessions -> the official reader prompt -> the reader model
  judge   each answer -> the official judge prompt -> the judge model
  report  accuracy per arm and question type, Wilson intervals, paired counts

Arms: ``full`` (every session: the full-context baseline), ``oracle`` (only the
evidence sessions: the upper bound), ``scone@k`` and ``llamaindex@k`` (the top
k distinct sessions each system ranks). A failed request is kept and scored
as incorrect; nothing is dropped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx

if TYPE_CHECKING:
    from scone_memory.bench.runner import BenchItem

from .prompts import Session, Turn, UPSTREAM_COMMIT, is_abstention, judge_prompt, judged_correct, reader_prompt

OPENROUTER = 'https://openrouter.ai/api/v1/chat/completions'
READER_MODEL = 'google/gemma-4-31b-it'
JUDGE_MODEL = 'openai/gpt-4o-2024-08-06'  # the model upstream's judge names
RETRIES = 3


class Item:
    def __init__(self, raw: dict[str, object]) -> None:
        self.question_id = str(raw['question_id'])
        self.question_type = str(raw['question_type'])
        self.question = str(raw['question'])
        self.answer = str(raw['answer'])
        self.question_date = str(raw['question_date'])
        ids = cast(list[str], raw['haystack_session_ids'])
        dates = cast(list[str], raw['haystack_dates'])
        bodies = cast(list[list[dict[str, str]]], raw['haystack_sessions'])
        self.sessions = {sid: Session(sid, date, tuple(Turn(t['role'], t['content']) for t in body))
                         for sid, date, body in zip(ids, dates, bodies, strict=True)}
        self.evidence = [str(s) for s in cast(list[str], raw.get('answer_session_ids', []))]
        self.raw = raw


def _bench_item(raw: dict[str, object], *, with_sessions: bool) -> BenchItem:
    """``scone_memory.bench.runner.load_items``'s conversion of one item. Without
    sessions it is only good for sampling, which reads ids and types."""
    from scone_memory.bench.runner import BenchItem, iso_date

    bodies = cast(list[list[dict[str, str]]], raw['haystack_sessions']) if with_sessions else []
    return BenchItem(
        question_id=str(raw['question_id']), question_type=str(raw['question_type']),
        question=str(raw['question']), question_date=str(raw['question_date']),
        sessions=tuple(tuple(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in body) for body in bodies),
        session_ids=tuple(str(x) for x in cast(list[str], raw['haystack_session_ids'])) if with_sessions else (),
        session_dates=tuple(iso_date(str(d)) for d in cast(list[str], raw['haystack_dates'])) if with_sessions else (),
        answer_session_ids=tuple(str(x) for x in cast(list[str], raw.get('answer_session_ids', []))),
    )


def load(dataset: Path, sample: int, seed: int) -> list[Item]:
    """Parses the dataset once and keeps only the sampled items: LongMemEval-M is
    2.7 GB of JSON, and a second parse or a copy of every session would not fit."""
    from scone_memory.bench.runner import stratified_sample

    raw = cast(list[dict[str, object]], json.loads(dataset.read_text(encoding='utf-8')))
    stubs = [_bench_item(r, with_sessions=False) for r in raw]
    chosen = stubs if sample <= 0 else stratified_sample(stubs, sample, seed)
    by_id = {str(r['question_id']): r for r in raw}
    del raw
    return [Item(by_id[b.question_id]) for b in chosen]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def append(path: Path, row: dict[str, object]) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------- rank

async def rank(items: Sequence[Item], run_dir: Path, depth: int, embed_model: str, embed_cache: Path | None) -> None:
    from scone_memory.bench.comparative import (CachedEmbedder, OneThreadCache, bench_embedder, distinct_sessions,
                                                llamaindex_session_ranking)
    from scone_memory.bench.runner import run as run_bench
    from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
    from scone_memory.retrieval.fusion import PER_EPISODE_CAP
    from scone_memory.runtime.config import Settings, build_in_process_engine

    out = run_dir / 'rankings.jsonl'
    done = {str(r['question_id']) for r in read_jsonl(out)}
    wanted = {item.question_id for item in items if item.question_id not in done}
    if not wanted:
        return
    model = bench_embedder(embed_model)
    cached: CachedEmbedder | None = None
    if embed_cache is not None:
        # Sized to hold every chunk of a run: an evicting cache would re-embed and misreport the cost.
        path = str(embed_cache)
        cached = CachedEmbedder(model, OneThreadCache(lambda: SqliteEmbeddingCache(path, max_entries=50_000_000)))
    embedder = cached or model
    settings = Settings.from_env({**os.environ, 'SCONE_EMBEDDER': 'local', 'SCONE_EMBED_MODEL': embed_model})

    async def make() -> object:
        return await build_in_process_engine(settings, embedder)

    for bench_item in (_bench_item(i.raw, with_sessions=True) for i in items if i.question_id in wanted):
        started = time.perf_counter()
        report = await run_bench(make, [bench_item], ks=(depth,), limit=depth * PER_EPISODE_CAP)  # type: ignore[arg-type]
        scone_ms = (time.perf_counter() - started) * 1000
        result = report.results[0]
        started = time.perf_counter()
        llama = await llamaindex_session_ranking(bench_item, embedder, k=depth, hybrid=True)
        llama_ms = (time.perf_counter() - started) * 1000
        append(out, {'question_id': bench_item.question_id, 'embedder': embed_model,
                     'scone': list(distinct_sessions(result.retrieved_sessions, depth)), 'scone_ms': scone_ms,
                     'scone_recall_ms': result.recall_ms, 'llamaindex': llama, 'llamaindex_ms': llama_ms})
        print(f'ranked {bench_item.question_id}' + (f' cache {cached.record()}' if cached else ''), flush=True)


# ---------------------------------------------------------------- answer / judge

async def chat(client: httpx.AsyncClient, key: str, model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    payload: dict[str, object] = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                                  'temperature': 0, 'max_tokens': max_tokens, 'stream': False}
    if model == READER_MODEL:
        payload['reasoning'] = {'effort': 'none'}
    error = ''
    started = time.perf_counter()
    for attempt in range(RETRIES):
        try:
            response = await client.post(OPENROUTER, json=payload, headers={'Authorization': 'Bearer ' + key}, timeout=180)
            response.raise_for_status()
            body = response.json()
            choice = body['choices'][0]
            text = (choice['message'].get('content') or '').strip()
            complete = choice.get('finish_reason') == 'stop' and bool(text)
            return {'completed': complete, 'text': text, 'error': None if complete else 'incomplete_generation',
                    'usage': body.get('usage'), 'ms': (time.perf_counter() - started) * 1000, 'attempts': attempt + 1}
        except (httpx.HTTPError, KeyError, ValueError) as caught:
            error = type(caught).__name__ + (f'_{caught.response.status_code}' if isinstance(caught, httpx.HTTPStatusError) else '')
            await asyncio.sleep(2 ** attempt)
    return {'completed': False, 'text': '', 'error': error, 'usage': None,
            'ms': (time.perf_counter() - started) * 1000, 'attempts': RETRIES}


def arm_sessions(item: Item, arm: str, rankings: dict[str, dict[str, object]]) -> list[Session]:
    if arm == 'full':
        return list(item.sessions.values())
    if arm == 'oracle':
        return [item.sessions[s] for s in item.evidence if s in item.sessions]
    system, _, k = arm.partition('@')
    ranked = cast(list[str], rankings[item.question_id][system])
    return [item.sessions[s] for s in ranked[:int(k)]]


async def gather_bounded(jobs: Sequence[Callable[[], Awaitable[None]]], concurrency: int) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def bounded(job: Callable[[], Awaitable[None]]) -> None:
        async with gate:
            await job()
    await asyncio.gather(*(bounded(job) for job in jobs))


async def answer(items: Sequence[Item], arms: Sequence[str], run_dir: Path, key: str, cot: bool, concurrency: int) -> None:
    out = run_dir / 'answers.jsonl'
    done = {(str(r['arm']), str(r['question_id'])) for r in read_jsonl(out)}
    rankings = {str(r['question_id']): r for r in read_jsonl(run_dir / 'rankings.jsonl')}
    async with httpx.AsyncClient() as client:
        def job(item: Item, arm: str) -> Callable[[], Awaitable[None]]:
            async def go() -> None:
                sessions = arm_sessions(item, arm, rankings)
                prompt = reader_prompt(sessions, item.question_date, item.question, cot=cot)
                result = await chat(client, key, READER_MODEL, prompt, 800 if cot else 500)
                append(out, {'arm': arm, 'question_id': item.question_id, 'sessions': [s.session_id for s in sessions],
                             'prompt_chars': len(prompt), **result})
            return go
        await gather_bounded([job(i, a) for a in arms for i in items if (a, i.question_id) not in done], concurrency)


async def judge(items: Sequence[Item], run_dir: Path, key: str, concurrency: int) -> None:
    out = run_dir / 'judgments.jsonl'
    done = {(str(r['arm']), str(r['question_id'])) for r in read_jsonl(out)}
    by_id = {item.question_id: item for item in items}
    answers = [a for a in read_jsonl(run_dir / 'answers.jsonl')
               if a['completed'] and str(a['question_id']) in by_id and (str(a['arm']), str(a['question_id'])) not in done]
    async with httpx.AsyncClient() as client:
        def job(row: dict[str, object]) -> Callable[[], Awaitable[None]]:
            async def go() -> None:
                item = by_id[str(row['question_id'])]
                prompt = judge_prompt(item.question_type, item.question_id, item.question, item.answer, str(row['text']))
                result = await chat(client, key, JUDGE_MODEL, prompt, 10)
                append(out, {'arm': row['arm'], 'question_id': item.question_id, 'verdict': result['text'],
                             'correct': bool(result['completed']) and judged_correct(str(result['text'])),
                             'judge_error': result['error']})
            return go
        await gather_bounded([job(a) for a in answers], concurrency)


# ---------------------------------------------------------------- report

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def report(items: Sequence[Item], arms: Sequence[str], run_dir: Path) -> dict[str, object]:
    answers = {(str(r['arm']), str(r['question_id'])): r for r in read_jsonl(run_dir / 'answers.jsonl')}
    verdicts = {(str(r['arm']), str(r['question_id'])): r for r in read_jsonl(run_dir / 'judgments.jsonl')}
    rankings = {str(r['question_id']): r for r in read_jsonl(run_dir / 'rankings.jsonl')}

    def correct(arm: str, qid: str) -> bool:
        row = verdicts.get((arm, qid))
        return bool(row and row['correct'] and not row['judge_error'])

    out: dict[str, object] = {'items': len(items), 'upstream_commit': UPSTREAM_COMMIT,
                              'reader': READER_MODEL, 'judge': JUDGE_MODEL, 'arms': {}}
    for arm in arms:
        hits = [correct(arm, i.question_id) for i in items]
        missing = sum((arm, i.question_id) not in answers for i in items)
        failed = sum(1 for i in items if (arm, i.question_id) in answers and not answers[(arm, i.question_id)]['completed'])
        unjudged = sum(1 for i in items if (arm, i.question_id) in answers
                       and answers[(arm, i.question_id)]['completed'] and (arm, i.question_id) not in verdicts)
        by_type: dict[str, list[bool]] = defaultdict(list)
        for item, hit in zip(items, hits, strict=True):
            by_type['abstention' if is_abstention(item.question_id) else item.question_type].append(hit)
        tokens = [int(cast(dict[str, int], a['usage'])['prompt_tokens']) for a in answers.values()
                  if a['arm'] == arm and a.get('usage')]
        low, high = wilson(sum(hits), len(hits))
        evidence_all = None
        if '@' in arm:
            system, _, k = arm.partition('@')
            scored = [i for i in items if i.evidence and not is_abstention(i.question_id) and i.question_id in rankings]
            evidence_all = (sum(set(i.evidence) <= set(cast(list[str], rankings[i.question_id][system])[:int(k)]) for i in scored)
                            / len(scored)) if scored else None
        cast(dict[str, object], out['arms'])[arm] = {
            'accuracy': sum(hits) / len(hits), 'ci95': [low, high], 'correct': sum(hits),
            'missing_answers': missing, 'failed_generations': failed, 'unjudged': unjudged,
            'by_type': {t: {'n': len(v), 'accuracy': sum(v) / len(v)} for t, v in sorted(by_type.items())},
            'prompt_tokens_median': sorted(tokens)[len(tokens) // 2] if tokens else None,
            'all_evidence_in_context': evidence_all,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['rank', 'answer', 'judge', 'report', 'all'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=100, help='stratified sample size; 0 for every item')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--depth', type=int, default=10, help='sessions each system ranks')
    parser.add_argument('--arms', default='full,oracle,scone@5,llamaindex@5')
    parser.add_argument('--embed-model', default='bge-base-en-v1.5')
    parser.add_argument('--embed-cache', type=Path, help='SQLite file reusing vectors across items and runs')
    parser.add_argument('--cot', action='store_true', help="upstream's step-by-step reader prompt")
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    items = load(args.dataset, args.sample, args.seed)
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('SCONE_CHAT_API_KEY') or ''
    if args.stage in ('answer', 'judge', 'all') and not key:
        raise SystemExit('set OPENROUTER_API_KEY or SCONE_CHAT_API_KEY')
    if args.stage in ('rank', 'all') and any('@' in a for a in arms):
        asyncio.run(rank(items, args.run_dir, args.depth, args.embed_model, args.embed_cache))
    if args.stage in ('answer', 'all'):
        asyncio.run(answer(items, arms, args.run_dir, key, args.cot, args.concurrency))
    if args.stage in ('judge', 'all'):
        asyncio.run(judge(items, args.run_dir, key, args.concurrency))
    result = report(items, arms, args.run_dir)
    (args.run_dir / 'report.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
