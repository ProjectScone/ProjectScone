"""Recover symmetric Jev billing failures through OpenRouter, preserving the parent."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import fcntl
import json
import math
from pathlib import Path
import time
from typing import Mapping, cast

import httpx

from scone_memory.testing.public_qa import Question, _array, _mapping

from .pipeline import (MODEL, OPENROUTER_JEV_MODEL, Passage, credential, error_name,
                       generate, messages, pack_context, rank, rerank, unique_passages)
from .run import (ARMS, append, digest, failed_row, previous_rows, repair_journal_tail,
                  save, source_hashes)
from .scoring import Observation, _integrity


def recovery_ids(rows: Mapping[tuple[str, str], Mapping[str, object]]) -> set[str]:
    identifiers = {identifier for (identifier, _), row in rows.items() if row.get('completed') is not True}
    for identifier in identifiers:
        for arm in ARMS:
            row = rows.get((identifier, arm), {})
            if row.get('completed') is not False or row.get('error') != 'rerank_http_402':
                raise ValueError('recovery requires symmetric rerank_http_402 failures')
    return identifiers


def strings(value: object) -> list[str]:
    items = _array(value)
    if any(not isinstance(item, str) for item in items):
        raise ValueError('expected string list')
    return cast(list[str], items)


def restore_candidates(row: Mapping[str, object], passages: Mapping[str, Passage]) -> list[Passage]:
    keys = strings(row.get('retrieved_chunk_ids'))
    if not 1 <= len(keys) <= 32 or len(set(keys)) != len(keys) or not set(keys) <= passages.keys():
        raise ValueError('invalid saved candidates')
    restored = [passages[key] for key in keys]
    if [p.document_id for p in restored] != strings(row.get('retrieved_ids')):
        raise ValueError('saved candidate source mismatch')
    return restored


@dataclass(frozen=True)
class RecoveryQuestion:
    position: int
    question: Question
    candidates: dict[str, list[Passage]]


def load_jobs(parent: Path, questions: list[Question], rows: dict[tuple[str, str], dict[str, object]],
              identifiers: set[str]) -> list[RecoveryQuestion]:
    by_id = {q.id: (i, q) for i, q in enumerate(questions)}
    jobs: dict[str, RecoveryQuestion] = {}
    with (parent / 'judgments.jsonl').open() as stream:
        for line in stream:
            receipt = _mapping(json.loads(line))
            identifier = receipt.get('id')
            if not isinstance(identifier, str) or identifier not in identifiers:
                continue
            if identifier in jobs:
                raise ValueError('duplicate saved judgment')
            position, question = by_id[identifier]
            if receipt.get('error') != 'http_402' or receipt.get('question') != question.question:
                raise ValueError('saved judgment differs from scheduled recovery')
            passages: dict[str, Passage] = {}
            for raw in _array(receipt['candidates']):
                p = _mapping(raw)
                if any(not isinstance(p.get(key), str) for key in ('key', 'document_id', 'text')):
                    raise ValueError('invalid saved passage')
                passage = Passage(cast(str, p['key']), cast(str, p['document_id']), cast(str, p['text']))
                if passage.key in passages:
                    raise ValueError('duplicate saved passage')
                passages[passage.key] = passage
            candidates = {arm: restore_candidates(rows[identifier, arm], passages) for arm in ARMS}
            if {p.key for p in unique_passages(candidates)} != passages.keys():
                raise ValueError('saved union differs from arm candidates')
            jobs[identifier] = RecoveryQuestion(position, question, candidates)
    if jobs.keys() != identifiers:
        raise ValueError('missing saved failed judgment')
    return sorted(jobs.values(), key=lambda job: job.position)


def cached_judgments(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = _mapping(json.loads(line))
            identifier = row.get('id')
            if not isinstance(identifier, str) or identifier in result:
                raise ValueError('invalid recovery judgment journal')
            result[identifier] = row
    return result


def saved_scores(receipt: Mapping[str, object], candidates: Mapping[str, list[Passage]]) -> dict[str, float]:
    raw = _mapping(receipt['scores'])
    scores: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid saved score')
        scores[key] = float(value)
    if scores.keys() != {p.key for p in unique_passages(candidates)}:
        raise ValueError('saved scores differ from original candidates')
    return scores


async def evaluate(jobs: list[RecoveryQuestion], parent_rows: dict[tuple[str, str], dict[str, object]],
                   output: Path, concurrency: int) -> None:
    old = previous_rows(output / 'observations.jsonl')
    attempted = previous_rows(output / 'attempts.jsonl')
    judgments = cached_judgments(output / 'judgments.jsonl')
    stop = asyncio.Event()
    with (output / 'observations.jsonl').open('a') as observations, \
            (output / 'attempts.jsonl').open('a') as attempts, \
            (output / 'judgments.jsonl').open('a') as audits, \
            (output / 'provider-errors.jsonl').open('a') as errors:
        for identifier, arm in set(attempted) - set(old):
            row = failed_row(identifier, arm, 'interrupted_recovery_attempt')
            append(observations, row)
            old[identifier, arm] = row
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            async def question_run(job: RecoveryQuestion) -> None:
                identifier = job.question.id
                order = ARMS if job.position % 2 == 0 else tuple(reversed(ARMS))
                arms = [arm for arm in order if (identifier, arm) not in old]
                if not arms or stop.is_set():
                    return
                if identifier in judgments:
                    audit = judgments[identifier]
                    scores = saved_scores(audit, job.candidates)
                    rerank_ms = float(cast(float, audit['rerank_ms']))
                else:
                    try:
                        scores, audit, rerank_ms = await rank(client, job.question.question,
                            unique_passages(job.candidates), provider='openrouter')
                    except (httpx.HTTPError, ValueError, TimeoutError) as error:
                        append(errors, {'id': identifier, 'stage': 'rerank', 'error': error_name(error),
                            'time_unix': time.time()})
                        stop.set()
                        return
                    append(audits, {'id': identifier, 'arms': list(ARMS), **audit,
                        'scores': scores, 'rerank_ms': rerank_ms})
                for arm in arms:
                    # Finish this already-started pair even when another job stops scheduling.
                    context, context_ids = pack_context(rerank(job.candidates[arm], scores))
                    append(attempts, {'id': identifier, 'arm': arm, 'provider': 'openrouter',
                        'time_unix': time.time()})
                    reply = await generate(client, messages(job.question.question, context))
                    retrieval_ms = float(cast(float, parent_rows[identifier, arm]['retrieval_ms']))
                    generation_ms = float(cast(float, reply['generation_ms']))
                    row = {**parent_rows[identifier, arm], **reply, 'context_ids': context_ids,
                        'context_bytes': len(context.encode()), 'rerank_ms': rerank_ms,
                        'total_ms': retrieval_ms + rerank_ms + generation_ms,
                        'recovery': {'provider': 'openrouter', 'retrieval_reused': True}}
                    append(observations, row)
                    old[identifier, arm] = row
                    if reply['completed'] is not True:
                        stop.set()
                    if len(old) % 20 == 0:
                        save(output / 'progress.json', {'phase': 'recovery', 'terminal': len(old),
                            'planned': len(parent_rows), 'recovered': len(old) - (len(parent_rows) - len(jobs) * 2)})
                        print(f'Answers {len(old)}/{len(parent_rows)}', flush=True)
            pending = [job for job in jobs if any((job.question.id, arm) not in old for arm in ARMS)]
            for offset in range(0, len(pending), concurrency):
                if stop.is_set():
                    break
                async with asyncio.TaskGroup() as group:
                    for job in pending[offset:offset + concurrency]:
                        group.create_task(question_run(job))


async def run(dataset: Path, parent: Path, output: Path, concurrency: int, resume: bool) -> None:
    if not 1 <= concurrency <= 4 or output.resolve() == parent.resolve():
        raise ValueError('use a separate recovery output and concurrency 1..4')
    credential('chat')
    original = _integrity(dataset, parent, False)
    if original.get('protocol') != 'matched-qa-full-v1' or original.get('chat_model') != MODEL:
        raise ValueError('unsupported original run')
    questions = [Question.model_validate_json(line) for line in (dataset / 'questions.jsonl').read_text().splitlines()]
    rows = previous_rows(parent / 'observations.jsonl')
    expected = {(q.id, arm) for q in questions for arm in ARMS}
    if len(questions) != 17975 or set(rows) != expected:
        raise ValueError('parent must cover the complete original schedule')
    for row in rows.values():
        Observation.model_validate(row)
    identifiers = recovery_ids(rows)
    jobs = load_jobs(parent, questions, rows, identifiers)
    sources = source_hashes()
    manifest = {**original, 'protocol': 'matched-qa-openrouter-recovery-v1', 'source_sha256': sources,
        'parent_completion_sha256': digest(parent / 'completion.json'),
        'recovery_protocol_sha256': digest(Path(__file__).with_name('RECOVERY.md')),
        'recovery_ids': sorted(identifiers), 'recovery_provider': 'openrouter',
        'recovery_jev_model': OPENROUTER_JEV_MODEL, 'concurrency': concurrency,
        'scope': 'full schedule with original successful answers and OpenRouter recovery of symmetric billing failures'}
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            if json.loads((output / 'manifest.json').read_text()) != manifest:
                raise ValueError('resume requires unchanged recovery sources, inputs and parent')
            for name in ('observations.jsonl', 'attempts.jsonl', 'judgments.jsonl', 'provider-errors.jsonl'):
                repair_journal_tail(output / name)
        else:
            save(output / 'manifest.json', manifest)
            with (output / 'observations.jsonl').open('w') as observations, (output / 'attempts.jsonl').open('w') as attempts:
                for (identifier, arm), row in rows.items():
                    if identifier not in identifiers:
                        append(observations, row)
                        append(attempts, {'id': identifier, 'arm': arm, 'origin': 'parent'})
        old = previous_rows(output / 'observations.jsonl')
        if not set(old) <= expected:
            raise ValueError('unscheduled recovery row')
        for key, row in old.items():
            if key[0] not in identifiers and row != rows[key]:
                raise ValueError('original successful answer changed')
        try:
            await evaluate(jobs, rows, output, concurrency)
        finally:
            observed = previous_rows(output / 'observations.jsonl')
            unchanged = sources == source_hashes() and manifest['parent_completion_sha256'] == digest(parent / 'completion.json')
            _integrity(dataset, parent, False)
            unchanged = unchanged and manifest['recovery_protocol_sha256'] == digest(Path(__file__).with_name('RECOVERY.md'))
            complete = set(observed) == expected
            successful = sum(row.get('completed') is True for row in observed.values())
            save(output / 'completion.json', {'completed': complete, 'answers_complete': successful == len(expected),
                'code_and_inputs_unchanged': unchanged, 'successful_answers': successful,
                'provider_errors_sha256': digest(output / 'provider-errors.jsonl'),
                'artifact_hashes': {name: digest(output / name) for name in
                    ('manifest.json', 'observations.jsonl', 'attempts.jsonl', 'judgments.jsonl')}})
            save(output / 'progress.json', {'phase': 'complete' if complete else 'stopped',
                'terminal': len(observed), 'successful': successful, 'planned': len(expected)})
        if not complete or successful != len(expected):
            raise RuntimeError('recovery stopped or contains failed answers; inspect progress and error journals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.parent, args.output, args.concurrency, args.resume))
