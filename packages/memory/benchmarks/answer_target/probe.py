"""Fixed local generation-only feasibility probe; inference never reads gold."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx
from pydantic import BaseModel, ConfigDict

from local_structure.run import Row
from matched_qa.run import append, digest, save
from qasper_structure.data import Question
from qasper_structure.pipeline import messages
from scone_memory.testing.public_qa import _mapping

MODEL = 'llama3.2-ctx8k:latest'
BASE = 'http://127.0.0.1:11434'


class Message(BaseModel):
    content: str


class Reply(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)
    model: str
    message: Message
    done: bool
    done_reason: str | None = None


class Result(BaseModel):
    id: str
    paper_id: str
    completed: bool
    answer: str
    error: str | None
    wall_ms: float
    response: dict[str, object] | None


async def generate(client: httpx.AsyncClient, identifier: str, paper_id: str,
                   query: str, context: str) -> Result:
    started = time.perf_counter()
    answer = ''
    completed = False
    error = None
    receipt = None
    try:
        response = await client.post(BASE + '/api/chat', json={'model': MODEL,
            'messages': messages(query, context), 'stream': False, 'keep_alive': '5m',
            'options': {'temperature': 0, 'num_predict': 256}}, timeout=60)
        response.raise_for_status()
        if len(response.content) > 256000:
            raise ValueError('oversized response')
        parsed = Reply.model_validate_json(response.content)
        receipt = _mapping(response.json())
        if parsed.model != MODEL:
            raise ValueError('model identity differs from protocol')
        answer = parsed.message.content
        completed = parsed.done and parsed.done_reason == 'stop' and bool(answer.strip())
        error = None if completed else 'incomplete_generation'
    except (httpx.HTTPError, TimeoutError, ValueError) as failure:
        error = type(failure).__name__
    return Result(id=identifier, paper_id=paper_id, completed=completed, answer=answer,
                  error=error, wall_ms=(time.perf_counter() - started) * 1000, response=receipt)


async def run(dataset: Path, prepared: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError('output not empty')
    questions = [Question.model_validate_json(s) for s in (dataset / 'questions.jsonl').read_text().splitlines()]
    if len(questions) != 1005 or len({q.id for q in questions}) != 1005:
        raise ValueError('complete development question export required')
    rows = [Row.model_validate_json(s) for s in (prepared / 'observations.jsonl').read_text().splitlines()]
    completion = json.loads((prepared / 'completion.json').read_text())
    prior = json.loads((prepared / 'manifest.json').read_text())
    if (not completion['complete'] or len(rows) != 1005 or len({r.id for r in rows}) != 1005
            or completion['manifest_sha256'] != digest(prepared / 'manifest.json')
            or completion['observations_sha256'] != digest(prepared / 'observations.jsonl')
            or prior['inputs']['questions.jsonl'] != digest(dataset / 'questions.jsonl')
            or {(r.id, r.paper_id) for r in rows} != {(q.id, q.paper_id) for q in questions}):
        raise ValueError('prepared context schedule or digest mismatch')
    contexts = {r.id: r.arms['hybrid_no_rerank'].context for r in rows}
    selected = sorted(questions, key=lambda q: hashlib.sha256(q.id.encode()).hexdigest())[:32]
    paths = [Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
             Path(__file__).parents[1] / 'qasper_structure/pipeline.py',
             dataset / 'questions.jsonl', prepared / 'observations.jsonl']
    frozen = {str(p): digest(p) for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        tags = await client.get(BASE + '/api/tags', timeout=10)
        tags.raise_for_status()
        save(output / 'manifest.json', {'protocol': 'answer-target-feasibility-v1',
            'model': MODEL, 'model_tags': _mapping(tags.json()), 'source_input_sha256': frozen,
            'scheduled': [q.model_dump() for q in selected], 'target_ms': 100,
            'scope': 'generation only using saved local-only contexts; no accuracy acceptance claim'})
        warm = await generate(client, 'warmup', 'fixture', 'Reply with OK.', 'The answer is OK.')
        save(output / 'warmup.json', warm.model_dump())
        with (output / 'observations.jsonl').open('x') as stream:
            for position, question in enumerate(selected, 1):
                row = await generate(client, question.id, question.paper_id, question.question, contexts[question.id])
                append(stream, row.model_dump())
                print(f'{position}/32 completed={row.completed} wall_ms={row.wall_ms:.2f}', flush=True)
        if any(digest(Path(p)) != expected for p, expected in frozen.items()):
            raise ValueError('source or inputs changed during run')
        save(output / 'completion.json', {'complete': True, 'questions': 32,
            'observations_sha256': digest(output / 'observations.jsonl'),
            'manifest_sha256': digest(output / 'manifest.json')})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.prepared, args.output))


if __name__ == '__main__':
    main()
