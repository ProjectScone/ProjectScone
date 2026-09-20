"""Frozen fixed-evidence pilot using Scone's generation adapter and QA scorer."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics

from scone_memory.providers.llm import OpenAICompatibleTextModel
from scone_memory.testing.generation_ablation import capture_public_reply
from scone_memory.testing.jev_answers import MODEL, PreparedAnswer
from scone_memory.testing.jev_public_qa import code_digest, digest, save
from scone_memory.testing.public_qa_run import request_digest

PROTOCOL = Path(__file__).with_name('jev-reasoning-answers-v1.protocol.md')


def selected_requests(source: Path, per_dataset: int = 20) -> list[PreparedAnswer]:
    if type(per_dataset) is not int or per_dataset not in (20, 100):
        raise ValueError('choose the frozen 20-question pilot or full 100 per dataset')
    completion = json.loads((source/'completion.json').read_text())
    if not completion['terminal'] or not completion['code_and_inputs_unchanged']:
        raise ValueError('baseline incomplete')
    if digest(source/'prepared.jsonl') != completion['prepared_sha256']:
        raise ValueError('baseline evidence changed')
    requests = [p for p in (PreparedAnswer.model_validate_json(line)
                           for line in (source/'prepared.jsonl').read_text().splitlines()) if p.arm == 'hybrid_jev']
    if len(requests) != 200 or len({p.question.id for p in requests}) != 200:
        raise ValueError('baseline schedule incomplete')
    if any(request_digest(p.request) != p.request_sha256 for p in requests):
        raise ValueError('request digest mismatch')
    order = lambda p: hashlib.sha256(('reasoning-v1:' + p.question.id).encode()).hexdigest()
    chosen = []
    for dataset in ('hotpotqa', 'squad'):
        eligible = sorted([p for p in requests if p.question.dataset == dataset], key=order)
        if len(eligible) != 100:
            raise ValueError('baseline dataset count differs')
        chosen.extend(eligible[:per_dataset])
    return sorted(chosen, key=order)


async def run(source: Path, output: Path, per_dataset: int = 20) -> None:
    requests = selected_requests(source, per_dataset)
    key = os.environ.get(os.environ.get('SCONE_JEV_API_KEY_ENV', 'OPENROUTER_API_KEY'))
    if not key:
        raise ValueError('configured token missing')
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = PROTOCOL if per_dataset == 20 else PROTOCOL.with_name('jev-reasoning-answers-full-v1.protocol.md')
    code, script, protocol = code_digest(), digest(Path(__file__)), digest(protocol_path)
    source_hash = digest(source/'prepared.jsonl')
    save(output/'manifest.json', {'baseline': str(source), 'prepared_sha256': source_hash,
        'code_sha256': code, 'script_sha256': script, 'protocol_sha256': protocol,
        'model': MODEL, 'max_output_tokens': 2048, 'temperature': 0, 'per_dataset': per_dataset,
        'protocol': protocol_path.name,
        'ids': [p.question.id for p in requests], 'arms': ['direct', 'reasoning']})
    with (output/'answers.jsonl').open('x') as stream:
        for index, prepared in enumerate(requests):
            for think in ((False, True) if index % 2 == 0 else (True, False)):
                model = OpenAICompatibleTextModel('https://openrouter.ai/api/v1', MODEL, api_key=key,
                    think=think, temperature=0, max_output_tokens=2048, timeout=90, trust_env=False)
                answer = await capture_public_reply(model, prepared.request, timeout=95)
                arm = 'reasoning' if think else 'direct'
                row = {'id': prepared.question.id, 'dataset': prepared.question.dataset, 'arm': arm,
                       'request_sha256': prepared.request_sha256, **answer}
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                print(f'{index+1}/{len(requests)} {arm}: {answer["status"]}', flush=True)
    unchanged = (code_digest() == code and digest(Path(__file__)) == script and digest(protocol_path) == protocol
                 and digest(source/'prepared.jsonl') == source_hash)
    save(output/'completion.json', {'terminal': True, 'code_and_inputs_unchanged': unchanged,
        'manifest_sha256': digest(output/'manifest.json'), 'answers_sha256': digest(output/'answers.jsonl')})
    if not unchanged:
        raise ValueError('inputs or code changed')


def score(dataset: Path, output: Path) -> None:
    from scone_memory.testing.public_qa import Gold, answer_score
    from scone_memory.testing.public_qa_score import latency

    completion = json.loads((output/'completion.json').read_text())
    if not completion['terminal'] or not completion['code_and_inputs_unchanged']:
        raise ValueError('run incomplete')
    for name in ('manifest', 'answers'):
        suffix = '.json' if name == 'manifest' else '.jsonl'
        if digest(output/(name+suffix)) != completion[name+'_sha256']:
            raise ValueError('artifact changed')
    manifest = json.loads((output/'manifest.json').read_text())
    source = Path(manifest['baseline'])
    source_manifest = json.loads((source/'manifest.json').read_text())
    if (digest(source/'prepared.jsonl') != manifest['prepared_sha256']
            or digest(dataset/'gold.jsonl') != source_manifest['gold_sha256']):
        raise ValueError('source or gold changed')
    requests = {p.question.id: p for p in selected_requests(source, manifest.get('per_dataset', 20))}
    rows = [json.loads(line) for line in (output/'answers.jsonl').read_text().splitlines()]
    expected = {(identifier, arm) for identifier in requests for arm in ('direct', 'reasoning')}
    if len(rows) != len(expected) or {(r['id'], r['arm']) for r in rows} != expected:
        raise ValueError('incomplete or duplicate schedule')
    if any(r['request_sha256'] != requests[r['id']].request_sha256
           or r['dataset'] != requests[r['id']].question.dataset for r in rows):
        raise ValueError('request changed')
    gold = {g.id: g for g in (Gold.model_validate_json(line) for line in (dataset/'gold.jsonl').read_text().splitlines())}
    for row in rows:
        row.update(answer_score(row['answer_text'], gold[row['id']].answers, row['dataset'], row['completed']))
    groups = []
    for name in ('all', 'hotpotqa', 'squad'):
        for arm in ('direct', 'reasoning'):
            selected = [r for r in rows if r['arm'] == arm and (name == 'all' or r['dataset'] == name)]
            groups.append({'dataset': name, 'arm': arm, 'n': len(selected),
                'em': statistics.mean(r['em'] for r in selected), 'f1': statistics.mean(r['f1'] for r in selected),
                'failures': sum(not r['completed'] for r in selected),
                'abstentions': sum(r['answer_text'].strip() == 'INSUFFICIENT_EVIDENCE' for r in selected),
                'latency': latency([r['total_ms'] for r in selected])})
    save(output/'scores.json', {'groups': groups, 'per_question': rows})
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'score'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-dataset', type=int, choices=(20, 100), default=20)
    args = parser.parse_args()
    if args.command == 'run':
        if args.baseline is None:
            parser.error('--baseline required')
        asyncio.run(run(args.baseline, args.output, args.per_dataset))
    else:
        if args.dataset is None:
            parser.error('--dataset required')
        score(args.dataset, args.output)
