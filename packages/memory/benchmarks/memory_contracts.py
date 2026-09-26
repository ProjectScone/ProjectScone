"""Live direct-Jev experiment, not an end-to-end RAG benchmark.

PYTHONPATH=packages/memory/src python packages/memory/benchmarks/memory_contracts.py --output /tmp/contracts.json
Set TYPESAFE_API_KEY, TYPESAFE_BASE_URL and TYPESAFE_DEFAULT_MODEL in the process
environment. Use scripts/local_env.py to load an owned private environment file.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import cast

from memory_contract_cases import Case, cases
from scone_memory.experimental.contract_judge import JevContractJudge
from scone_memory.experimental.memory_contracts import World, compile_contract, evidence_worlds


async def run_case(case: Case, judge: JevContractJudge) -> dict[str, object]:
    worlds = evidence_worlds(case.request)
    if len(case.expected) != len(worlds):
        raise ValueError('fixture labels do not cover all evidence worlds')
    compiled = await judge.assess(case.request, worlds)
    contract = compile_contract(case.request, compiled.judgments, model=compiled.model,
                                expires_at=time.time() + 3600)
    rows: list[dict[str, object]] = []
    baseline_input = baseline_output = baseline_calls = 0
    for world, expected in zip(worlds, case.expected, strict=True):
        current = replace(case.request, evidence=world.evidence)
        started = time.perf_counter()
        decision = contract.evaluate(current, now=time.time())
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Same evidence, model, prompts and thresholds. Only precomputation differs.
        baseline = await judge.assess(case.request, (world,))
        baseline_input += baseline.input_tokens
        baseline_output += baseline.output_tokens
        baseline_calls += baseline.requests
        rows.append({'mask': world.mask, 'expected': expected, 'contract': decision.status,
                     'contract_ms': elapsed_ms, 'baseline': baseline.judgments[0].status,
                     'baseline_ms': baseline.elapsed_ms, 'baseline_model': baseline.model,
                     'probabilities': asdict(compiled.judgments[world.mask]),
                     'baseline_probabilities': asdict(baseline.judgments[0]),
                     'origins': sorted({item.origin for item in world.evidence})})
    return {'name': case.name, 'compile': asdict(compiled),
            'minimal_withdrawals': contract.minimal_withdrawals(),
            'baseline_input_tokens': baseline_input, 'baseline_output_tokens': baseline_output,
            'baseline_requests': baseline_calls, 'rows': rows}


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    rows = [row for result in results for row in cast(list[dict[str, object]], result['rows'])]
    live_rows = [row for row in rows if row['mask'] != 0]
    compiler = [cast(dict[str, object], result['compile']) for result in results]
    compile_ms = sum(float(cast(float, item['elapsed_ms'])) for item in compiler)
    baseline_ms = sum(float(cast(float, row['baseline_ms'])) for row in live_rows)
    return {
        'cases': len(results), 'worlds': len(rows),
        'contract_correct': sum(row['contract'] == row['expected'] for row in rows),
        'baseline_correct': sum(row['baseline'] == row['expected'] for row in rows),
        'contract_false_support': sum(row['contract'] == 'supported' and row['expected'] != 'supported' for row in rows),
        'baseline_false_support': sum(row['baseline'] == 'supported' and row['expected'] != 'supported' for row in rows),
        'disagreements': sum(row['contract'] != row['baseline'] for row in rows),
        'compile_total_ms': compile_ms,
        'compile_median_ms': statistics.median(float(cast(float, item['elapsed_ms'])) for item in compiler),
        'contract_lookup_median_ms': statistics.median(float(cast(float, row['contract_ms'])) for row in live_rows),
        'fresh_check_median_ms': statistics.median(float(cast(float, row['baseline_ms'])) for row in live_rows),
        'amortized_latency_break_even_lookups_per_contract': math.ceil(compile_ms / (baseline_ms / len(live_rows)) / len(results)),
        'compile_requests': sum(int(cast(int, item['requests'])) for item in compiler),
        'baseline_requests': sum(int(cast(int, item['baseline_requests'])) for item in results),
        'compile_questions': sum(int(cast(int, item['questions'])) for item in compiler),
        'compile_input_tokens': sum(int(cast(int, item['input_tokens'])) for item in compiler),
        'compile_output_tokens': sum(int(cast(int, item['output_tokens'])) for item in compiler),
        'baseline_input_tokens': sum(int(cast(int, item['baseline_input_tokens'])) for item in results),
        'baseline_output_tokens': sum(int(cast(int, item['baseline_output_tokens'])) for item in results),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    judge = JevContractJudge(os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai'),
                             os.environ.get('TYPESAFE_DEFAULT_MODEL', 'jev-latest'),
                             api_key=os.environ['TYPESAFE_API_KEY'])
    fixtures = cases()
    fixture_hash = hashlib.sha256(json.dumps([asdict(case) for case in fixtures], sort_keys=True).encode()).hexdigest()
    results: list[dict[str, object]] = []
    report: dict[str, object] = {'schema': 1, 'fixture_sha256': fixture_hash,
                               'scope': 'synthetic evidence withdrawal; no retrieval or generation', 'cases': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in fixtures:
        result = await run_case(case, judge)
        results.append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'case': case.name, 'completed_worlds': len(case.expected)}), flush=True)
    report['summary'] = summarize(results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    asyncio.run(main())
