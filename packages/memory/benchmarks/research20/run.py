"""Run the portfolio: --live uses configured direct Jev; default is offline only."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .common import ExperimentResult
from .jev import JevResearchClient


def _manifest() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob('*.py'))}


def save(path: Path, results: list[ExperimentResult], client: JevResearchClient | None,
         error: str | None = None) -> None:
    ids = [item.experiment_id for item in results]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate experiment IDs')
    report: dict[str, object] = {
        'schema': 1, 'complete': sorted(ids) == list(range(1, 21)) and error is None,
        'source_sha256': _manifest(), 'results': [asdict(item) for item in results],
        'provider_audits': client.audits if client else [], 'error': error,
        'scope': 'authored diagnostics; simulation and live API evidence are separate',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--track', choices=('all', 'validity', 'selection', 'efficiency', 'reliability'), default='all')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output path; do not overwrite experimental evidence')
    if args.track == 'reliability' and not args.live:
        parser.error('the reliability track requires --live')
    results: list[ExperimentResult] = []
    client: JevResearchClient | None = None
    try:
        tracks: list[Callable[[], list[ExperimentResult]]] = []
        if args.track in ('all', 'validity'):
            from .validity import run as run_validity
            tracks.append(run_validity)
        if args.track in ('all', 'selection'):
            from .selection import run as run_selection
            tracks.append(run_selection)
        if args.track in ('all', 'efficiency'):
            from .efficiency import run as run_efficiency
            tracks.append(run_efficiency)
        for track in tracks:
            for result in track():
                results.append(result)
                save(args.output, results, client)
                print(json.dumps({'id': result.experiment_id, 'title': result.title,
                    'baseline': result.baseline_score, 'method': result.method_score,
                    'metric': result.metric, 'evidence': result.evidence_kind}), flush=True)
        if args.live and args.track in ('all', 'reliability'):
            client = JevResearchClient(os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai'),
                                       os.environ.get('TYPESAFE_DEFAULT_MODEL', 'jev-latest'),
                                       os.environ['TYPESAFE_API_KEY'])
            from .reliability import (abstention_calibration, conflict_decomposition,
                                     negative_evidence, paraphrase_stability, self_citation)
            for experiment in (negative_evidence, self_citation, paraphrase_stability,
                               conflict_decomposition, abstention_calibration):
                result = await experiment(client)
                results.append(result)
                save(args.output, results, client)
                print(json.dumps({'id': result.experiment_id, 'title': result.title,
                    'baseline': result.baseline_score, 'method': result.method_score,
                    'metric': result.metric, 'evidence': result.evidence_kind}), flush=True)
        requested = (20 if args.live else 15) if args.track == 'all' else 5
        print(json.dumps({'completed': len(results), 'requested': requested,
                          'report': str(args.output)}))
    except Exception as error:
        save(args.output, results, client, type(error).__name__)
        raise


if __name__ == '__main__':
    asyncio.run(main())
