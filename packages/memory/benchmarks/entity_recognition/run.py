"""Score entity recognizers on complete public test sets, in Scone's entity types.

Recognizers:
- ``gliner:<model>``: GLiNER, a local model that finds spans of any named type. Runs on CPU; needs the ``gliner``
  package (kept out of Scone's own dependencies; install it in a separate environment).
- ``spacy:<model>``: a spaCy pipeline, e.g. ``en_core_web_trf``, trained on OntoNotes' training split (so its OntoNotes
  score is the standard held-out test, and its Few-NERD score measures transfer).
- ``llm:<model>``: an OpenAI-compatible chat model asked for the entities of each sentence as JSON, each with its
  exact text and one of Scone's types. A returned text is placed on the first matching run of tokens not already
  taken; text that matches no run of tokens is counted as unplaced and scores nothing.

Scoring is strict: a prediction counts only with the gold span's exact tokens and type. Precision, recall and F1
are micro-averaged over every sentence, and reported per type. Predictions of a type a test set never labels
(dates on Few-NERD, for example) are not counted against it. Every prediction is kept in
``predictions-<dataset>-<recognizer>.jsonl``, and a run resumes from it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .data import SCONE_TYPES, Sentence, load_fewnerd, load_ontonotes, scored_types, to_scone

GLINER_LABELS = {
    'person': 'person', 'organization': 'organization', 'nationality': 'nationality or religious or political group',
    'location': 'location', 'facility': 'facility', 'product': 'product', 'event': 'event',
    'work_of_art': 'work of art', 'law': 'law', 'language': 'language', 'date': 'date', 'time': 'time',
    'money': 'money', 'percent': 'percent', 'quantity': 'quantity', 'ordinal': 'ordinal', 'cardinal': 'cardinal number',
}


def load(dataset: str, data_dir: Path) -> list[Sentence]:
    if dataset == 'ontonotes':
        return load_ontonotes(data_dir / 'ontonotes5-test.parquet')
    return load_fewnerd(data_dir / 'fewnerd-supervised-test.parquet', data_dir / 'fewnerd-labels.json')


def char_to_tokens(sentence: Sentence, start: int, end: int) -> tuple[int, int] | None:
    """The tokens a character span covers; None unless it starts at a token start and ends at a token end."""
    offsets = sentence.offsets()
    first = next((i for i, (s, _) in enumerate(offsets) if s == start), None)
    last = next((i for i, (_, e) in enumerate(offsets) if e == end), None)
    if first is None or last is None or last < first:
        return None
    return first, last + 1


def place_text(sentence: Sentence, text: str, taken: set[tuple[int, int]]) -> tuple[int, int] | None:
    """The first run of tokens spelling ``text`` (tokens joined by spaces, or with no space before punctuation)."""
    wanted = ' '.join(text.split())
    n = len(sentence.tokens)
    for start in range(n):
        for end in range(start + 1, min(n, start + 40) + 1):
            joined = ' '.join(sentence.tokens[start:end])
            loose = re.sub(r' (?=[,.;:!?%)\]}\'])|(?<=[(\[{$]) ', '', joined)
            if (joined == wanted or loose == wanted) and (start, end) not in taken:
                return start, end
            if len(joined) > len(wanted) + 10:
                break
    return None


def predict_gliner(sentences: Sequence[Sentence], model_name: str, threshold: float, batch: int,
                   out: Path) -> None:
    from gliner import GLiNER  # type: ignore[import-not-found]

    done = {json.loads(line)['id'] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in sentences if s.id not in done]
    if not todo:
        return
    model = GLiNER.from_pretrained(model_name)
    labels = list(GLINER_LABELS.values())
    back = {v: k for k, v in GLINER_LABELS.items()}
    started = time.perf_counter()
    with out.open('a', encoding='utf-8') as handle:
        for i in range(0, len(todo), batch):
            group = todo[i:i + batch]
            texts = [s.text for s in group]
            began = time.perf_counter()
            if hasattr(model, 'inference'):
                results = model.inference(texts, labels, threshold=threshold, batch_size=batch)
            else:  # older releases
                results = model.batch_predict_entities(texts, labels, threshold=threshold)
            ms = (time.perf_counter() - began) * 1000 / len(group)
            for sentence, found in zip(group, results):
                spans, unplaced = [], 0
                for entity in found:
                    tokens = char_to_tokens(sentence, int(entity['start']), int(entity['end']))
                    if tokens is None:
                        unplaced += 1
                        continue
                    spans.append([tokens[0], tokens[1], back[entity['label']], float(entity.get('score', 0))])
                handle.write(json.dumps({'id': sentence.id, 'spans': spans, 'unplaced': unplaced, 'ms': ms}) + '\n')
            if (i // batch) % 50 == 0:
                rate = (i + len(group)) / (time.perf_counter() - started)
                print(f'  gliner {i + len(group)}/{len(todo)} sentences, {rate:.1f}/s', flush=True)


def predict_spacy(sentences: Sequence[Sentence], model_name: str, batch: int, out: Path) -> None:
    """spaCy's pipeline on each sentence's text; its OntoNotes labels map onto Scone's types as the gold does."""
    import spacy  # type: ignore[import-not-found]

    from .data import ONTONOTES_TO_SCONE

    done = {json.loads(line)['id'] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in sentences if s.id not in done]
    if not todo:
        return
    nlp = spacy.load(model_name)
    started = time.perf_counter()
    with out.open('a', encoding='utf-8') as handle:
        for i in range(0, len(todo), batch):
            group = todo[i:i + batch]
            began = time.perf_counter()
            docs = list(nlp.pipe([s.text for s in group], batch_size=batch))
            ms = (time.perf_counter() - began) * 1000 / len(group)
            for sentence, doc in zip(group, docs):
                spans, unplaced = [], 0
                for ent in doc.ents:
                    kind = ONTONOTES_TO_SCONE.get(ent.label_)
                    tokens = char_to_tokens(sentence, ent.start_char, ent.end_char) if kind else None
                    if tokens is None:
                        unplaced += 1
                        continue
                    spans.append([tokens[0], tokens[1], kind, 1.0])
                handle.write(json.dumps({'id': sentence.id, 'spans': spans, 'unplaced': unplaced, 'ms': ms}) + '\n')
            if (i // batch) % 50 == 0:
                print(f'  spacy {i + len(group)}/{len(todo)}, {(i + len(group)) / (time.perf_counter() - started):.1f}/s',
                      flush=True)


LLM_SYSTEM = ('You find every named entity and value mention in a sentence and give its type. Types:\n'
              + '\n'.join(f'- {name}: {meaning}' for name, meaning in SCONE_TYPES.items())
              + '\nReturn only JSON: {"entities": [{"text": "<exact text from the sentence>", "type": "<one type>"}]}. '
              'Copy each text exactly as it appears. Return {"entities": []} when there are none.')


async def predict_llm(sentences: Sequence[Sentence], url: str, model: str, key: str, concurrency: int,
                      out: Path) -> None:
    import httpx

    done = {json.loads(line)['id'] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in sentences if s.id not in done]
    gate = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    count = 0
    async with httpx.AsyncClient() as client:
        async def one(sentence: Sentence) -> None:
            nonlocal count
            payload = {'model': model, 'temperature': 0, 'max_tokens': 600, 'reasoning_effort': 'none',
                       'messages': [{'role': 'system', 'content': LLM_SYSTEM},
                                    {'role': 'user', 'content': sentence.text}]}
            row: dict[str, Any] = {'id': sentence.id, 'spans': [], 'unplaced': 0}
            async with gate:
                began = time.perf_counter()
                for attempt in range(3):
                    try:
                        response = await client.post(url.rstrip('/') + '/chat/completions', json=payload,
                                                     headers={'Authorization': 'Bearer ' + key} if key else {}, timeout=120)
                        response.raise_for_status()
                        body = response.json()
                        text = body['choices'][0]['message'].get('content') or ''
                        row['usage'] = body.get('usage')
                        match = re.search(r'\{.*\}', text, re.S)
                        entities = json.loads(match.group(0))['entities'] if match else []
                        taken: set[tuple[int, int]] = set()
                        for entity in entities:
                            kind = str(entity.get('type', '')).strip().lower()
                            placed = place_text(sentence, str(entity.get('text', '')), taken) if kind in SCONE_TYPES else None
                            if placed is None:
                                row['unplaced'] += 1
                                continue
                            taken.add(placed)
                            row['spans'].append([placed[0], placed[1], kind, 1.0])
                        row['error'] = None
                        break
                    except Exception as error:  # noqa: BLE001 - recorded, and the sentence scores as nothing found
                        row['error'] = type(error).__name__
                        await asyncio.sleep(2 ** attempt)
                row['ms'] = (time.perf_counter() - began) * 1000
            async with lock:
                with out.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')
                count += 1
                if count % 500 == 0:
                    print(f'  llm {count}/{len(todo)}', flush=True)
        await asyncio.gather(*(one(s) for s in todo))


def score(sentences: Sequence[Sentence], dataset: str, predictions: Path) -> dict[str, Any]:
    rows = {}
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        rows[row['id']] = row  # a retried sentence's last row counts
    allowed = set(scored_types(dataset))
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    missing = errors = unplaced = 0
    ms = []
    for sentence in sentences:
        gold = {span for span in to_scone(sentence, dataset) if span[2] in allowed}
        row = rows.get(sentence.id)
        if row is None:
            missing += 1
            predicted: set[tuple[int, int, str]] = set()
        else:
            errors += bool(row.get('error'))
            unplaced += int(row.get('unplaced', 0))
            ms.append(float(row.get('ms', 0)))
            predicted = {(int(s), int(e), str(k)) for s, e, k, *_ in row['spans'] if k in allowed}
        for span in predicted & gold:
            tp[span[2]] += 1
        for span in predicted - gold:
            fp[span[2]] += 1
        for span in gold - predicted:
            fn[span[2]] += 1

    def prf(t: int, p: int, n: int) -> dict[str, float]:
        precision = t / (t + p) if t + p else 0.0
        recall = t / (t + n) if t + n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1, 'gold': t + n}

    total = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_type = {kind: prf(tp[kind], fp[kind], fn[kind]) for kind in sorted(allowed) if tp[kind] + fp[kind] + fn[kind]}
    ms.sort()
    return {'dataset': dataset, 'sentences': len(sentences), 'missing': missing, 'errors': errors,
            'unplaced_predictions': unplaced, 'micro': total, 'per_type': per_type,
            'ms_per_sentence_p50': ms[len(ms) // 2] if ms else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['predict', 'score'])
    parser.add_argument('--dataset', choices=['ontonotes', 'fewnerd'], required=True)
    parser.add_argument('--recognizer', required=True, help='gliner:<hf model>, spacy:<model> or llm:<model>')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--llm-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--concurrency', type=int, default=16)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    sentences = load(args.dataset, args.data_dir)[:args.limit] if args.limit else load(args.dataset, args.data_dir)
    kind, _, name = args.recognizer.partition(':')
    out = args.run_dir / f"predictions-{args.dataset}-{args.recognizer.replace('/', '_').replace(':', '-')}.jsonl"
    if args.stage == 'predict':
        if kind == 'gliner':
            predict_gliner(sentences, name, args.threshold, args.batch, out)
        elif kind == 'spacy':
            predict_spacy(sentences, name, args.batch, out)
        elif kind == 'llm':
            key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('SCONE_CHAT_API_KEY') or ''
            asyncio.run(predict_llm(sentences, args.llm_url, name, key, args.concurrency, out))
        else:
            raise SystemExit('recognizer must be gliner:<model>, spacy:<model> or llm:<model>')
    result = score(sentences, args.dataset, out)
    result['recognizer'] = args.recognizer
    (args.run_dir / f'report-{out.stem.removeprefix("predictions-")}.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ('dataset', 'recognizer', 'sentences', 'missing', 'errors', 'unplaced_predictions', 'micro')}, indent=1))


if __name__ == '__main__':
    main()
