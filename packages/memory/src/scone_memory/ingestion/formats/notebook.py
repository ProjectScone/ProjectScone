"""Read notebook cells and saved text observations without executing code."""
from __future__ import annotations

import json
import re
from typing import cast

from ...core.errors import InvalidInput
from .text import _Collector, _bad_constant, _decode, _html, _unique_object
from .types import DocumentLimits, ParsedDocument, validate_document


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidInput('notebook expects an object')
    return cast(dict[str, object], value)


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return ''.join(cast(list[str], value))
    raise InvalidInput('notebook text must be a string or a list of strings')


def _output(value: object, pointer: str, metadata: dict[str, str], out: _Collector) -> bool:
    output = _object(value)
    kind = output.get('output_type')
    evidence = {**metadata, 'evidence_kind': 'saved_output'}
    if kind == 'stream':
        text = _text(output.get('text'))
        out.add(text, pointer + '/text', {**evidence, 'output_type': 'stream'})
        return bool(text.strip())
    if kind == 'error':
        traceback = output.get('traceback', [])
        if not isinstance(traceback, list) or not all(isinstance(line, str) for line in traceback):
            raise InvalidInput('notebook traceback must be a list of strings')
        text = '\n'.join(cast(list[str], traceback))
        evidence['output_type'] = 'error'
        if text.strip():
            out.add(text, pointer + '/traceback', evidence)
            return True
        present = False
        for field in ('ename', 'evalue'):
            text = _text(output.get(field, ''))
            out.add(text, pointer + '/' + field, evidence)
            present = present or bool(text.strip())
        return present
    if kind not in {'execute_result', 'display_data'}:
        raise InvalidInput('notebook contains an unsupported output type')
    data = _object(output.get('data'))
    evidence['output_type'] = str(kind)
    for media_type in ('text/plain', 'text/html'):
        if media_type not in data:
            continue
        text = _text(data[media_type])
        if media_type == 'text/html':
            visible = _Collector(out.limits)
            _html(text, visible)
            text = '\n'.join(segment.text for segment in visible.segments)
        if not text.strip():
            continue
        out.add(text, pointer + '/data/' + media_type.replace('/', '~1'),
                {**evidence, 'mime_type': media_type})
        return True
    return False


def parse_notebook(data: bytes, limits: DocumentLimits) -> ParsedDocument:
    try:
        notebook = _object(json.loads(_decode(data), object_pairs_hook=_unique_object,
                                     parse_constant=_bad_constant))
    except (ValueError, RecursionError):
        raise InvalidInput('notebook contains malformed or excessively nested JSON') from None
    if type(notebook.get('nbformat')) is not int or notebook['nbformat'] != 4:
        raise InvalidInput('notebook reader requires format version four')
    cells = notebook.get('cells')
    if not isinstance(cells, list) or len(cells) > limits.max_segments:
        raise InvalidInput('notebook cells must be a list within the segment limit')
    out = _Collector(limits)
    skipped = 0
    for index, value in enumerate(cells):
        out.check()
        cell = _object(value)
        kind = cell.get('cell_type')
        if kind not in {'markdown', 'code', 'raw'}:
            raise InvalidInput('notebook contains an unsupported cell type')
        pointer = f'#/cells/{index}'
        metadata = {'cell_index': str(index), 'cell_type': str(kind), 'evidence_kind': 'cell_source'}
        if 'id' in cell:
            identifier = cell['id']
            if not isinstance(identifier, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier) is None:
                raise InvalidInput('notebook cell id is invalid')
            metadata['cell_id'] = identifier
        out.add(_text(cell.get('source')), pointer + '/source', metadata)
        if kind != 'code':
            continue
        outputs = cell.get('outputs', [])
        if not isinstance(outputs, list) or len(outputs) > limits.max_segments:
            raise InvalidInput('notebook outputs must be a list within the segment limit')
        for number, output in enumerate(outputs):
            out.check()
            if not _output(output, f'{pointer}/outputs/{number}', metadata, out):
                skipped += 1
    if not out.segments:
        raise InvalidInput('notebook contains no extractable text')
    parsed = ParsedDocument(format='ipynb', parser='scone-notebook-v1', segments=tuple(out.segments),
        metadata={'notebook_format': '4', 'cells': str(len(cells)), 'outputs_without_text': str(skipped)})
    validate_document(parsed, limits)
    return parsed
