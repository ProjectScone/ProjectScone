"""A layout engine behind an executable that answers in JSON lines.

Any layout model -- PP-Structure, a YOLO layout detector, a service
wrapped in a script -- can label a page for this framework by being
an executable that reads a PNG on standard input and writes, on
standard output, one JSON object per line: first the page's size,
then one region per line::

    {"width": 1240, "height": 1754}
    {"label": "paragraph_title", "box": [80, 120, 900, 160], "score": 0.93, "order": 0}
    {"label": "text", "box": [80, 180, 1160, 620], "score": 0.99, "order": 1}

Boxes are pixels on that image, ``score`` and ``order`` optional. A
label is read in this framework's vocabulary or in PP-Structure's or
deepdoc's words (``ocr.labels.ENGINE_LABELS``); a label with no name here
is dropped and counted. The executable is run once per page under the
page's deadline and a byte bound, like Tesseract is, and nothing about
it is trusted beyond that: a box outside the image, a size that does not
match the PNG, or a line that is not JSON is a refused page, not a
guess. Configured by ``SCONE_DOCUMENT_LAYOUT_EXECUTABLE``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.errors import InvalidInput
from .labels import as_label
from .process import run_bounded
from .tesseract import png_dimensions
from .types import LayoutRegion, LayoutResult

#: Bytes of output read from the executable before the page is refused.
MAX_OUTPUT = 8_000_000
STRATEGY = 'layout-json-v1'


class JsonLayoutEngine:
    """The executable at ``executable``, named as ``name`` in receipts."""

    def __init__(self, executable: str, *, name: str = 'layout-json') -> None:
        path = Path(executable)
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise InvalidInput('the layout executable must be an installed absolute executable')
        if not name or len(name) > 96:
            raise InvalidInput('the layout engine name must be 1 to 96 characters')
        self.executable = str(path)
        self.name = name

    async def analyse(self, image: bytes, *, max_pixels: int = 20_000_000,
                      max_regions: int = 10_000, timeout_seconds: float = 30.0) -> LayoutResult:
        width, height = png_dimensions(image, max_pixels)
        raw = await run_bounded([self.executable], image, timeout=timeout_seconds, max_output=MAX_OUTPUT,
                                label='layout engine')
        lines = [line for line in raw.decode('utf-8', 'replace').split('\n') if line.strip()]
        if not lines:
            raise InvalidInput('the layout engine answered nothing')
        try:
            head = json.loads(lines[0])
            records = [json.loads(line) for line in lines[1:]]
        except ValueError as error:
            raise InvalidInput('the layout engine answered with something that is not JSON') from error
        if not isinstance(head, dict) or (head.get('width'), head.get('height')) != (width, height):
            raise InvalidInput('the layout engine did not name the page it was given')
        if len(records) > max_regions:
            raise InvalidInput('the layout engine answered with more regions than the page allows')
        regions: list[LayoutRegion] = []
        dropped = 0
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get('label'), str):
                raise InvalidInput('a layout region has no label')
            label = as_label(record['label'])
            if label is None:
                dropped += 1
                continue
            box = record.get('box')
            if (not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box)
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                raise InvalidInput('a layout region box is not a rectangle on the page')
            score = record.get('score')
            order = record.get('order')
            try:
                regions.append(LayoutRegion(
                    label=label, box=(box[0] / width, box[1] / height, box[2] / width, box[3] / height),
                    score=float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                    order=order if isinstance(order, int) and not isinstance(order, bool) else None))
            except ValueError as error:
                raise InvalidInput('a layout region is out of bounds') from error
        return LayoutResult(engine=self.name, width=width, height=height, regions=tuple(regions), dropped=dropped)
