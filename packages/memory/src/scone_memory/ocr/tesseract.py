"""Explicit, self-managed Tesseract baseline. No installation or network code."""
from __future__ import annotations

import math
import re
import struct
import time
import zlib

from pydantic import ValidationError

from ..core.errors import InvalidInput
from .process import python_worker, run_bounded
from .types import OcrRegion, OcrResult


HEADER = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext'


def png_dimensions(image: bytes, max_pixels: int) -> tuple[int, int]:
    """Bound the declared raster before a native decoder sees its compressed data."""
    if (not isinstance(image, bytes) or not 33 <= len(image) <= MAX_PNG_BYTES
            or image[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
            or zlib.crc32(image[12:29]) != int.from_bytes(image[29:33], 'big')):
        raise InvalidInput('OCR requires a bounded PNG image with a valid header')
    width, height = struct.unpack('>II', image[16:24])
    if type(max_pixels) is not int or not 1 <= max_pixels <= 20_000_000 or min(width, height) < 1 or width * height > max_pixels:
        raise InvalidInput('OCR image exceeds its pixel limit')
    return width, height


def parse_tsv(raw: bytes, *, width: int, height: int, max_regions: int, engine: str) -> OcrResult:
    try:
        rows = raw.decode('utf-8').split('\n')
        if rows and rows[-1] == '':
            rows.pop()
        rows = [row.removesuffix('\r') for row in rows]
        if not rows or rows[0] != HEADER:
            raise ValueError('invalid header')
        regions: list[OcrRegion] = []
        lines: dict[tuple[int, int, int], int] = {}
        for row in rows[1:]:
            fields = row.split('\t', 11)
            if len(fields) != 12:
                raise ValueError('invalid row')
            if int(fields[0]) != 5:
                continue
            text = fields[11].strip()
            if not text:
                continue
            if int(fields[1]) != 1:
                raise ValueError('multiple image pages are unsupported')
            block, paragraph, line = map(int, fields[2:5])
            left, top, box_width, box_height = map(int, fields[6:10])
            score = float(fields[10])
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError('invalid score')
            line_key = (block, paragraph, line)
            line_number = lines.setdefault(line_key, len(lines))
            regions.append(OcrRegion(text=text,
                box=(left / width, top / height, (left + box_width) / width, (top + box_height) / height),
                score=score / 100., block=block, paragraph=paragraph, line=line_number))
            if len(regions) > max_regions:
                raise InvalidInput('OCR exceeded its region limit')
        return OcrResult(engine=engine, width=width, height=height, regions=tuple(regions))
    except (ValueError, UnicodeError, ValidationError) as error:
        raise InvalidInput('OCR returned invalid TSV geometry, scores or text') from error


#: The largest PNG the engine reads, and so the largest the turning worker must read.
MAX_PNG_BYTES = 80_000_000
#: The longest outcome the engine name carries with orientation on.
_LONGEST_OUTCOME = ':osd-unknown'
#: Orientation confidence, as Tesseract's detection reports it, from which a page is turned.
ORIENTATION_CONFIDENCE = 2.0
_ROTATE = re.compile(rb'^Rotate:\s*(\d+)\s*$', re.M)
_ORIENTATION_CONFIDENCE = re.compile(rb'^Orientation confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*$', re.M)


def parse_osd(raw: bytes) -> tuple[int, float] | None:
    """How far Tesseract's orientation detection says to turn the page clockwise, and how
    sure it is; nothing when it gave no quarter turn and confidence to act on."""
    turn, sure = _ROTATE.search(raw), _ORIENTATION_CONFIDENCE.search(raw)
    if turn is None or sure is None or int(turn.group(1)) not in (0, 90, 180, 270):
        return None
    return int(turn.group(1)), float(sure.group(1))


def unrotated_box(box: tuple[float, float, float, float], degrees: int) -> tuple[float, float, float, float]:
    """A box read on the page turned clockwise by ``degrees``, given back in the page as it was."""
    x0, y0, x1, y1 = box
    if degrees == 90:
        return (y0, 1 - x1, y1, 1 - x0)
    if degrees == 180:
        return (1 - x1, 1 - y1, 1 - x0, 1 - y0)
    if degrees == 270:
        return (1 - y1, x0, 1 - y0, x1)
    return box


class TesseractOcr:
    """Use installed language data. PSM 3 detects blocks; PSM 6 assumes one block.

    With ``orientation``, Tesseract's orientation detection runs first. A page it is at
    least ``min_orientation_confidence`` sure is turned is turned in a bounded child
    process, read, and every box given back in the page as it was given. The engine
    name ends ``:rotated<degrees>``, ``:osd-upright``, ``:osd-unsure`` (a turn below the
    confidence) or ``:osd-unknown`` (the detection could not judge, as on a page with
    too few characters), so the result says which."""
    def __init__(self, *, executable: str = 'tesseract', language: str = 'eng', page_segmentation: int = 3,
                 orientation: bool = False, min_orientation_confidence: float = ORIENTATION_CONFIDENCE):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,32}(\+[A-Za-z0-9_]{1,32}){0,7}', language):
            raise InvalidInput('OCR language must name installed language data')
        if type(page_segmentation) is not int or page_segmentation not in (3, 6, 11, 12):
            raise InvalidInput('OCR page segmentation must be 3, 6, 11 or 12')
        if not isinstance(executable, str) or not executable or '\x00' in executable:
            raise InvalidInput('OCR executable must be configured explicitly')
        if type(orientation) is not bool:
            raise InvalidInput('OCR orientation must be true or false')
        if (isinstance(min_orientation_confidence, bool) or not isinstance(min_orientation_confidence, (int, float))
                or not math.isfinite(min_orientation_confidence) or min_orientation_confidence < 0):
            raise InvalidInput('OCR orientation confidence must be a finite number from 0')
        self.executable = executable
        self.language = language
        self.page_segmentation = page_segmentation
        if orientation and len(f'tesseract:{language}:psm{page_segmentation}{_LONGEST_OUTCOME}') > 96:
            raise InvalidInput('OCR language list is too long to name the engine with orientation on; '
                               'the engine name holds 96 characters')
        self.orientation = orientation
        self.min_orientation_confidence = float(min_orientation_confidence)
        self._has_osd: bool | None = None

    async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                        max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult:
        width, height = png_dimensions(image, max_pixels)
        if type(max_regions) is not int or not 1 <= max_regions <= 50_000:
            raise InvalidInput('OCR region limit must be between 1 and 50000')
        engine = f'tesseract:{self.language}:psm{self.page_segmentation}'
        deadline = time.monotonic() + timeout_seconds
        turn = 0
        if self.orientation:
            turn, outcome = await self._orientation(image, deadline)
            engine += f':{outcome}'
        read, read_width, read_height = image, width, height
        if turn:
            read = await run_bounded(python_worker('scone_memory.ocr._rotate_worker', str(turn), str(max_pixels)),
                image, timeout=_left(deadline), max_output=min(128_000_000, max_pixels * 4 + 1_048_576),
                label='OCR orientation process')
            read_width, read_height = png_dimensions(read, max_pixels)
        raw = await run_bounded([self.executable, 'stdin', 'stdout', '-l', self.language,
            '--psm', str(self.page_segmentation), 'tsv'], read, timeout=_left(deadline),
            max_output=min(8_000_000, max_regions * 1024 + 4096), label='OCR process')
        result = parse_tsv(raw, width=read_width, height=read_height, max_regions=max_regions, engine=engine)
        if not turn:
            return result
        regions = tuple(region.model_copy(update={'box': unrotated_box(region.box, turn)}) for region in result.regions)
        return OcrResult(engine=engine, width=width, height=height, regions=regions)

    async def _orientation_data(self, deadline: float) -> bool:
        """Whether Tesseract has its orientation data, asked once per engine."""
        if self._has_osd is None:
            try:
                self._has_osd = 'osd' in await _installed_languages(self.executable, deadline)
            except InvalidInput:
                return True  # the listing failed too; the recognition that follows will say why
        return self._has_osd

    async def _orientation(self, image: bytes, deadline: float) -> tuple[int, str]:
        """How far to turn the page, and the word the engine name carries for it."""
        try:
            raw = await run_bounded([self.executable, 'stdin', 'stdout', '--psm', '0'], image,
                timeout=_left(deadline), max_output=4096, label='OCR orientation detection')
        except InvalidInput:
            # Tesseract exits with an error on a page with too few characters to judge, and also
            # when its orientation data is not installed, which must be said, not read as the first.
            if not await self._orientation_data(deadline):
                raise InvalidInput("OCR orientation needs Tesseract's osd language data, which is not installed") from None
            return 0, 'osd-unknown'
        found = parse_osd(raw)
        if found is None:  # an answer without a quarter turn and a confidence to act on
            return 0, 'osd-unknown'
        turn, sure = found
        if turn == 0:
            return 0, 'osd-upright'
        if sure < self.min_orientation_confidence:
            return 0, 'osd-unsure'
        return turn, f'rotated{turn}'


async def _installed_languages(executable: str, deadline: float) -> set[str]:
    raw = await run_bounded([executable, '--list-langs'], b'', timeout=_left(deadline), max_output=65_536,
                            label='OCR language listing')
    return {line.strip() for line in raw.decode('utf-8', errors='replace').splitlines() if line.strip()}


def _left(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise InvalidInput('OCR exceeded its wall time limit')
    return left
