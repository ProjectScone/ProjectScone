"""Explicit, self-managed Tesseract baseline. No installation or network code."""
from __future__ import annotations

import math
import re
import struct
import zlib

from pydantic import ValidationError

from ..core.errors import InvalidInput
from .process import run_bounded
from .types import OcrRegion, OcrResult


HEADER = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext'


def png_dimensions(image: bytes, max_pixels: int) -> tuple[int, int]:
    """Bound the declared raster before a native decoder sees its compressed data."""
    if (not isinstance(image, bytes) or not 33 <= len(image) <= 80_000_000
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
                score=score / 100., block=block, line=line_number))
            if len(regions) > max_regions:
                raise InvalidInput('OCR exceeded its region limit')
        return OcrResult(engine=engine, width=width, height=height, regions=tuple(regions))
    except (ValueError, UnicodeError, ValidationError) as error:
        raise InvalidInput('OCR returned invalid TSV geometry, scores or text') from error


class TesseractOcr:
    """Use installed language data. PSM 3 detects blocks; PSM 6 assumes one block."""
    def __init__(self, *, executable: str = 'tesseract', language: str = 'eng', page_segmentation: int = 3):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,32}(\+[A-Za-z0-9_]{1,32}){0,7}', language):
            raise InvalidInput('OCR language must name installed language data')
        if type(page_segmentation) is not int or page_segmentation not in (3, 6, 11, 12):
            raise InvalidInput('OCR page segmentation must be 3, 6, 11 or 12')
        if not isinstance(executable, str) or not executable or '\x00' in executable:
            raise InvalidInput('OCR executable must be configured explicitly')
        self.executable = executable
        self.language = language
        self.page_segmentation = page_segmentation

    async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                        max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult:
        width, height = png_dimensions(image, max_pixels)
        if type(max_regions) is not int or not 1 <= max_regions <= 50_000:
            raise InvalidInput('OCR region limit must be between 1 and 50000')
        raw = await run_bounded([self.executable, 'stdin', 'stdout', '-l', self.language,
            '--psm', str(self.page_segmentation), 'tsv'], image, timeout=timeout_seconds,
            max_output=min(8_000_000, max_regions * 1024 + 4096), label='OCR process')
        return parse_tsv(raw, width=width, height=height, max_regions=max_regions,
            engine=f'tesseract:{self.language}:psm{self.page_segmentation}')
