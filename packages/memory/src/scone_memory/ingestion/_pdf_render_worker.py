"""Isolated PDF rasterization. No OCR executables are launched by this worker."""
from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
import json
import logging
import math
import sys
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput


class RenderOptions(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    page: int = Field(ge=1, le=1000)
    dpi: int = Field(ge=72, le=300)
    max_pixels: int = Field(ge=1, le=20_000_000)


def render(data: bytes, options: RenderOptions) -> bytes:
    import pypdfium2 as pdfium

    # The upstream constructor is unannotated; this adapter accepts bytes only.
    document_from_bytes = cast(Callable[[bytes], pdfium.PdfDocument], pdfium.PdfDocument)
    with document_from_bytes(data) as document:
        if options.page > len(document):
            raise InvalidInput('PDF render page does not exist')
        page = document[options.page - 1]
        try:
            width, height = page.get_size()
            scale = options.dpi / 72.
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise InvalidInput('PDF render dimensions are invalid')
            if math.ceil(width * scale) * math.ceil(height * scale) > options.max_pixels:
                raise InvalidInput('PDF rendering exceeds its pixel limit')
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil()
                try:
                    output = BytesIO()
                    image.save(output, format='PNG', dpi=(options.dpi, options.dpi))
                    return output.getvalue()
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()


def main() -> None:
    logging.disable(logging.CRITICAL)
    try:
        options = RenderOptions.model_validate_json(sys.argv[1])
        data = sys.stdin.buffer.read(25 * 1024 * 1024 + 1)
        if len(data) > 25 * 1024 * 1024 or not data.startswith(b'%PDF-'):
            raise InvalidInput('PDF rendering input is invalid or exceeds its byte limit')
        payload = render(data, options)
    except Exception as error:
        message = str(error) if isinstance(error, InvalidInput) else 'PDF rasterization failed'
        payload = json.dumps({'error': message}).encode()
    sys.stdout.buffer.write(payload)


if __name__ == '__main__':
    main()
