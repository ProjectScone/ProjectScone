"""Isolated optional dependency entry point. Never imported during engine startup."""
from __future__ import annotations

from io import BytesIO
import json
import logging
import sys

from ..core.errors import InvalidInput
from .pdf import ParsedPdf, PdfLimits, PdfPage


def extract(data: bytes, limits: PdfLimits, *, allow_empty: bool = False, metadata_only: bool = False) -> ParsedPdf:
    import pypdf

    try:
        reader = pypdf.PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise InvalidInput('encrypted PDFs are unsupported; provide a decrypted original')
        count = len(reader.pages)
        if count < 1 or count > limits.max_pages:
            raise InvalidInput('PDF exceeds its page limit or has no pages')
        texts: list[str] = []
        pages: list[PdfPage] = []
        offset = 0
        for number, page in enumerate(reader.pages, 1):
            text = '' if metadata_only else page.extract_text(extraction_mode='layout', layout_mode_space_vertically=False,
                                     layout_mode_strip_rotated=False).rstrip()
            if number > 1:
                offset += 2
            end = offset + len(text.encode('utf-8'))
            if end > limits.max_text_bytes:
                raise InvalidInput('PDF extracted text exceeds its byte limit')
            pages.append(PdfPage(number=number, start=offset, end=end,
                width_points=float(page.mediabox.width) * float(page.user_unit),
                height_points=float(page.mediabox.height) * float(page.user_unit),
                rotation=int(page.rotation) % 360, empty=not text.strip()))
            texts.append(text)
            offset = end
        if not allow_empty and all(page.empty for page in pages):
            raise InvalidInput('PDF has no extractable text; OCR may be required and is not enabled')
        return ParsedPdf(text='\n\n'.join(texts), parser=f'pypdf/{pypdf.__version__}:' + ('pages-v1' if metadata_only else 'layout-v1'), pages=tuple(pages))
    except InvalidInput:
        raise
    except Exception as error:
        raise InvalidInput('PDF is corrupt or uses unsupported text/layout encoding') from error


def main() -> None:
    logging.disable(logging.CRITICAL)
    try:
        limits = PdfLimits.model_validate_json(sys.argv[1])
        data = sys.stdin.buffer.read(limits.max_input_bytes + 1)
        if len(data) > limits.max_input_bytes:
            raise InvalidInput('PDF input exceeds its byte limit')
        payload = extract(data, limits, allow_empty='--allow-empty' in sys.argv[2:],
            metadata_only='--metadata-only' in sys.argv[2:]).model_dump_json()
    except Exception as error:
        message = str(error) if isinstance(error, InvalidInput) else 'PDF parser failed'
        payload = json.dumps({'error': message})
    sys.stdout.buffer.write(payload.encode('utf-8'))


if __name__ == '__main__':
    main()
