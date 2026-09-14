"""Isolated optional dependency entry point. Never imported during engine startup."""
from __future__ import annotations

from io import BytesIO
import json
import logging
import sys

from ..core.errors import InvalidInput
from .pdf import ParsedPdf, PdfLimits, PdfPage
from .pdf_layout import lay_out_page, running_rows


def extract(data: bytes, limits: PdfLimits, *, allow_empty: bool = False, metadata_only: bool = False,
            allow_text_errors: bool = False, columns: bool = False) -> ParsedPdf:
    import pypdf

    try:
        reader = pypdf.PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise InvalidInput('encrypted PDFs are unsupported; provide a decrypted original')
        count = len(reader.pages)
        if count < 1 or count > limits.max_pages:
            raise InvalidInput('PDF exceeds its page limit or has no pages')
        texts: list[str] = []
        geometry: list[tuple[float, float, int]] = []
        recovered_text_error = False
        for page in reader.pages:
            text = ''
            if not metadata_only and page.get('/Contents') is not None:
                try:
                    text = page.extract_text(extraction_mode='layout', layout_mode_space_vertically=False,
                                             layout_mode_strip_rotated=False).rstrip()
                except (InvalidInput, MemoryError, RecursionError, pypdf.errors.LimitReachedError):
                    raise
                except Exception:
                    if not allow_text_errors:
                        raise
                    recovered_text_error = True
            texts.append(text)
            geometry.append((float(page.mediabox.width) * float(page.user_unit),
                             float(page.mediabox.height) * float(page.user_unit), int(page.rotation) % 360))
        # Running lines are judged across the document before any page is
        # laid out, since a header is a header by recurring.
        left_out = running_rows([text.split('\n') for text in texts]) if columns else [{} for _ in texts]
        pages: list[PdfPage] = []
        offset = 0
        for number, text in enumerate(texts, 1):
            if number > 1:
                offset += 2
            updates: dict[str, object] = {}
            if columns and text:
                laid = lay_out_page(text.split('\n'), offset, drop=left_out[number - 1])
                text = laid.text
                updates = {'reading_order': laid.receipt, 'running': laid.running, 'regions': laid.regions,
                           **({'region_geometry': 'normalized_text_grid'} if laid.regions else {})}
            end = offset + len(text.encode('utf-8'))
            if end > limits.max_text_bytes:
                raise InvalidInput('PDF extracted text exceeds its byte limit')
            width_points, height_points, rotation = geometry[number - 1]
            pages.append(PdfPage(number=number, start=offset, end=end, width_points=width_points,
                height_points=height_points, rotation=rotation, empty=not text.strip(), **updates))  # type: ignore[arg-type]
            texts[number - 1] = text
            offset = end
        if not allow_empty and all(page.empty for page in pages):
            raise InvalidInput('PDF has no extractable text; OCR may be required and is not enabled')
        strategy = 'pages-v1' if metadata_only else ('layout-fallback-v1' if recovered_text_error else 'layout-v1')
        if columns and not metadata_only:
            strategy += '+grid-columns-v1'
        return ParsedPdf(text='\n\n'.join(texts), parser=f'pypdf/{pypdf.__version__}:{strategy}', pages=tuple(pages))
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
            metadata_only='--metadata-only' in sys.argv[2:],
            allow_text_errors='--allow-text-errors' in sys.argv[2:],
            columns='--columns' in sys.argv[2:]).model_dump_json()
    except Exception as error:
        message = str(error) if isinstance(error, InvalidInput) else 'PDF parser failed'
        payload = json.dumps({'error': message})
    sys.stdout.buffer.write(payload.encode('utf-8'))


if __name__ == '__main__':
    main()
