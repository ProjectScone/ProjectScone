"""Isolated optional dependency entry point. Never imported during engine startup."""
from __future__ import annotations

from io import BytesIO
import json
import logging
import sys
from typing import TYPE_CHECKING, Literal

from ..core.errors import InvalidInput
from .pdf import ParsedPdf, PdfEncryption, PdfLimits, PdfPage, Restriction
from .pdf_layout import lay_out_page, running_rows

if TYPE_CHECKING:  # pragma: no cover - typing only; the worker imports pypdf when it runs
    import pypdf

#: Bookmarks read from one PDF; past it the outline is read to here and said to be capped.
MAX_OUTLINE_ITEMS = 2_000
#: Levels of bookmarks followed.
MAX_OUTLINE_DEPTH = 8
MAX_TITLE_CHARS = 256


def _outline(reader: object) -> tuple[list[tuple[int, str, int]], Literal['none', 'read', 'capped', 'unreadable']]:
    """The bookmarks as (level, title, zero-based page) in outline order, and how they were read."""
    try:
        items = reader.outline  # type: ignore[attr-defined]
    except Exception:
        return [], 'unreadable'
    if not items:
        return [], 'none'
    found: list[tuple[int, str, int]] = []
    capped = False

    def walk(level_items: list, level: int) -> None:
        nonlocal capped
        for item in level_items:
            if capped:
                return
            if isinstance(item, list):
                if level + 1 >= MAX_OUTLINE_DEPTH:
                    capped = True
                    return
                walk(item, level + 1)
                continue
            if len(found) >= MAX_OUTLINE_ITEMS:
                capped = True
                return
            try:
                page = reader.get_destination_page_number(item)  # type: ignore[attr-defined]
            except Exception:
                continue
            title = ' '.join(str(getattr(item, 'title', '') or '').split())
            if page is None or page < 0 or not title:
                continue
            if len(title) > MAX_TITLE_CHARS:
                title = title[:MAX_TITLE_CHARS - 1] + '…'
            found.append((level, title, page))

    try:
        walk(list(items), 0)
    except Exception:
        return [], 'unreadable'
    return found, 'capped' if capped else 'read'


def _sections(entries: list[tuple[int, str, int]], count: int) -> list[tuple[str, ...]]:
    """For each page, the last bookmark at each level that begins on or before it."""
    ordered = sorted(enumerate(entries), key=lambda pair: (pair[1][2], pair[0]))
    sections: list[tuple[str, ...]] = []
    stack: list[str] = []
    position = 0
    for page in range(count):
        while position < len(ordered) and ordered[position][1][2] <= page:
            level, title, _ = ordered[position][1]
            stack = stack[:level] + [title]
            position += 1
        sections.append(tuple(stack))
    return sections


def extract(data: bytes, limits: PdfLimits, *, allow_empty: bool = False, metadata_only: bool = False,
            allow_text_errors: bool = False, columns: bool = False) -> ParsedPdf:
    import pypdf

    try:
        reader = pypdf.PdfReader(BytesIO(data), strict=True)
        encryption = _open_encrypted(reader) if reader.is_encrypted else None
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
                except pypdf.errors.DependencyError as error:
                    # An AES-128 file opens without the cipher and needs it
                    # for its pages: a missing package is not a broken page.
                    raise InvalidInput(NEEDS_CRYPTOGRAPHY) from error
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
                laid = lay_out_page(text.split('\n'), offset, drop=left_out[number - 1], first_page=number == 1)
                text = laid.text
                updates = {'reading_order': laid.receipt, 'running': laid.running, 'regions': laid.regions,
                           'labels': laid.labels,
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
        entries, outline = _outline(reader)
        sections = _sections(entries, len(pages))
        pages = [page.model_copy(update={'section': sections[index]}) if sections[index] else page
                 for index, page in enumerate(pages)]
        # An extraction carrying bookmark sections names that, so its manifest is told apart from one before them.
        marked = '+outline-v1' if outline != 'none' else ''
        return ParsedPdf(text='\n\n'.join(texts), parser=f'pypdf/{pypdf.__version__}:{strategy}{marked}',
                         pages=tuple(pages), outline=outline, encryption=encryption)
    except InvalidInput:
        raise
    except pypdf.errors.DependencyError as error:
        # Reading an AES-256 file tries the empty password as it opens, and
        # its page tree, streams and pages all need the cipher: a missing
        # package is named, not reported as a broken file.
        raise InvalidInput(NEEDS_CRYPTOGRAPHY) from error
    except Exception as error:
        raise InvalidInput('PDF is corrupt or uses unsupported text/layout encoding') from error


NEEDS_CRYPTOGRAPHY = 'decrypting this PDF needs the cryptography package'
#: The PDF's user access permission bits, by what each lets a reader do.
_PERMISSIONS: tuple[tuple[str, Restriction], ...] = (
    ('PRINT', 'print'), ('MODIFY', 'modify'), ('EXTRACT', 'extract'), ('ADD_OR_MODIFY', 'annotate'),
    ('FILL_FORM_FIELDS', 'fill_forms'), ('EXTRACT_TEXT_AND_GRAPHICS', 'extract_for_accessibility'),
    ('ASSEMBLE_DOC', 'assemble'), ('PRINT_TO_REPRESENTATION', 'print_high_quality'))


def _open_encrypted(reader: pypdf.PdfReader) -> PdfEncryption:
    """Open an encrypted file with the empty password -- an owner password
    alone restricts what a reader may do and hides nothing -- and say what
    was restricted; a file that needs a user password is refused."""
    import pypdf
    from pypdf.constants import UserAccessPermissions

    try:
        matched = reader.decrypt('')
    except pypdf.errors.DependencyError as error:
        raise InvalidInput(NEEDS_CRYPTOGRAPHY) from error
    if matched == pypdf.PasswordType.NOT_DECRYPTED:
        raise InvalidInput('encrypted PDFs need their user password; provide a decrypted original')
    allowed = reader.user_access_permissions
    if allowed is None or not reader.are_permissions_valid:
        # Permissions the file cannot vouch for (a tampered AES-256 record)
        # are all reported restricted, so a caller honouring them errs safe.
        restricted = tuple(name for _, name in _PERMISSIONS)
    else:
        restricted = tuple(name for flag, name in _PERMISSIONS if UserAccessPermissions[flag] not in allowed)
    return PdfEncryption(matched='owner' if matched == pypdf.PasswordType.OWNER_PASSWORD else 'user',
                         restricted=restricted)


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
