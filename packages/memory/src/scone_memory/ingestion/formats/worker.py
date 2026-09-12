"""Child-only parsing entrypoint; no shell execution or network lookups."""
from __future__ import annotations

import json
import sys
import os

from ...core.errors import InvalidInput
from .registry import extension
from .types import DocumentLimits, validate_document


def main() -> None:
    try:
        filename = sys.argv[1]
        limits = DocumentLimits.model_validate_json(sys.argv[2])
        data = sys.stdin.buffer.read(limits.max_input_bytes + 1)
        if not data or len(data) > limits.max_input_bytes:
            raise InvalidInput('document exceeds its input byte limit or is empty')
        from .office import OFFICE_EXTENSIONS, parse_office
        from .text import TEXT_EXTENSIONS, parse_text
        from .converters import CONVERTER_EXTENSIONS, parse_converted
        if os.name == 'posix':
            import resource
            resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_archive_bytes, limits.max_archive_bytes))
        suffix = extension(filename)
        if suffix.lstrip('.') in OFFICE_EXTENSIONS:
            parsed = parse_office(data, filename, limits)
        elif suffix in TEXT_EXTENSIONS:
            parsed = parse_text(data, filename, limits)
        elif suffix in CONVERTER_EXTENSIONS:
            parsed = parse_converted(data, filename, limits)
        else:
            raise InvalidInput('document format requires an explicitly configured parser')
        validate_document(parsed, limits)
        payload = parsed.model_dump_json()
    except Exception as error:
        payload = json.dumps({'error': str(error) if isinstance(error, InvalidInput) else 'document parser failed'})
    sys.stdout.buffer.write(payload.encode('utf-8'))


if __name__ == '__main__':
    main()
