"""Installed parser dependencies, separate from file validity and model availability."""
from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import sys

from .office import OFFICE_EXTENSIONS
from .text import TEXT_EXTENSIONS


def document_formats() -> dict[str, dict[str, object]]:
    formats = {suffix: {'available': True, 'parser': 'text'} for suffix in TEXT_EXTENSIONS}
    formats['.xml'] = {'available': find_spec('defusedxml') is not None, 'parser': 'xml', 'requires': 'documents extra'}
    formats.update({'.'+suffix: {'available': find_spec('defusedxml') is not None,
                                'parser': 'office-xml', 'requires': 'documents extra'} for suffix in OFFICE_EXTENSIONS})
    formats['.pdf'] = {'available': find_spec('pypdf') is not None, 'parser': 'pdf-text', 'requires': 'pdf extra; OCR is opt-in'}
    for suffix, module in (('.rtf', 'striprtf'), ('.xls', 'python_calamine'), ('.xlsb', 'python_calamine'), ('.msg', 'extract_msg')):
        formats[suffix] = {'available': find_spec(module) is not None, 'parser': module, 'requires': 'document-converters extra'}
    converter = os.environ.get('SCONE_MEMORY_DOCUMENT_CONVERTER', '')
    configured = bool(converter and Path(converter).is_absolute() and os.access(converter, os.X_OK))
    formats['.doc'] = {'available': configured or (sys.platform == 'darwin' and os.access('/usr/bin/textutil', os.X_OK)),
                       'parser': 'converter', 'requires': 'configured offline converter or macOS textutil'}
    formats['.ppt'] = {'available': configured, 'parser': 'converter', 'requires': 'configured offline converter'}
    return dict(sorted(formats.items()))
