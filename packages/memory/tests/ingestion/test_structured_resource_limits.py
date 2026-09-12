"""Bound allocations before recursive path construction or XML tree expansion."""
import tracemalloc

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser, DocumentLimits
from scone_memory.ingestion.formats.text import parse_text


@pytest.mark.parametrize('body', [
    '<root>Visible' + '<n/>' * 200_000 + '</root>',
    '<root xmlns="' + 'x' * 1025 + '">Visible</root>',
], ids=['nodes', 'namespace'])
async def test_standalone_xml_uses_construction_limits_before_returning_visible_text(body):
    with pytest.raises(InvalidInput, match='XML.*limit'):
        await BuiltinDocumentParser().parse(body.encode(), 'source.xml')


def test_nested_json_keys_are_rejected_before_allocating_every_path_prefix():
    # Under 1 MiB of JSON previously accumulated over 40 MiB of path strings
    # before the leaf's locator check. The limit leaves room for parser overhead.
    raw = (('{"' + 'k' * 10_000 + '":') * 90 + '"value"' + '}' * 90).encode()
    tracemalloc.start()
    try:
        with pytest.raises(InvalidInput, match='locator'):
            parse_text(raw, 'source.json', DocumentLimits())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * 1024 * 1024, f'oversized JSON paths allocated {peak} bytes before rejection'
