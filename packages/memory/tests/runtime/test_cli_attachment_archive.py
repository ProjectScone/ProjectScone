"""The archive profile is an explicit CLI choice and imports report evidence."""
import io
import json

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.runtime.cli import build_parser, run


async def test_cli_attachment_archive_round_trip_and_default_profile():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    blob = await engine.attach('source', b'CLI evidence', media_type='text/plain')
    await engine.remember('source', 'CLI source', attachment_ids=[blob.attachment_id])
    parser = build_parser()
    ordinary = io.StringIO()
    await run(parser.parse_args(['export', '--space', 'source']), engine, io.StringIO(), ordinary)
    assert json.loads(ordinary.getvalue().splitlines()[0])['profile'] == 'scone.archive/1'
    output = io.StringIO()
    await run(parser.parse_args(['export', '--space', 'source', '--include-attachments']),
              engine, io.StringIO(), output)
    result = io.StringIO()
    await run(parser.parse_args(['import', '--space', 'target', '--json']),
              engine, io.StringIO(output.getvalue()), result)
    summary = json.loads(result.getvalue())
    assert summary['profile'] == 'scone.archive/2'
    assert summary['attachments'] == 1 and summary['attachment_links'] == 1
    assert await engine.attachment('target', blob.attachment_id) == (blob, b'CLI evidence')
    await engine.close()
