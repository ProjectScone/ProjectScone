"""``scone doc-markdown FILE``: a document read and written back as Markdown.

Nothing is stored and no store is opened. The Markdown goes to stdout so it
can be piped; one line to stderr says what it holds and whether the byte
bound cut it, so a cut document is never mistaken for a whole one. With
``--json`` stdout is the whole record instead, spans and all.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TextIO

from ..core.errors import SconeError

if TYPE_CHECKING:
    from ..ingestion.formats.markdown_assembly import MarkdownDocument

#: The parser is built for every command, so the reader stack is imported only
#: when this one runs; the default here must equal MAX_MARKDOWN_BYTES (a test checks).
DEFAULT_MAX_BYTES = 8_000_000


def add_document_markdown_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("doc-markdown", help="read a document (DOCX, HTML, PDF, ...) and write it as Markdown; "
                                            "no store is opened")
    p.add_argument("file", help="the document; its extension chooses the reader")
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                   help=f"Markdown written at most; the receipt says where it cut (default {DEFAULT_MAX_BYTES})")


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def summary(result: "MarkdownDocument") -> str:
    record = result.record()
    blocks: dict[str, int] = record["blocks"]  # type: ignore[assignment]
    named = [_plural(blocks[key], label) for key, label in (("heading", "heading"), ("list_item", "list item"),
                                                            ("table", "table")) if blocks.get(key)]
    line = (f"doc-markdown: {_plural(sum(blocks.values()), 'block')} from {_plural(result.segments, 'segment')}"
            + (f" ({', '.join(named)})" if named else "")
            + ("" if result.structure_declared else "; the reader declared no headings, lists or tables"))
    if result.cut_at is None:
        return line + "; not cut"
    return (line + f"; cut at extracted byte {result.cut_at} by the {result.max_bytes}-byte bound; "
            f"{_plural(result.segments_omitted, 'segment')} not wholly written")


def document_markdown_command(args: argparse.Namespace, out: TextIO) -> int:
    from ..ingestion.formats.markdown_assembly import assemble_markdown
    from ..ingestion.formats.registry import BuiltinDocumentParser
    from ..ingestion.formats.types import DocumentLimits

    limit = DocumentLimits().max_input_bytes
    try:
        with open(args.file, "rb") as stream:
            data = stream.read(limit + 1)
    except OSError as error:
        print(f"error: {args.file} cannot be read: {error.strerror}", file=sys.stderr)
        return 2
    try:
        # One byte past the limit is enough for the reader to refuse it.
        parsed = asyncio.run(BuiltinDocumentParser().parse(data, Path(args.file).name))
        result = assemble_markdown(parsed, max_bytes=args.max_bytes)
    except SconeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps({"file": args.file, **result.record()}, ensure_ascii=False, indent=2), file=out)
    else:
        out.write(result.markdown + "\n" if result.markdown else "")
    print(summary(result), file=sys.stderr)
    return 0
