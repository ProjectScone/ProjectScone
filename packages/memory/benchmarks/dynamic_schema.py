"""Dynamic-schema extraction on this repository's own documents, with a local model.

Run with Python 3.14 and PYTHONPATH=packages/memory/src, from packages/memory.

``run`` (model calls): stores every document in ``docs/`` in an in-memory
engine, takes a seeded sample of ``--chunks`` chunks of at least
``--min-chars`` characters (or the chunks saved by an earlier run, with
``--chunks-from``: a seeded sample moves when a document changes), and
stores each sampled chunk as an episode of
its own in two fresh engines. The suggested vocabulary is ``--vocabulary``:
``scone``, Scone's fixed schema (the entity kinds ``entities.kinds.EntityKind``
names and the predicates its kind hints know), or ``llamaindex``, the
default entities and relations of LlamaIndex's ``SchemaLLMPathExtractor``
as terms. One engine runs the pass with new types allowed, the other with
them off, on the same chunks. Saved to ``--out``: both reports, every raw
reply, and the sampled chunks.

``replay`` (no model): runs the pass again over the saved chunks with the
saved replies in call order (a call that failed fails again), as the pass
is now and with a rule turned back: ``no-settle`` does not settle a quote's whitespace, ``old-clause``
also reads a quote's clause as it was before the full-stop fix. The same
replies, so a difference is the rule's.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import get_args

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.questions import store_corpus
from scone_memory.entities import kinds as kind_hints
from scone_memory.ingestion import distill, dynamic_schema
from scone_memory.ingestion.dynamic_schema import extract_dynamic_schema
from scone_memory.providers.llm import ChatError, FakeChat, OpenAICompatibleChat

SPACE = "corpus"
DOCS = Path(__file__).resolve().parent.parent / "docs"


class Recorded:
    """Forwards to the model and keeps every reply, in call order."""

    def __init__(self, chat: OpenAICompatibleChat) -> None:
        self.chat = chat
        self.replies: list[dict[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        reply = await self.chat.complete(system, user)
        self.replies.append({"user": user, "reply": reply})
        return reply

    async def complete_structured(self, system: str, user: str, schema: dict[str, object]) -> str:
        try:
            reply = await self.chat.complete_structured(system, user, schema)
        except Exception as error:
            self.replies.append({"user": user, "error": f"{type(error).__name__}: {error}"[:300]})
            raise
        self.replies.append({"user": user, "reply": reply})
        return reply


def fixed_schema(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if name == "llamaindex":
        from llama_index.core.indices.property_graph.transformations.schema_llm import (
            DEFAULT_ENTITIES,
            DEFAULT_RELATIONS,
        )

        return (tuple(str(kind).lower() for kind in get_args(DEFAULT_ENTITIES)),
                tuple(str(relation).lower() for relation in get_args(DEFAULT_RELATIONS)))
    kinds = tuple(get_args(kind_hints.EntityKind))
    predicates = tuple(sorted(set(kind_hints._AS_SUBJECT) | set(kind_hints._AS_OBJECT)))
    return kinds, predicates


async def sample(count: int, seed: int, min_chars: int) -> list[dict[str, object]]:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with tempfile.TemporaryDirectory() as root:
        for path in sorted(DOCS.glob("*.md")):
            shutil.copyfile(path, Path(root) / path.name)
        stored = await store_corpus(engine, SPACE, root)
    rows = []
    for episode in await engine.documents.recent_episodes(SPACE, (await engine.documents.counts(SPACE)).episodes):
        for chunk in await engine.documents.chunks_of(SPACE, episode.episode_id):
            if len(chunk.text.strip()) >= min_chars:
                rows.append({"source": episode.source, "ordinal": chunk.ordinal, "text": chunk.text})
    rows.sort(key=lambda row: (str(row["source"]), int(row["ordinal"])))
    print(f"stored {stored}; {len(rows)} chunk(s) of at least {min_chars} characters", flush=True)
    return random.Random(seed).sample(rows, count)


async def stored(chunks: list[dict[str, object]]) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=8000,
                                events=InMemoryEventLog()).open()
    for row in chunks:
        await engine.remember(SPACE, str(row["text"]), kind="file", source=f"{row['source']}#{row['ordinal']}")
    count = (await engine.documents.counts(SPACE)).chunks
    if count != len(chunks):
        raise SystemExit(f"expected one chunk per sampled chunk, stored {count}")
    return engine


async def arm(chunks: list[dict[str, object]], chat: OpenAICompatibleChat, *, allow_new_types: bool,
              model: str, vocabulary: str) -> dict[str, object]:
    engine = await stored(chunks)
    recorded = Recorded(chat)
    kinds, predicates = fixed_schema(vocabulary)
    started = time.perf_counter()
    report = await extract_dynamic_schema(engine, SPACE, recorded, entity_kinds=kinds, predicates=predicates,
                                          allow_new_types=allow_new_types, max_calls=len(chunks), model_name=model)
    print(f"allow_new_types={allow_new_types} in {time.perf_counter() - started:.0f}s: {report.text()}", flush=True)
    return {"record": report.record(), "replies": recorded.replies}


async def run(args: argparse.Namespace) -> None:
    chunks = (json.loads(Path(args.chunks_from).read_text())["chunks"] if args.chunks_from
              else await sample(args.chunks, args.seed, args.min_chars))
    chat = OpenAICompatibleChat(args.url, args.model, timeout=args.timeout)
    result: dict[str, object] = {
        "model": args.model, "seed": args.seed, "min_chars": args.min_chars, "vocabulary": args.vocabulary,
        "chunks": chunks, "suggested": dict(zip(("entity_kinds", "predicates"), fixed_schema(args.vocabulary)))}
    result["open"] = await arm(chunks, chat, allow_new_types=True, model=args.model, vocabulary=args.vocabulary)
    result["closed"] = await arm(chunks, chat, allow_new_types=False, model=args.model, vocabulary=args.vocabulary)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"saved {args.out}", flush=True)


def _old_clause(source: str, start: int, end: int) -> str:
    """``distill._clause_around`` as it was before the full-stop fix."""
    left = max(source.rfind(mark, 0, start) for mark in ".?!;\n") + 1
    boundaries = [position for mark in ".?!;\n" if (position := source.find(mark, end)) >= 0]
    right = min(boundaries) + 1 if boundaries else len(source)
    return source[left:right]


async def replay(args: argparse.Namespace) -> None:
    saved = json.loads(Path(args.out).read_text())
    kinds, predicates = saved["suggested"]["entity_kinds"], saved["suggested"]["predicates"]
    settle, clause = dynamic_schema._settled, distill._clause_around
    variants = {"now": (settle, clause), "no-settle": (lambda read, text: (read, False), clause),
                "old-clause": (lambda read, text: (read, False), _old_clause)}
    for name in ("open", "closed"):
        replies = saved[name]["replies"]
        for variant, (settled, clause_rule) in variants.items():
            dynamic_schema._settled = settled  # type: ignore[assignment]
            distill._clause_around = clause_rule  # type: ignore[assignment]
            try:
                chat = FakeChat([call["reply"] if "reply" in call else ChatError(call["error"]) for call in replies])
                report = await extract_dynamic_schema(
                    await stored(saved["chunks"]), SPACE, chat, entity_kinds=kinds, predicates=predicates,
                    allow_new_types=saved[name]["record"]["allow_new_types"], max_calls=len(saved["chunks"]),
                    model_name=saved["model"])
            finally:
                dynamic_schema._settled, distill._clause_around = settle, clause
            if [user for _, user in chat.calls] != [call["user"] for call in replies]:
                raise SystemExit(f"{name} {variant}: the replay asked in another order than the measurement")
            print(f"{name} {variant}: {report.text()}")
            record = report.record()
            print("  " + json.dumps({key: record[key] for key in (
                "triples_read", "quotes_settled", "unquoted", "quoted_share", "restated", "proposed_outside_schema")}
                | {"proposed": len(report.proposed), "rejected_reasons": record["rejected_reasons"],
                   "new_predicates": {t.term: t.uses for t in report.new_predicates},
                   "new_kinds": {t.term: t.uses for t in report.new_kinds}}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--out", required=True)
    go.add_argument("--chunks", type=int, default=20)
    go.add_argument("--seed", type=int, default=42)
    go.add_argument("--min-chars", type=int, default=200)
    go.add_argument("--url", default="http://127.0.0.1:11434/v1")
    go.add_argument("--model", default="llama3.2-ctx8k")
    go.add_argument("--timeout", type=float, default=300.0)
    go.add_argument("--vocabulary", choices=("scone", "llamaindex"), default="scone")
    go.add_argument("--chunks-from", help="a saved run whose chunks to use instead of sampling")
    again = sub.add_parser("replay")
    again.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(run(args) if args.command == "run" else replay(args))


if __name__ == "__main__":
    main()
