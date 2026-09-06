"""``scone-memory``: the engine from any shell.

Stores come from the environment (see ``scone_memory.config``); with
nothing set, the CLI persists to SQLite at ~/.scone-memory/memory.db so
two invocations see the same memory. Every command takes ``--json`` for
machine-readable output, so it pipes into jq, into another process, or
into a data pipeline step.

    echo "Moved to Lisbon in March" | scone-memory remember --created-at 2024-03-02 --tag life
    scone-memory recall "where do I live" --json
    scone-memory assert mark lives_in Lisbon --valid-from 2024-03-02
    scone-memory export > memory.jsonl
    SCONE_DOCUMENTS=mongo SCONE_MONGO_URL=... scone-memory import < memory.jsonl
"""

from __future__ import annotations

import argparse
import pathlib
import asyncio
import json
import os
import sys
from typing import Mapping, Optional, Sequence

from .config import Settings, build_engine
from .engine import MemoryEngine, Record
from .errors import SconeError

CLI_DEFAULTS = {"SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite"}


def settings_for_cli(env: Mapping[str, str]) -> Settings:
    merged = {**CLI_DEFAULTS, **env}
    return Settings.from_env(merged)


def parse_pairs(pairs: Sequence[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"{flag} takes key=value, got {pair!r}")
        out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scone-memory", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="default", help="memory space (default: default)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    # The same two options are accepted after the subcommand as well;
    # SUPPRESS keeps an absent one from overwriting the top-level value.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--space", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))

    p = sub.add_parser("remember", help="store text from a file or stdin")
    p.add_argument("file", nargs="?", default="-", help="path, or - for stdin (default)")
    p.add_argument("--kind", default="note")
    p.add_argument("--source")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--created-at", help="when it happened, RFC 3339 or YYYY-MM-DD")
    p.add_argument("--meta", action="append", default=[], help="key=value scope, repeatable")
    p.add_argument("--jsonl", action="store_true", help="input is one JSON record per line, ingested as a batch")

    p = sub.add_parser("recall", help="hybrid recall plus the facts that hold")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--as-of")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--where", action="append", default=[], help="key=value scope filter, repeatable")

    p = sub.add_parser("forget", help="delete an episode")
    p.add_argument("episode_id", type=int)

    p = sub.add_parser("facts", help="list facts")
    p.add_argument("--all", action="store_true", help="include closed facts")
    p.add_argument("--as-of", help="facts that held at this time")
    p.add_argument("--status", choices=["active", "closed", "proposed", "declined"], help="one status only")
    p.add_argument("--excluded", action="store_true", help="include facts excluded from recall")

    p = sub.add_parser("assert", help="record that subject predicate object holds")
    p.add_argument("subject")
    p.add_argument("predicate")
    p.add_argument("object")
    p.add_argument("--valid-from")
    p.add_argument("--confidence", type=float, default=1.0)
    p.add_argument("--origin", choices=["stated", "extracted", "inferred"], default="stated")
    p.add_argument("--propose", action="store_true", help="park it for review instead of entering the ledger")

    p = sub.add_parser("close", help="close a fact with a reason: it stopped holding")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", required=True)

    sub.add_parser("review", help="list proposed facts awaiting a decision")
    p = sub.add_parser("approve", help="accept a proposed fact into the ledger")
    p.add_argument("fact_id", type=int)
    p = sub.add_parser("decline", help="reject a proposed fact with a reason")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("exclude", help="hide a fact from recall, keeping its history")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("include", help="undo exclude")
    p.add_argument("fact_id", type=int)

    sub.add_parser("status", help="counts and which stores are in use")
    sub.add_parser("tags", help="tag counts")
    sub.add_parser("profile", help="identity facts plus recent activity")
    sub.add_parser("export", help="dump the space as JSON lines to stdout")
    p = sub.add_parser("import", help="load JSON lines (an export) from a file or stdin")
    p.add_argument("file", nargs="?", default="-")
    sub.add_parser("serve", help="run the HTTP server (see SCONE_API_KEY, SCONE_HOST, SCONE_PORT)")
    p = sub.add_parser("distill", help="one consolidation pass: read pending episodes through the configured model")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("bench", help="measure retrieval on a LongMemEval-style file with the Rust harness's definitions")
    p.add_argument("dataset", help="path to longmemeval_s.json or a same-shaped file")
    p.add_argument("--k", default="5,10,15", help="comma-separated k values (default 5,10,15)")
    p.add_argument("--limit", type=int, help="recall limit (default max k)")
    p.add_argument("--first", type=int, help="only the first N items")
    p.add_argument("--stratified", type=int, help="N items per question type, in file order")
    p.add_argument("--include-abstention", action="store_true", help="count items with no evidence session in the denominator")
    p.add_argument("--out", help="write the full report (with per-item results) to this JSON file")
    p = sub.add_parser("agent-hook", help="observe an agent's hook payload from stdin and post it as an agent event",
                       add_help=False)
    p.add_argument("hook_args", nargs=argparse.REMAINDER)
    return parser


def read_source(path: str, stdin) -> str:
    if path == "-":
        return stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def fact_line(f) -> str:
    until = f" until {f.valid_until[:10]}" if f.valid_until else ""
    reason = f"  ({f.closed_reason})" if f.closed_reason else ""
    origin = "" if f.origin == "stated" else f" [{f.origin}]"
    excluded = f"  excluded: {f.excluded_reason}" if f.excluded_reason else ""
    return f"#{f.fact_id} [{f.status}]{origin} {f.subject} {f.predicate} {f.object}  since {f.valid_from[:10]}{until}{reason}{excluded}"


async def run(args: argparse.Namespace, engine: MemoryEngine, stdin, out, settings=None) -> int:
    space = args.space
    emit = lambda obj: print(json.dumps(obj, ensure_ascii=False), file=out)  # noqa: E731

    if args.command == "remember":
        raw = read_source(args.file, stdin)
        if args.jsonl:
            records = [Record.from_dict(json.loads(line)) for line in raw.splitlines() if line.strip()]
            added = await engine.remember_many(space, records)
        else:
            added = [
                await engine.remember(
                    space, raw, kind=args.kind, source=args.source, tags=args.tag,
                    created_at=args.created_at, metadata=parse_pairs(args.meta, "--meta"),
                )
            ]
        if args.json:
            for a in added:
                emit(a.model_dump())
        else:
            fresh = sum(1 for a in added if not a.deduplicated)
            dup = len(added) - fresh
            print(f"remembered {fresh} episode(s)" + (f", {dup} already known" if dup else ""), file=out)
        return 0

    if args.command == "recall":
        result = await engine.recall(
            space, args.query, limit=args.limit, as_of=args.as_of, tags=args.tag, where=parse_pairs(args.where, "--where")
        )
        if args.json:
            emit(result.model_dump() | {"context_reduction": result.context_reduction})
            return 0
        for f in result.facts:
            print(f"fact  {f.subject} {f.predicate} {f.object}  (since {f.valid_from[:10]})", file=out)
        for item in result.items:
            sim = f" sim={item.similarity:.2f}" if item.similarity is not None else ""
            print(f"{item.score:.2f}{sim}  {item.created_at[:10]}  #{item.episode_id}  {item.text.strip()[:200]}", file=out)
        for d in result.degraded:
            print(f"degraded: {d}", file=sys.stderr)
        if not result.items and not result.facts:
            print("nothing matched", file=out)
        return 0

    if args.command == "forget":
        await engine.forget(space, args.episode_id)
        emit({"forgotten": args.episode_id}) if args.json else print(f"forgot episode {args.episode_id}", file=out)
        return 0

    if args.command == "facts":
        facts = await engine.facts(space, include_closed=args.all, as_of=args.as_of, status=args.status, include_excluded=args.excluded)
        if args.json:
            for f in facts:
                emit(f.model_dump())
        else:
            for f in facts:
                print(fact_line(f), file=out)
            if not facts:
                print("no facts", file=out)
        return 0

    if args.command == "assert":
        fact = await engine.assert_fact(
            space, args.subject, args.predicate, args.object, valid_from=args.valid_from, confidence=args.confidence,
            origin=args.origin, proposed=args.propose,
        )
        emit(fact.model_dump()) if args.json else print(fact_line(fact), file=out)
        return 0

    if args.command == "review":
        facts = await engine.facts(space, status="proposed")
        if args.json:
            for f in facts:
                emit(f.model_dump())
        else:
            for f in facts:
                print(fact_line(f) + f"  confidence {f.confidence:.2f}", file=out)
            if not facts:
                print("nothing awaiting review", file=out)
        return 0

    if args.command in ("approve", "decline", "exclude", "include"):
        if args.command == "approve":
            fact = await engine.approve(space, args.fact_id)
        elif args.command == "decline":
            fact = await engine.decline(space, args.fact_id, args.reason)
        elif args.command == "exclude":
            fact = await engine.exclude(space, args.fact_id, args.reason)
        else:
            fact = await engine.include(space, args.fact_id)
        emit(fact.model_dump()) if args.json else print(fact_line(fact), file=out)
        return 0

    if args.command == "close":
        fact = await engine.close_fact(space, args.fact_id, args.reason)
        emit(fact.model_dump()) if args.json else print(fact_line(fact), file=out)
        return 0

    if args.command == "status":
        status = await engine.status(space)
        if args.json:
            emit(status.model_dump())
        else:
            for k, v in status.model_dump().items():
                print(f"{k:15} {v}", file=out)
        return 0

    if args.command == "tags":
        tags = await engine.tags(space)
        emit(tags) if args.json else [print(f"{n:6} {t}", file=out) for t, n in tags.items()]
        return 0

    if args.command == "profile":
        profile = await engine.profile(space)
        if args.json:
            emit({"static_facts": [f.model_dump() for f in profile.static_facts], "dynamic": profile.dynamic})
        else:
            for f in profile.static_facts:
                print(fact_line(f), file=out)
            for line in profile.dynamic:
                print(f"- {line}", file=out)
        return 0

    if args.command == "distill":
        from .config import build_worker

        worker = build_worker(engine, settings, [space])
        if worker is None:
            print("error: no consolidation model configured (SCONE_CHAT_URL and SCONE_CHAT_MODEL)", file=sys.stderr)
            return 2
        worker.batch = args.limit
        report = await worker.run_once(space)
        if args.json:
            emit({"space": space, **report.as_payload()})
        else:
            print(f"read {report.episodes} episode(s): {report.proposed} proposed, {report.accepted} accepted, "
                  f"{report.closed} closed, {report.skipped} restated, {report.parked} parked"
                  + (f"; error: {report.error}" if report.error else ""), file=out)
        return 0 if report.error is None else 1

    if args.command == "bench":
        from collections import defaultdict

        from .bench import load_items, run as run_bench
        from .config import build_embedder, build_documents, build_events, build_vectors

        items = load_items(args.dataset)
        if args.stratified:
            by_type: dict = defaultdict(list)
            for it in items:
                if len(by_type[it.question_type]) < args.stratified:
                    by_type[it.question_type].append(it)
            items = [it for group in by_type.values() for it in group]
        if args.first:
            items = items[: args.first]
        ks = tuple(int(k) for k in args.k.split(",") if k.strip())
        embedder = build_embedder(settings)  # one embedder (model load) shared across items

        async def make():
            # Fresh stores per item so nothing leaks between questions; the
            # bench always uses in-process stores, so the number measures the
            # engine, not a database.
            from .backends import InMemoryDocumentStore, InMemoryVectorIndex

            return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()

        def progress(n, total):
            if not args.json:
                print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True)

        report = await run_bench(make, items, ks=ks, limit=args.limit, include_abstention=args.include_abstention,
                                 dataset=str(args.dataset), progress=progress)
        if not args.json:
            print("", file=sys.stderr)
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(report.as_dict(with_items=True), indent=1), encoding="utf-8")
        if args.json:
            emit(report.as_dict(with_items=False))
        else:
            print(f"{report.scored} scored of {report.items} items ({'with' if report.include_abstention else 'without'} abstention), "
                  f"embedder {report.embedder}, {report.errors} error(s)", file=out)
            for k in report.ks:
                print(f"  R@{k:<3} any {report.recall_any[k] * 100:5.1f}%   all {report.recall_all[k] * 100:5.1f}%", file=out)
            print(f"  context reduction median {(report.context_reduction_median or 0) * 100:.1f}% (bytes, not tokens); "
                  f"recall p50 {report.recall_ms_p50:.1f} ms, p95 {report.recall_ms_p95:.1f} ms", file=out)
            for qt, row in report.by_type.items():
                print(f"  {qt:<28} n={row['n']:<4}" + "  ".join(f"all@{k} {row[f'all@{k}'] * 100:5.1f}%" for k in report.ks), file=out)
        return 0

    if args.command == "export":
        async for record in engine.export(space):
            emit(record)
        return 0

    if args.command == "import":
        raw = read_source(args.file, stdin)
        summary = await engine.import_records(space, [json.loads(line) for line in raw.splitlines() if line.strip()])
        emit(summary.__dict__) if args.json else print(
            f"imported {summary.episodes} episode(s), {summary.facts} fact(s); already known: "
            f"{summary.deduplicated} episode(s), {summary.facts_skipped} fact(s)", file=out
        )
        return 0

    raise SystemExit(f"unknown command {args.command}")


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None, stdin=None, out=None) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    if args.command == "agent-hook":
        from .agent_hook import run_hook

        return run_hook(args.hook_args, (stdin or sys.stdin).read(), env, stdout=out or sys.stdout)
    settings = settings_for_cli(env)
    if args.command == "serve":
        from .api.__main__ import main as serve

        serve(settings)  # same SQLite default as the other commands
        return 0

    async def go() -> int:
        engine = await build_engine(settings)

        try:
            return await run(args, engine, stdin or sys.stdin, out or sys.stdout, settings)
        finally:
            for store in (engine.documents, engine.vectors):
                if hasattr(store, "close"):
                    await store.close()

    try:
        return asyncio.run(go())
    except SconeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
