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
    p.add_argument("--history", action="store_true", help="also show the closed facts that preceded the matched ones")
    p.add_argument("--kind", help="only episodes of this kind (note, file, conversation, ...)")
    p.add_argument("--source-prefix", help="only episodes whose source starts with this text (literal)")
    p.add_argument("--since", help="only episodes that happened at or after this instant")
    p.add_argument("--until", help="only episodes that happened at or before this instant")

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

    p = sub.add_parser("audit-grounding",
                       help="re-check extracted facts against the text they came from")
    p.add_argument("--status", action="append", default=None,
                   help="which statuses to audit (repeatable, default active)")
    p.add_argument("--flagged-only", action="store_true", help="only claims their source cannot support")

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
    p.add_argument("--sample", type=int, help="the Rust harness's proportional stratified sample of N items (E21 is --sample 60)")
    p.add_argument("--seed", type=int, default=42, help="seed for --sample (default 42, the harness's default)")
    p.add_argument("--include-abstention", action="store_true", help="count items with no evidence session in the denominator")
    p.add_argument("--history", action="store_true", help="ask every recall for the closed chain behind matched facts (experiment 3)")
    p.add_argument("--cross-queries", action="store_true",
                   help="also ask each item's store another item's question whose evidence is absent: no-evidence queries for the abstention sweep (experiment 9)")
    p.add_argument("--out", help="write the full report (with per-item results) to this JSON file")
    p = sub.add_parser("bench-conflicts",
                       help="MemoryAgentBench Conflict Resolution (FactConsolidation): retrieval of the latest fact, and accuracy with a reader")
    p.add_argument("dataset", help="the Conflict_Resolution parquet (needs pyarrow) or a JSON export of its rows")
    p.add_argument("--k", type=int, default=10, help="recall limit and the k of the retrieval numbers (default 10)")
    p.add_argument("--sources", help="comma-separated item sources to run, e.g. factconsolidation_sh_6k (default all)")
    p.add_argument("--questions", type=int, help="only the first N questions of each item")
    p.add_argument("--reader", action="store_true",
                   help="answer with the configured chat model (SCONE_CHAT_URL, SCONE_CHAT_MODEL); without it only retrieval is measured")
    p.add_argument("--out", help="write every item's report with per-question results to this JSON file")
    # The hook's own flags are parsed by agent_hook; this subparser accepts
    # anything after its name and hands it over untouched. parse_known_args
    # is used at the call site because REMAINDER does not capture a flag
    # that appears first (argparse reports it as unknown and exits 2).
    p = sub.add_parser("agent-hook", help="observe an agent's hook payload from stdin and post it as an agent event",
                       add_help=False)
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


async def bench_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """The bench never opens the configured store: it measures the engine on
    fresh in-process stores per item, so running it must not touch, and
    must not migrate, whatever SCONE_SQLITE_PATH points at."""
    emit = lambda obj: print(json.dumps(obj, ensure_ascii=False), file=out)  # noqa: E731
    from collections import defaultdict

    from .bench import load_items, run as run_bench, stratified_sample
    from .config import build_embedder, build_in_process_engine

    items = load_items(args.dataset)
    if args.stratified:
        by_type: dict = defaultdict(list)
        for it in items:
            if len(by_type[it.question_type]) < args.stratified:
                by_type[it.question_type].append(it)
        items = [it for group in by_type.values() for it in group]
    if args.sample:
        items = stratified_sample(items, args.sample, args.seed)
    if args.first:
        items = items[: args.first]
    ks = tuple(int(k) for k in args.k.split(",") if k.strip())
    embedder = build_embedder(settings)  # one embedder (model load) shared across items

    async def make():
        # Fresh stores per item so nothing leaks between questions; the
        # bench always uses in-process stores, so the number measures the
        # engine, not a database.
        return await build_in_process_engine(settings, embedder)

    def progress(n, total):
        if not args.json:
            print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True)

    report = await run_bench(make, items, ks=ks, limit=args.limit, include_abstention=args.include_abstention,
                             dataset=str(args.dataset), progress=progress, history=args.history,
                             cross_queries=args.cross_queries)
    if not args.json:
        print("", file=sys.stderr)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report.as_dict(with_items=True), indent=1), encoding="utf-8")
    if args.json:
        emit(report.as_dict(with_items=False))
    else:
        print(f"{report.scored} scored of {report.items} items ({'with' if report.include_abstention else 'without'} abstention), "
              f"embedder {report.embedder}, contextual embeddings {'on' if report.contextual_embeddings else 'off'}, "
              f"{report.errors} error(s)", file=out)
        for k in report.ks:
            print(f"  R@{k:<3} any {report.recall_any[k] * 100:5.1f}%   all {report.recall_all[k] * 100:5.1f}%", file=out)
        print(f"  context reduction median {(report.context_reduction_median or 0) * 100:.1f}% (bytes, not tokens); "
              f"recall p50 {report.recall_ms_p50:.1f} ms, p95 {report.recall_ms_p95:.1f} ms", file=out)
        for qt, row in report.by_type.items():
            print(f"  {qt:<28} n={row['n']:<4}" + "  ".join(f"all@{k} {row[f'all@{k}'] * 100:5.1f}%" for k in report.ks), file=out)
        if report.history:
            print(f"  history asked on every recall: {report.items_with_facts} item(s) had facts, {report.items_with_history} had a chain"
                  + ("" if report.items_with_facts else " (no facts in any item's space: nothing was distilled, so history had nothing to show)"), file=out)
        if report.similarity_floor is not None:
            print(f"  similarity floor {report.similarity_floor}: " + ", ".join(f"{k} {v}" for k, v in sorted(report.low_confidence_counts.items())), file=out)
        sweep = report.abstention
        if sweep is None:
            print("  abstention sweep: not measurable (no item without evidence ran)", file=out)
        else:
            print(f"  abstention sweep over {sweep['no_evidence_n']} no-evidence ({sweep['cross_item_n']} cross-item) "
                  f"and {sweep['evidence_n']} evidence item(s): floor -> abstained / wrongly withheld", file=out)
            for f in sweep["floors"]:
                withheld = sweep["false_abstain_rate"][f]
                print(f"    {f:.2f} -> {sweep['abstain_rate'][f] * 100:5.1f}% / " + ("   n/a" if withheld is None else f"{withheld * 100:5.1f}%"), file=out)
    return 0


async def conflicts_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Like bench: in-process stores per item, the configured store untouched."""
    from .bench.memoryagentbench import load_conflict_resolution, run_conflict_resolution
    from .config import build_chat, build_embedder, build_in_process_engine

    items = load_conflict_resolution(args.dataset)
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        items = [it for it in items if it.source in wanted]
    embedder = build_embedder(settings)
    reader = build_chat(settings) if args.reader else None
    if args.reader and reader is None:
        print("error: --reader needs SCONE_CHAT_URL and SCONE_CHAT_MODEL", file=sys.stderr)
        return 2
    reader_name = f"{settings.chat_model} at {settings.chat_url}" if reader is not None else None

    async def make():
        return await build_in_process_engine(settings, embedder)

    def progress(n, total):
        if not args.json:
            print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True)

    reports = []
    for item in items:
        if not args.json:
            print(f"{item.source}: {len(item.facts)} facts, {len(item.questions)} questions", file=sys.stderr)
        report = await run_conflict_resolution(make, item, reader=reader, reader_name=reader_name, k=args.k,
                                               questions=args.questions, progress=progress)
        if not args.json:
            print("", file=sys.stderr)
        reports.append(report)
        if args.json:
            print(json.dumps(report.as_dict(with_items=False), ensure_ascii=False), file=out)
        else:
            acc = "no reader" if report.accuracy is None else f"accuracy {report.accuracy * 100:.1f}% ({report.correct}/{report.answered}, {report.reader})"
            print(f"{report.source:<28} {report.hops:<6} n={report.questions:<4} gold@{report.k} {report.gold_at_k * 100:5.1f}%  "
                  f"stale above gold {report.stale_above_gold * 100:5.1f}%  {acc}  {report.errors} error(s)", file=out)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps([r.as_dict(with_items=True) for r in reports], indent=1), encoding="utf-8")
    return 0


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
            space, args.query, limit=args.limit, as_of=args.as_of, tags=args.tag, where=parse_pairs(args.where, "--where"),
            history=args.history, kind=args.kind, source_prefix=args.source_prefix, since=args.since, until=args.until,
        )
        if args.json:
            emit(result.model_dump() | {"context_reduction": result.context_reduction})
            return 0
        for f in result.facts:
            print(f"fact  {f.subject} {f.predicate} {f.object}  (since {f.valid_from[:10]})", file=out)
        for f in result.history:
            until = f.valid_until[:10] if f.valid_until else "?"
            print(f"was   {f.subject} {f.predicate} {f.object}  ({f.valid_from[:10]} to {until}; {f.closed_reason})", file=out)
        for item in result.items:
            sim = f" sim={item.similarity:.2f}" if item.similarity is not None else ""
            print(f"{item.score:.2f}{sim}  {item.created_at[:10]}  #{item.episode_id}  {item.text.strip()[:200]}", file=out)
        for d in result.degraded:
            print(f"degraded: {d}", file=sys.stderr)
        if result.low_confidence:
            top = "nothing found" if result.top_similarity is None else f"top similarity {result.top_similarity:.2f}"
            print(f"low confidence: {top}, floor {engine.similarity_floor:.2f}; the evidence above is weak", file=out)
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

    if args.command == "audit-grounding":
        from dataclasses import asdict

        from .audit import audit_grounding

        found = await audit_grounding(engine, space, statuses=tuple(args.status or ("active",)))
        shown = [f for f in found if f.flagged] if args.flagged_only else found
        if args.json:
            for finding in shown:
                emit(asdict(finding))
            return 0
        for finding in shown:
            mark = "!" if finding.flagged else " "
            print(f"{mark} fact {finding.fact_id}  {finding.subject} {finding.predicate} {finding.object}"
                  f"  {finding.verdict}", file=out)
            if finding.evidence:
                print(f"    source says: {finding.evidence.strip()}", file=out)
        flagged = sum(1 for f in found if f.flagged)
        counted = "claim needs" if flagged == 1 else "claims need"
        print(f"{flagged} of {len(found)} {counted} a person" if flagged
              else f"nothing flagged in {len(found)} extracted claims", file=out)
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
        import getpass

        actor = f"cli:{getpass.getuser()}"
        if args.command == "approve":
            fact = await engine.approve(space, args.fact_id, actor=actor)
        elif args.command == "decline":
            fact = await engine.decline(space, args.fact_id, args.reason, actor=actor)
        elif args.command == "exclude":
            fact = await engine.exclude(space, args.fact_id, args.reason, actor=actor)
        else:
            fact = await engine.include(space, args.fact_id, actor=actor)
        emit(fact.model_dump()) if args.json else print(fact_line(fact), file=out)
        return 0

    if args.command == "close":
        import getpass

        fact = await engine.close_fact(space, args.fact_id, args.reason, actor=f"cli:{getpass.getuser()}")
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
    raw = list(sys.argv[1:] if argv is None else argv)
    args, rest = build_parser().parse_known_args(raw)
    env = os.environ if env is None else env
    if args.command == "agent-hook":
        from .agent_hook import run_hook

        return run_hook(rest, (stdin or sys.stdin).read(), env, stdout=out or sys.stdout)
    if rest:
        build_parser().error(f"unrecognized arguments: {' '.join(rest)}")
    settings = settings_for_cli(env)
    if args.command == "serve":
        from .api.__main__ import main as serve

        serve(settings)  # same SQLite default as the other commands
        return 0
    if args.command in ("bench", "bench-conflicts"):
        command = bench_command if args.command == "bench" else conflicts_command
        try:
            return asyncio.run(command(args, settings, out or sys.stdout))
        except SconeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

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
