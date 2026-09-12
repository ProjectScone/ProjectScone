"""``scone-memory``: the engine from any shell.

Stores come from the environment (see ``scone_memory.runtime.config``); with
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

from dataclasses import asdict

import argparse
import hashlib
import pathlib
import posixpath
import asyncio
import json
import os
import sys
import stat
from typing import Mapping, Optional, Sequence

from .config import Settings, build_engine
from ..memory.engine import MemoryEngine, Record
from ..core.errors import InvalidInput, SconeError
from ..retrieval.filters import read_conditions
from ..ingestion.chunker import DEFAULT_TARGET as DEFAULT_CHUNK_TARGET

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
    class CommandParser(argparse.ArgumentParser):
        def __init__(self, **kwargs):
            super().__init__(parents=[common], **kwargs)

    sub = parser.add_subparsers(dest="command", required=True, parser_class=CommandParser)

    p = sub.add_parser("remember", help="store text from a file or stdin")
    p.add_argument("file", nargs="?", default="-", help="path, or - for stdin (default)")
    p.add_argument("--kind", default="note")
    p.add_argument("--source")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--created-at", help="when it happened, RFC 3339 or YYYY-MM-DD")
    p.add_argument("--meta", action="append", default=[], help="key=value scope, repeatable")
    p.add_argument("--key", dest="dedup_key", help="identity across writes: the same key again is a duplicate, not a second record")
    p.add_argument("--replace", action="store_true", help="with --key: changed content replaces the record the key names")
    p.add_argument("--jsonl", action="store_true", help="input is one JSON record per line, ingested as a batch")
    p.add_argument("--image", help="explicit original PNG/JPEG/GIF/WebP file, up to 25 MB; not with --jsonl")

    p = sub.add_parser("when", help="a question about dates answered by computation, with its working")
    p.add_argument("question")
    p.add_argument("--now", help="the moment to answer from (RFC 3339); defaults to now")
    p.add_argument("--limit", type=int, default=None, help="passages read for each event named")
    p.add_argument("--max-bytes", type=int, default=None, help="byte budget for the answer text")

    p = sub.add_parser("recall", help="hybrid recall plus the facts that hold")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--as-of")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--where", action="append", default=[], help="key=value scope filter, repeatable")
    p.add_argument("--conditions", help='metadata filter as JSON, e.g. {"field": "status", "is": "published"}')
    p.add_argument("--history", action="store_true", help="also show the closed facts that preceded the matched ones")
    p.add_argument("--kind", help="only episodes of this kind (note, file, conversation, ...)")
    p.add_argument("--source-prefix", help="only episodes whose source starts with this text (literal)")
    p.add_argument("--since", help="only episodes that happened at or after this instant")
    p.add_argument("--until", help="only episodes that happened at or before this instant")
    p.add_argument("--candidate-limit", type=int,
                   help="how many candidates each lane fetches before fusion (1 to 1000)")
    p.add_argument("--no-rerank", action="store_true", help="skip the configured reranker for this search")
    p.add_argument("--graph-boost", action="store_true",
                   help="add the entity lane: passages naming what the question is about, or one relation away")
    p.add_argument("--merge", action="store_true",
                   help="join neighbouring chunks of one episode into the passage holding them, "
                        "and say which chunks went into each")
    p.add_argument("--window", type=int, metavar="BYTES",
                   help="return each passage with this many bytes of its episode either side; "
                        "serves the single precise hit that --merge cannot")
    p.add_argument("--withhold", metavar="KINDS",
                   help="withhold matches of these kinds from the answer, comma separated "
                        "(email,phone,ip,card,secret); a net of patterns, never a guarantee")
    p.add_argument("--code-context", action="store_true",
                   help="for a passage of code, also quote the signature it sits inside and the "
                        "imports of its file; the passage itself is not changed")
    p.add_argument("--parts", action="store_true",
                   help="search each part of a multi-part question and give every part a turn "
                        "(measured to change nothing on LongMemEval; off by default)")

    p = sub.add_parser("attachments", help="list an episode's original attachment metadata (no download)")
    p.add_argument("episode_id", type=int)

    p = sub.add_parser("source-key", help="read the current source stored with remember --key")
    p.add_argument("dedup_key")

    p = sub.add_parser("sync-directory", help="reconcile local documents through an encrypted source journal")
    p.add_argument("root")
    p.add_argument("--journal", required=True, help="private journal path outside the source root")
    p.add_argument("--key-file", required=True, help="private file containing a 32-byte journal key")
    p.add_argument("--store-id", required=True, help="stable identity for this memory catalog")
    p.add_argument("--parser-revision", required=True, help="change when parser configuration changes")
    p.add_argument("--delete-missing", action="store_true", help="retire managed missing sources after a complete stable scan")
    p.add_argument("--max-files", type=int, default=1000)
    p.add_argument("--max-total-bytes", type=int, default=256_000_000)
    p.add_argument("--extension", action="append", help="restrict to a dotted suffix; repeat for several")

    p = sub.add_parser("jobs", help="recent ingest batches and how far each has got")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("job", help="one ingest batch: what each record became")
    p.add_argument("job_id")
    p = sub.add_parser("cancel-job", help="stop expecting more of a batch; stored records stay stored")
    p.add_argument("job_id")
    p = sub.add_parser("merge-space", help="move everything this space holds into another; --dry-run previews it")
    p.add_argument("--into", required=True, help="the space to move it into")
    p.add_argument("--confirm", help="repeat the space being merged; a whole space does not move by accident")
    p.add_argument("--dry-run", action="store_true", help="say what would move and move nothing")

    p = sub.add_parser("delete-space", help="delete everything the space holds; --dry-run previews the receipt")
    p.add_argument("--confirm", metavar="SPACE", help="repeat the space name to do it")
    p.add_argument("--dry-run", action="store_true", help="show what would go, and remove nothing")
    p = sub.add_parser("forget", help="delete an episode; the receipt says what went and what stayed")
    p.add_argument("episode_id", type=int)
    p.add_argument("--dry-run", action="store_true", help="show the impact and remove nothing")

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
    p.add_argument("--extends", type=int, metavar="FACT_ID", help="a fact this one adds detail to; both stay as they are")
    p.add_argument("--derived-from", type=int, action="append", default=[], metavar="FACT_ID",
                   help="a ledger fact this one was inferred from (repeatable); the claim is stored as inferred")
    p = sub.add_parser("link", help="relate one fact to another: FROM extends | derived_from | contradicts | supports TO")
    p.add_argument("from_fact", type=int)
    p.add_argument("to_fact", type=int)
    p.add_argument("kind", choices=["extends", "derived_from", "contradicts", "supports"])
    p.add_argument("--source", type=int, metavar="EPISODE_ID", help="the episode the relation rests on")
    p.add_argument("--quote", help="an exact substring of that episode supporting the relation")
    p = sub.add_parser("links", help="show the relations a fact takes part in, from either end")
    p.add_argument("fact_id", type=int)
    sub.add_parser("doctor", help="what references what across the stores, read only: orphans by id, nothing repaired")
    p = sub.add_parser("vectors", help="which embedder wrote the stored vectors, and whether recall can compare them")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--reembed", action="store_true",
                        help="re-embed every stored chunk with this embedder and record it as the writer")
    action.add_argument("--adopt", action="store_true",
                        help="vouch that vectors stored before writers were recorded came from this embedder")
    p = sub.add_parser("expire", help="forget episodes older than a retention policy (oldest first, bounded); facts never expire")
    p.add_argument("--keep", action="append", default=[], metavar="KIND=DAYS", required=True,
                   help="keep this kind for this many days by the episode's own time (repeatable)")
    p.add_argument("--limit", type=int, default=100, help="at most this many in one pass")
    p.add_argument("--dry-run", action="store_true", help="report what would go and forget nothing")

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
    p = sub.add_parser("reconsider", help="undo decline: put a declined fact back for review")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("reopen", help="undo a close made by hand: the fact holds again")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", required=True)

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
    p.add_argument("--resurrect", action="store_true", help="store content this space forgot on purpose; the tombstone stays")
    sub.add_parser("serve", help="run the HTTP server (see SCONE_API_KEY, SCONE_HOST, SCONE_PORT; "
                                "SCONE_CONVERSATIONS_JOURNAL composes the conversation service on the same origin; "
                                "SCONE_CONVERSATIONS_PERSONAS + SCONE_CONVERSATIONS_REGISTRY add a persona catalog)")
    p = sub.add_parser("serve-conversations", help="run the optional authenticated conversation service")
    p.add_argument("--journal", required=True, help="separate conversation SQLite database; parent must exist")
    runtime = p.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--model-factory", help="trusted module:callable returning a fresh native TextModel adapter")
    runtime.add_argument("--history-only", action="store_true", help="inspect saved conversations without a model")
    p = sub.add_parser("distill", help="one consolidation pass: read pending episodes through the configured model")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("derive", help="one derivation pass: propose claims that follow from the claims held, with their premises")
    p.add_argument("--limit", type=int, default=50, help="groups sent to the model in this pass")
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
    p.add_argument("--merge", action="store_true",
                   help="join neighbouring chunks of one episode into the passage holding them "
                        "before scoring, to measure what that changes")
    p.add_argument("--window", type=int, default=0, metavar="BYTES",
                   help="widen every returned passage by this many bytes either side before "
                        "scoring, to measure what that changes")
    p.add_argument("--cross-queries", action="store_true",
                   help="also ask each item's store another item's question whose evidence is absent: no-evidence queries for the abstention sweep (experiment 9)")
    p.add_argument("--out", help="write the full report (with per-item results) to this JSON file")
    p = sub.add_parser("graph", help="the entity graph: report, path, context, entity, timeline, walk, schema, match, "
                                     "overview, changes, duplicates, export")
    graph = p.add_subparsers(dest="graph_command", required=True)
    g = graph.add_parser("report", help="communities, central entities, surprising links and questions")
    g.add_argument("--markdown", action="store_true", help="print Markdown instead of JSON")
    g.add_argument("--resolution", type=float, default=1.0, help="how fine the communities are (default 1)")
    g.add_argument("--usage", action="store_true", help="also say what recent recalls returned, and what they never reach")
    g = graph.add_parser("path", help="how two entities connect, each hop with its facts")
    g.add_argument("source")
    g.add_argument("target")
    g.add_argument("--max-hops", type=int, default=3)
    g = graph.add_parser("context", help="what the graph records around names or a question, for a model")
    g.add_argument("names", nargs="*")
    g.add_argument("--question")
    g.add_argument("--max-bytes", type=int, default=8000)
    g.add_argument("--similar", action="store_true", help="also centre on entities the question resembles")
    g.add_argument("--min-similarity", type=float, help="keep out resembling entities below this cosine similarity")
    g = graph.add_parser("entity", help="one entity: its relations both ways and its values")
    g.add_argument("name")
    graph.add_parser("meanings", help="what this process takes the space's predicates to mean to each other")
    g = graph.add_parser("timeline", help="one entity's facts in valid time")
    g.add_argument("name")
    g.add_argument("--as-of")
    g = graph.add_parser("walk", help="entities reached from names hop by hop; --direction in finds what depends on them")
    g.add_argument("names", nargs="+")
    g.add_argument("--direction", default="both", choices=["both", "out", "in"])
    g.add_argument("--hops", type=int, help="most steps from the names (1 to 8; default: as far as it reaches)")
    g.add_argument("--limit", type=int, default=150, help="most entities shown (1 to 1000)")
    g = graph.add_parser("schema", help="the kinds of entity and the predicates the graph holds")
    g.add_argument("--limit", type=int, default=200, help="predicates to list, most used first (default 200)")
    g.add_argument("--max-bytes", type=int, default=64_000, help="byte budget for the listed predicates")
    g = graph.add_parser("match", help="a structured question: triple patterns joined by ?variables (quote them)")
    g.add_argument("--pattern", nargs=3, action="append", required=True, metavar=("SUBJECT", "PREDICATE", "OBJECT"),
                   help="one pattern; repeat for more, up to 6. A term starting with ? is a variable")
    g.add_argument("--returns", action="append", help="a variable to answer with; repeat for more (default: every one)")
    g.add_argument("--limit", type=int, default=20, help="rows to answer (1 to 100)")
    g.add_argument("--status", default="current", choices=["current", "history", "proposed", "all"])
    g.add_argument("--as-of")
    g.add_argument("--apart", action="store_true", help="join facts across time, not only those that held at one moment")
    g.add_argument("--follows", action="store_true",
                   help="also match what follows from the claims under the space's vocabulary, marked in each row")
    g.add_argument("--max-bytes", type=int, default=8000, help="byte budget for the answer (512 to 64000)")
    g = graph.add_parser("overview", help="each community digested with cited facts, for a question about the whole")
    g.add_argument("--question")
    g.add_argument("--limit", type=int, default=12, help="communities to digest (1 to 50)")
    g.add_argument("--facts", type=int, default=3, help="facts cited for each (0 to 10)")
    g.add_argument("--resolution", type=float, default=1.0, help="how fine the communities are (above 0, at most 10)")
    g.add_argument("--max-bytes", type=int, default=8000, help="byte budget for the answer (512 to 64000)")
    g = graph.add_parser("changes", help="what changed in the graph between two moments, cited")
    g.add_argument("--since", required=True, help="RFC 3339 moment to compare from")
    g.add_argument("--until", help="RFC 3339 moment to compare to (default: now)")
    g.add_argument("--limit", type=int, default=50, help="changes to list (1 to 500)")
    g.add_argument("--max-bytes", type=int, default=8000, help="byte budget for the answer (512 to 64000)")
    g = graph.add_parser("health", help="what in the graph wants attention, counted with examples")
    g.add_argument("--limit", type=int, default=None, help="examples shown for each concern")
    g.add_argument("--max-bytes", type=int, default=None)

    g = graph.add_parser("duplicates", help="entities that may be one thing under two names, and why (nothing merged)")
    g.add_argument("--limit", type=int, default=50, help="pairs to suggest (1 to 500)")
    g.add_argument("--min-score", type=float, default=0.5, help="suggest only pairs at least this likely (0 to 1)")
    g.add_argument("--max-bytes", type=int, default=8000, help="byte budget for the answer (512 to 64000)")
    g = graph.add_parser("export", help="the whole graph as a file another tool reads")
    g.add_argument("--format", default="json", choices=["json", "graphml", "gexf", "cypher", "csv", "jsonld", "obsidian", "wiki",
                                                               "mermaid", "svg", "canvas", "html"])
    g.add_argument("--out", help="write here instead of standard output (needed for the zip formats)")
    p = sub.add_parser("calibrate",
                       help="measure the floor this engine abstains by, on questions with and without an answer")
    p.add_argument("dataset", help="a LongMemEval-shaped JSON file of questions")
    p.add_argument("--sample", type=int, help="measure on a stratified sample of N questions")
    p.add_argument("--target-false-abstain", type=float, default=None,
                   help="the share of answerable questions the floor may withhold (default 0.05)")
    p.add_argument("--out", help="write the policy here; without it, nothing is written")

    p = sub.add_parser("bench-temporal",
                       help="score computed temporal answers on a file of dated questions (no model called)")
    p.add_argument("dataset", help="a LongMemEval-shaped JSON file, e.g. bench-data/temporal-40.json")
    p.add_argument("--limit", type=int, help="only the first N questions")

    p = sub.add_parser("bench-parts",
                       help="measure what splitting multi-part questions changes, paired (no model called)")
    p.add_argument("dataset", help="a LongMemEval-shaped JSON file")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--k", type=int, default=10, help="passages compared per question")

    p = sub.add_parser("bench-route",
                       help="score the rule that chooses a route, on a file of questions (no model called)")
    p.add_argument("dataset", help="a LongMemEval-shaped JSON file, e.g. bench-data/temporal-40.json")
    p.add_argument("--limit", type=int, help="only the first N questions")

    p = sub.add_parser("answer", help="answer a question with whichever machinery suits it, and say which")
    p.add_argument("question")
    p.add_argument("--route", choices=("temporal", "graph", "recall"),
                   help="insist on one route instead of letting the rule choose")
    p.add_argument("--limit", type=int, default=5, help="passages an ordinary search answers with")
    p.add_argument("--now", help="the moment to answer from (RFC 3339); defaults to now")

    p = sub.add_parser("sync", help="bring a space into step with a directory: added, changed and gone")
    p.add_argument("directory")
    p.add_argument("--marker", help="the name these memories are held under; defaults to the "
                                    "directory's absolute path, so rename it with this")
    p.add_argument("--suffix", action="append", default=[],
                   help="only files with this suffix, repeatable; defaults to code and prose")
    p.add_argument("--apply", action="store_true", help="actually write; without it this is a plan")
    p.add_argument("--remove", action="store_true",
                   help="also forget memories whose file is gone from disk (destructive; needs --apply)")
    p.add_argument("--limit", type=int, default=100_000, help="files to read (1 to 100000)")
    p.add_argument("--max-bytes", type=int, default=1_000_000, help="bytes read from one file")

    p = sub.add_parser("map", help="remember every source file under a directory, and optionally what each says")
    p.add_argument("directory")
    p.add_argument("--graph", action="store_true",
                   help="also record what each file defines, imports and calls, as claims quoted from the line")
    p.add_argument("--limit", type=int, default=5_000, help="files to read at most (default 5000)")
    p.add_argument("--max-bytes", type=int, default=400_000, help="bytes of one file to read (default 400000)")

    p = sub.add_parser("fs", help="the space as a tree: ls, cat, find and write a note")
    tree = p.add_subparsers(dest="fs_command", required=True)
    t = tree.add_parser("ls", help="what is under a path")
    t.add_argument("path", nargs="?", default="/")
    t.add_argument("--limit", type=int, default=100, help="entries to show (1 to 1000)")
    t.add_argument("--offset", type=int, default=0, help="where to continue a listing")
    t = tree.add_parser("cat", help="one file of the tree")
    t.add_argument("path")
    t.add_argument("--max-bytes", type=int, default=64_000, help="bytes to show; a longer file says it was cut")
    t = tree.add_parser("find", help="paths whose content answers a query")
    t.add_argument("query")
    t.add_argument("--under", default="/", help="a directory to search under")
    t.add_argument("--limit", type=int, default=10, help="paths to answer with (1 to 200)")
    t = tree.add_parser("write", help="write a note under /notes, reading it from stdin")
    t.add_argument("path")
    t.add_argument("--writable", action="store_true",
                   help="say so before writing; without it the tree is read only")
    t.add_argument("--if-version", type=int, help="the version cat reported, to refuse a write onto a newer note")

    p = sub.add_parser("bench-code",
                       help="measure what recall does with code, on a corpus of source files (no model called)")
    p.add_argument("root", help="a directory of source files, e.g. src/scone_memory")
    p.add_argument("--k", type=int, default=5, help="recall limit and the k of the numbers (default 5)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--asked", choices=("docstring", "name"), default="docstring",
                   help="ask with the docstring as written, or with 'what <name> does' (default docstring)")
    p.add_argument("--chunk-target", type=int, default=DEFAULT_CHUNK_TARGET,
                   help=f"the chunk size the corpus is stored at (default {DEFAULT_CHUNK_TARGET})")
    p.add_argument("--by-length", action="store_true",
                   help="store the corpus with the ordinary chunker instead of cutting at declarations")

    p = sub.add_parser("tune",
                       help="measure retrieval settings against each other on a dataset, and say which to take")
    p.add_argument("dataset", help="a LongMemEval-shaped JSON file")
    p.add_argument("--sample", type=int, default=30, help="questions per setting, stratified (default 30)")
    p.add_argument("--k", type=int, default=5, help="the k the settings are scored at (default 5)")
    p.add_argument("--seed", type=int, default=42, help="the sample's seed (default 42)")
    p.add_argument("--candidates", default="", help="comma-separated candidate limits to try beside the default")
    p.add_argument("--restatement", action="store_true", help="also try with restatement demotion off")
    p.add_argument("--contextual", action="store_true", help="also try with contextual embedding prefixes on")

    p = sub.add_parser("bench-graph", help="score entity graph quality on a versioned synthetic fixture")
    p.add_argument("--fixtures", required=True, help="a JSON lines fixture, e.g. benchmarks/entity_graph/fixtures-v1.jsonl")
    p = sub.add_parser("bench-conflicts",
                       help="MemoryAgentBench Conflict Resolution (FactConsolidation): retrieval of the latest fact, and accuracy with a reader")
    p.add_argument("dataset", help="the Conflict_Resolution parquet (needs pyarrow) or a JSON export of its rows")
    p.add_argument("--k", type=int, default=10, help="recall limit and the k of the retrieval numbers (default 10)")
    p.add_argument("--sources", help="comma-separated item sources to run, e.g. factconsolidation_sh_6k (default all)")
    p.add_argument("--questions", type=int, help="only the first N questions of each item")
    p.add_argument("--reader", action="store_true",
                   help="answer with the configured chat model (SCONE_CHAT_URL, SCONE_CHAT_MODEL); without it only retrieval is measured")
    p.add_argument("--distill", action="store_true",
                   help="E36: extract claims from every statement with the configured chat model and approve them for the run")
    p.add_argument("--derive", action="store_true", help="E36: after --distill, one derivation pass; bridges stay proposed")
    p.add_argument("--derive-approve", action="store_true", help="E36: approve the derivation pass's bridges to measure the ceiling")
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


def job_payload(job) -> dict:
    return {**job.model_dump(), "searchable": job.searchable, "consolidated": job.consolidated, "state": job.state}


def job_line(job) -> str:
    return (f"{job.job_id}  {job.created_at}  {len(job.items)} record(s): "
            f"{job.searchable} searchable, {job.consolidated} read  [{job.state}]")


def space_line(receipt) -> str:
    return (f"{receipt.episodes} episodes, {receipt.chunks} chunks, {receipt.facts} claims, {receipt.links} links, "
            f"{receipt.tombstones} tombstones, {receipt.events} events; attachments released "
            f"{len(receipt.attachments_released)}, kept {len(receipt.attachments_kept)}")


def receipt_line(r) -> str:
    return (f"{r.chunks} chunk(s), {len(r.attachments_released)} attachment(s) released, {len(r.attachments_kept)} kept, "
            f"{len(r.facts_citing)} claim(s), {len(r.links_citing)} link(s) and "
            f"{len(r.affirmations_citing)} restatement(s) cite it and stand")


def link_line(link) -> str:
    where = f"  (episode {link.source_episode_id})" if link.source_episode_id is not None else ""
    return f"link #{link.link_id}: #{link.from_fact} {link.kind.replace('_', ' ')} #{link.to_fact}{where}"


def fact_line(f) -> str:
    until = f" until {f.valid_until[:10]}" if f.valid_until else ""
    reason = f"  ({f.closed_reason})" if f.closed_reason else ""
    origin = "" if f.origin == "stated" else f" [{f.origin}]"
    excluded = f"  excluded: {f.excluded_reason}" if f.excluded_reason else ""
    # Where a claim came from belongs beside it: a claim with no episode
    # rests on whoever wrote it, and that is worth seeing in a list.
    came = f"  from episode {f.source_episode_id}" if f.source_episode_id else ""
    return (f"#{f.fact_id} [{f.status}]{origin} {f.subject} {f.predicate} {f.object}  "
            f"since {f.valid_from[:10]}{until}{reason}{came}{excluded}")


def graph_bench_command(args: argparse.Namespace, out) -> int:
    """Graph quality on a fixture, in process; the configured store is never
    opened. Exits 1 when a threshold is breached, and 2 when the fixture's
    gold names something it never loads."""
    from pathlib import Path

    from ..testing.entity_graph_benchmark import THRESHOLDS_V1, FixtureError, failures, run_entity_graph_benchmark

    try:
        report = asyncio.run(run_entity_graph_benchmark(Path(args.fixtures)))
    except FixtureError as refused:
        print(f"error: fixture refused: {refused}", file=sys.stderr)
        return 2
    breached = failures(report, THRESHOLDS_V1)
    if getattr(args, "json", False):
        print(json.dumps({"report": report.record(), "failures": breached}, indent=2, sort_keys=True), file=out)
    else:
        for key, value in report.record().items():
            print(f"{key}: {value}", file=out)
        print("thresholds: " + ("pass" if not breached else "; ".join(breached)), file=out)
    return 1 if breached else 0


async def calibrate_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Measure the floor to abstain by, with the configured embedder, and
    write it down with what it cost. Each question is measured on its own
    memory, so the configured store is not read or written."""
    from ..bench.calibrate import DEFAULT_TARGET, calibrate, write_policy
    from ..bench.runner import load_items, stratified_sample
    from ..retrieval.abstention import PolicyError
    from .config import build_embedder, build_in_process_engine

    # Refused before anything is measured: a sample of none and a target
    # outside 0 to 1 are questions nobody can answer, and finding that out
    # after the work is done helps no one.
    target = DEFAULT_TARGET if args.target_false_abstain is None else args.target_false_abstain
    if not 0.0 <= target <= 1.0:
        raise InvalidInput("--target-false-abstain must be the share of answers a floor may withhold, from 0 to 1")
    if args.sample is not None and args.sample < 1:
        raise InvalidInput("--sample must be at least 1 question")
    items = load_items(args.dataset)
    if args.sample:
        items = stratified_sample(items, args.sample)
    embedder = build_embedder(settings)
    try:
        policy, report = await calibrate(lambda: build_in_process_engine(settings, embedder), items,
                                         target_false_abstain=target, dataset=str(args.dataset))
    except PolicyError as refused:
        raise InvalidInput(str(refused)) from None
    if policy is None:
        said = {"policy": None, "reason": "no floor withholds few enough answers to take",
                "sweep": report.abstention}
        print(json.dumps(said) if args.json else
              f"abstention: no floor is within {target} of answers withheld; none written", file=out)
        return 1
    if args.out:
        write_policy(policy, args.out)
    print(json.dumps(policy.record()) if args.json else
          policy.text() + (f"\nwritten: {args.out}" if args.out else "\nnot written: pass --out to keep it"),
          file=out)
    return 0


async def temporal_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Score computed temporal answers. Each question gets its own memory,
    so the configured store is not read or written."""
    from ..bench.temporal import run_temporal

    scored = await run_temporal(args.dataset, limit=args.limit)
    print(json.dumps(scored.record()) if args.json else scored.text(), file=out)
    return 0


async def parts_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Ask every multi-part question both ways on its own memory. Nothing is
    written anywhere, and the report says when the rule split nothing."""
    from ..bench.parts import run_parts_bench

    if not 1 <= args.k <= 100:
        raise InvalidInput(f"--k must be from 1 to 100, not {args.k}")
    scored = await run_parts_bench(args.dataset, limit=args.limit, k=args.k)
    print(json.dumps(scored.record()) if args.json else scored.text(), file=out)
    return 0


async def route_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Score where the routing rule sends each question. Each question gets
    its own memory, so the configured store is not read or written, and the
    report says plainly what the file cannot tell us."""
    from ..bench.route import run_route_bench

    scored = await run_route_bench(args.dataset, limit=args.limit)
    print(json.dumps(scored.record()) if args.json else scored.text(), file=out)
    return 0


async def tune_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Measure retrieval settings against each other. Its own in-process
    stores per item, so the configured store is neither read nor written,
    and nothing is written anywhere: what comes back is a recommendation
    with its measurement attached."""
    from ..bench.tune import DEFAULT_SETTINGS, MAX_SETTINGS, Setting, tune

    swept = [DEFAULT_SETTINGS]
    for entry in (item.strip() for item in args.candidates.split(",")):
        if not entry:
            continue
        if not entry.isdigit() or not 1 <= int(entry) <= 10_000:
            raise InvalidInput(f"--candidates takes whole numbers from 1 to 10000, not {entry!r}")
        swept.append(Setting(candidate_limit=int(entry)))
    if args.restatement:
        swept.append(Setting(demote_restated=False))
    if args.contextual:
        swept.append(Setting(contextual_embeddings=True))
    if len(swept) > MAX_SETTINGS:
        raise InvalidInput(f"a sweep measures at most {MAX_SETTINGS} settings")
    tuned = await tune(args.dataset, settings=swept, k=args.k, sample=args.sample, seed=args.seed,
                       base=settings)
    print(json.dumps(tuned.record()) if args.json else tuned.text(), file=out)
    return 0


def _parses(text: str) -> bool:
    """Whether Python reads it at all, which is a different question from
    whether it had anything to say."""
    import ast

    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


async def sync_command(args: argparse.Namespace, engine: MemoryEngine, out) -> int:
    """Bring a space into step with a directory. A plan by default, because
    the interesting half is destructive: a file gone from disk means a
    memory to forget, and that needs asking for twice (--apply --remove).

    Deliberately not on the HTTP surface: it reads whatever local directory
    it is pointed at, and an API caller should not choose that."""
    from ..ingestion.sync import NoEventLog, last_sync, sync_directory

    if args.remove and not args.apply:
        print("note: --remove needs --apply to forget anything; this is a plan", file=sys.stderr)
    chosen = {"suffixes": tuple(args.suffix)} if args.suffix else {}
    done = await sync_directory(engine, args.space, args.directory, marker=args.marker,
                                apply=args.apply, remove=args.remove, limit=args.limit,
                                max_bytes=args.max_bytes, **chosen)
    if args.json:
        print(json.dumps(done.record()), file=out)
        return 0
    print(done.text(), file=out)
    if not args.apply:
        try:
            before = await last_sync(engine, args.space, done.marker)
        except NoEventLog as unknown:
            print(f"last sync: unknown ({unknown})", file=out)
        else:
            print(f"last sync: {before['at']} ({before['added']} added, {before['updated']} updated)"
                  if before else "last sync: never", file=out)
    return 0


async def map_command(args: argparse.Namespace, engine: MemoryEngine, out) -> int:
    """Remember every source file under a directory, stored under the path
    it was read from. With --graph, also record what each file says about
    itself. What was read and what was not is said: a map that quietly
    skipped half a repository is worse than no map."""
    from ..ingestion.code import BRACE_SUFFIXES, PYTHON_SUFFIXES, code_language, declarations
    from ..ingestion.code_graph import record_claims

    root = pathlib.Path(args.directory)
    if not root.is_dir():
        raise InvalidInput(f"{args.directory} is not a directory to map")
    if not 1 <= args.limit <= 100_000:
        raise InvalidInput("--limit must be from 1 to 100000 files")
    if not 1 <= args.max_bytes <= 50_000_000:
        raise InvalidInput("--max-bytes must be from 1 to 50000000")
    # Judged on the part below the root, never on the whole path: the rule
    # is about a repository's own `.git`, and a root reached through a
    # dot-segment would otherwise skip its entire tree and report nothing
    # read -- exactly the quietly-skipped map this command warns about.
    found = [path for path in sorted(root.rglob("*"))
             if path.is_file() and path.suffix in (*PYTHON_SUFFIXES, *BRACE_SUFFIXES)
             and not any(part.startswith(".") or part == "__pycache__"
                         for part in path.relative_to(root).parts)]
    # Resolution belongs here, because this is what knows which files
    # exist: a relative import is followed only to a file actually read,
    # and one that leads anywhere else is left out rather than guessed at.
    seen = {str(path.relative_to(root)) for path in found[: args.limit]}

    def resolve(path: str, level: int, module: str) -> str | None:
        here = posixpath.dirname(path)
        for _ in range(level - 1):
            here = posixpath.dirname(here)
        stem = posixpath.join(here, *module.split(".")) if module else here
        # A relative import names a file however the language spells it:
        # Python by module, the brace family by path with the extension
        # left off. Only a file this walk really read is followed.
        stems = [stem, posixpath.normpath(posixpath.join(posixpath.dirname(path), module))
                 if module.startswith(".") else stem]
        for base in dict.fromkeys(stems):
            for suffix in ("py", "ts", "tsx", "js", "jsx", "go", "rs"):
                if f"{base}.{suffix}" in seen:
                    return f"{base}.{suffix}"
                if f"{base}/index.{suffix}" in seen:
                    return f"{base}/index.{suffix}"
        for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
            if candidate in seen:
                return candidate
        return None

    read, again, claims, quiet, unread, cut = 0, 0, 0, 0, 0, 0
    for path in found[: args.limit]:
        raw = path.read_bytes()
        cut += len(raw) > args.max_bytes
        text = raw[: args.max_bytes].decode("utf-8", errors="replace")
        if not text.strip():
            continue
        where = str(path.relative_to(root))
        added = await engine.remember(args.space, text, kind="file", source=where)
        if added.deduplicated:
            again += 1
            continue
        read += 1
        if args.graph:
            said = await record_claims(engine, args.space, episode_id=added.episode_id,
                                       content=text, path=where, when=engine.clock(),
                                       resolve=resolve)
            claims += said
            # A file with nothing to say and a file this cannot read are
            # different things, and a count that adds them together tells
            # a reader neither.
            if said == 0 and code_language(where) is not None:
                if declarations(text, language=code_language(where)) or _parses(text):
                    quiet += 1
                else:
                    unread += 1
    parts = [f"{read} file(s) read"]
    if again:
        parts.append(f"{again} already here")
    if args.graph:
        parts.append(f"{claims} claim(s)")
        if quiet:
            parts.append(f"{quiet} had nothing to say")
        if unread:
            parts.append(f"{unread} could not be read")
    if len(found) > args.limit:
        parts.append(f"{len(found) - args.limit} left unread of {len(found)}")
    if cut:
        parts.append(f"{cut} read only to {args.max_bytes} bytes")
    if getattr(args, "json", False):
        print(_ledger_json({"read": read, "deduplicated": again, "claims": claims, "quiet": quiet,
                            "unread": unread, "found": len(found), "limit": args.limit,
                            "cut": cut}), file=out)
    else:
        print("map: " + ", ".join(parts), file=out)
    return 0


async def filesystem_command(args: argparse.Namespace, engine: MemoryEngine, stdin, out) -> int:
    """The tree at the command line. Writing needs --writable, so that a
    mistyped path cannot write where somebody only meant to look."""
    from ..filesystem import FilesystemPolicy, MemoryFilesystem, PathConflict, PathRefused

    space = args.space
    writable = bool(getattr(args, "writable", False))
    tree = MemoryFilesystem(engine, space, FilesystemPolicy(writable=writable))
    try:
        if args.fs_command == "ls":
            listing = await tree.list(args.path, limit=args.limit, offset=args.offset)
            if getattr(args, "json", False):
                print(_ledger_json(listing.record()), file=out)
                return 0
            for entry in listing.entries:
                size = "" if entry.bytes is None else f"  {entry.bytes} bytes"
                print(f"{entry.path}{'/' if entry.kind == 'directory' else ''}{size}", file=out)
            if listing.truncated:
                print(f"({listing.total} in all; --offset {listing.next_offset} for more)", file=out)
            return 0
        if args.fs_command == "cat":
            page = await tree.read(args.path, max_bytes=args.max_bytes)
            if getattr(args, "json", False):
                print(_ledger_json(page.record()), file=out)
                return 0
            print(page.text, file=out)
            if page.truncated:
                print(f"({page.bytes} bytes shown; --max-bytes for more)", file=out)
            return 0
        if args.fs_command == "find":
            found = await tree.search(args.query, under=args.under, limit=args.limit)
            if getattr(args, "json", False):
                print(_ledger_json(found.record()), file=out)
                return 0
            for hit in found.hits:
                print(f"{hit.path}: {hit.excerpt}", file=out)
            if not found.hits:
                print("nothing found", file=out)
            return 0
        text = stdin.read()
        written = await tree.write(args.path, text, if_version=args.if_version)
        if getattr(args, "json", False):
            print(_ledger_json(written.record()), file=out)
        else:
            print(f"{written.path}: {written.bytes} bytes, version {written.version}", file=out)
        return 0
    except (PathRefused, PathConflict) as refused:
        raise InvalidInput(str(refused)) from None


async def bench_code_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """Measure code retrieval on a corpus of files. Its own in-process
    store, so the configured one is neither read nor written."""
    from ..bench.code import run_code_bench

    if args.k < 1:
        raise InvalidInput("--k must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise InvalidInput("--limit must be at least 1 question")
    if args.chunk_target < 1:
        raise InvalidInput("--chunk-target must be a positive number of characters")
    if not pathlib.Path(args.root).is_dir():
        raise InvalidInput(f"{args.root} is not a directory of source files")
    scored = await run_code_bench(args.root, k=args.k, limit=args.limit, asked=args.asked,
                                  code_aware=not args.by_length, chunk_target=args.chunk_target)
    print(json.dumps(scored.record()) if args.json else scored.text(), file=out)
    return 0


async def bench_command(args: argparse.Namespace, settings: Settings, out) -> int:
    """The bench never opens the configured store: it measures the engine on
    fresh in-process stores per item, so running it must not touch, and
    must not migrate, whatever SCONE_SQLITE_PATH points at."""
    emit = lambda obj: print(json.dumps(obj, ensure_ascii=False), file=out)  # noqa: E731
    from collections import defaultdict

    from ..bench import load_items, run as run_bench, stratified_sample
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
        # Always on stderr, --json or not. A run that takes hours with no
        # sign of life cannot be told from a wedged one, and the JSON goes
        # to stdout so this spoils nothing that reads it. Learned by
        # waiting two and a half hours without knowing whether a benchmark
        # was a tenth or nine tenths of the way through.
        print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True)

    report = await run_bench(make, items, ks=ks, limit=args.limit, include_abstention=args.include_abstention,
                             dataset=str(args.dataset), progress=progress, history=args.history,
                             cross_queries=args.cross_queries, merge=args.merge, window=args.window)
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
    from ..bench.memoryagentbench import load_conflict_resolution, run_conflict_resolution
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
    model = build_chat(settings) if args.distill else None
    if args.distill and model is None:
        print("error: --distill needs SCONE_CHAT_URL and SCONE_CHAT_MODEL", file=sys.stderr)
        return 2
    if (args.derive or args.derive_approve) and not args.distill:
        print("error: --derive needs --distill (the pass runs over extracted claims)", file=sys.stderr)
        return 2

    async def make():
        return await build_in_process_engine(settings, embedder)

    def progress(n, total):
        # Always on stderr, --json or not. A run that takes hours with no
        # sign of life cannot be told from a wedged one, and the JSON goes
        # to stdout so this spoils nothing that reads it. Learned by
        # waiting two and a half hours without knowing whether a benchmark
        # was a tenth or nine tenths of the way through.
        print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True)

    reports = []
    for item in items:
        if not args.json:
            print(f"{item.source}: {len(item.facts)} facts, {len(item.questions)} questions", file=sys.stderr)
        report = await run_conflict_resolution(make, item, reader=reader, reader_name=reader_name, k=args.k,
                                               questions=args.questions, progress=progress,
                                              model=model, model_name=f"{settings.chat_model} at {settings.chat_url}" if model is not None else None,
                                              distill=args.distill, derive=args.derive or args.derive_approve,
                                              derive_approve=args.derive_approve)
        if not args.json:
            print("", file=sys.stderr)
        reports.append(report)
        if args.json:
            print(json.dumps(report.as_dict(with_items=False), ensure_ascii=False), file=out)
        else:
            acc = "no reader" if report.accuracy is None else f"accuracy {report.accuracy * 100:.1f}% ({report.correct}/{report.answered}, {report.reader})"
            print(f"{report.source:<28} {report.hops:<6} n={report.questions:<4} gold@{report.k} {report.gold_at_k * 100:5.1f}%  "
                  f"stale above gold {report.stale_above_gold * 100:5.1f}%  {acc}  {report.errors} error(s)", file=out)
            if report.distilled is not None:
                bridges = "no derivation pass" if report.derived is None else f"{report.derived} bridge(s) {'approved' if report.derive_approved else 'left proposed'}"
                print(f"{'':<28} claims: {report.distilled} extracted, {bridges}; claim_gold@{report.k} {(report.claim_gold_at_k or 0) * 100:5.1f}%  "
                      f"derived_gold@{report.k} {(report.derived_gold_at_k or 0) * 100:5.1f}%", file=out)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps([r.as_dict(with_items=True) for r in reports], indent=1), encoding="utf-8")
    return 0


def _staged(record: dict) -> dict:
    """A stage's receipt without its own copy of the passages.

    Every stage replaces the answer's items with its output -- widening,
    withholding, code context and merging all do -- so a receipt's copy
    of the text is always redundant with `items`, and always older than
    it. Emitting one hands back what a later stage removed: text a
    withholding policy took out, or a passage whose source a later stage
    found deleted.

    So the copy is dropped from every stage, unconditionally. Dropping it
    only when some flag is set treats one symptom of the class -- it was
    written that way first, conditioned on a withholding policy, and the
    deletion case walked straight through the gap. The counts are the
    useful part of a receipt and they stay.
    """
    kept = {key: value for key, value in record.items() if key != "items"}
    if "items" in record:
        kept["items_not_repeated"] = (
            "the passages this answer returns are in `items`; this receipt described them at "
            "an earlier stage and its copy is not returned")
    return kept


def read_original_image(filename: str, limit: int) -> tuple[bytes, str, str]:
    """Read only the selected regular file, bounded even if its size changes.

    Signatures identify the raster format, not successful decoding or safety.
    Nonblocking open prevents a selected FIFO from waiting for another writer.
    """
    try:
        descriptor = os.open(filename, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as image:
            info = os.fstat(image.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise InvalidInput("--image requires a regular file")
            if not 0 < info.st_size <= limit:
                raise InvalidInput(f"--image requires a nonempty file up to {limit} bytes")
            data = image.read(limit + 1)
    except OSError as exc:
        raise InvalidInput(f"could not read selected --image file ({type(exc).__name__})") from exc
    if not 0 < len(data) <= limit:
        raise InvalidInput(f"--image requires a nonempty file up to {limit} bytes")
    media_type = (
        "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else
        "image/jpeg" if data.startswith(b"\xff\xd8\xff") else
        "image/gif" if data.startswith((b"GIF87a", b"GIF89a")) else
        "image/webp" if data.startswith(b"RIFF") and data[8:12] == b"WEBP" else None
    )
    if media_type is None:
        raise InvalidInput("--image requires PNG, JPEG, GIF or WebP bytes, not just an image filename")
    return data, media_type, pathlib.Path(filename).name


def _ledger_json(value: object, indent: int | None = 2) -> str:
    """JSON any stream can take. A lone surrogate the ledger holds cannot be
    written as UTF-8, so it is written as its JSON escape, which reads back
    as the same character; everything else stays as it is."""
    return json.dumps(value, ensure_ascii=False, indent=indent).encode("utf-8", "backslashreplace").decode("utf-8")


async def graph_command(args: argparse.Namespace, engine: MemoryEngine, out) -> int:
    """The graph subcommands, on the same projection as the HTTP routes.
    Each reads the clock once, so what it shows and the instant it says it
    was read at agree. Exits 1 when a name is ambiguous or unknown."""
    from ..core.timeutil import format_rfc3339, parse_rfc3339
    from ..entities.context import MAX_NAME, MAX_NAMES, MAX_QUESTION, ContextLimits, graph_connections, graph_context
    from ..entities.export import export_graph
    from ..entities.read import load_projection
    from ..entities.report import render_markdown, report_record
    from ..entities.view import SeedRefused, knowledge_view, walk_seeds
    from ..entities.schema import MAX_BYTES_LIMIT, MAX_PREDICATES, schema_record
    from ..entities.timeline import TimelineEntityAmbiguous, TimelineEntityMissing, timeline_view

    space, command = args.space, args.graph_command
    # Refused here as the HTTP routes refuse them, before anything is read;
    # NaN fails the comparison, so it is refused too.
    if command == "report" and not 0 < args.resolution <= 10:
        raise InvalidInput("--resolution must be a number above 0 and at most 10")
    named = {"path": [args.source, args.target] if command == "path" else [],
             "context": args.names if command == "context" else [],
             "entity": [args.name] if command == "entity" else [],
             "walk": args.names if command == "walk" else []}.get(command, [])
    if len(named) > MAX_NAMES or any(not 1 <= len(name) <= MAX_NAME for name in named):
        raise InvalidInput(f"names: at most {MAX_NAMES}, each 1 to {MAX_NAME} characters")
    if command == "path" and not 1 <= args.max_hops <= 4:
        raise InvalidInput("--max-hops must be from 1 to 4")
    if command == "walk" and args.hops is not None and not 1 <= args.hops <= 8:
        raise InvalidInput("--hops must be from 1 to 8")
    if command == "walk" and not 1 <= args.limit <= 1000:
        raise InvalidInput("--limit must be from 1 to 1000")
    if command == "schema" and not 1 <= args.limit <= MAX_PREDICATES:
        raise InvalidInput(f"--limit must be from 1 to {MAX_PREDICATES}")
    if command == "schema" and not 1_024 <= args.max_bytes <= MAX_BYTES_LIMIT:
        raise InvalidInput(f"--max-bytes must be from 1024 to {MAX_BYTES_LIMIT}")
    if command == "context":
        if args.question is not None and not 1 <= len(args.question) <= MAX_QUESTION:
            raise InvalidInput(f"--question must be 1 to {MAX_QUESTION} characters")
        if args.min_similarity is not None and not -1.0 <= args.min_similarity <= 1.0:
            raise InvalidInput("--min-similarity must be from -1 to 1")
        try:
            ContextLimits(max_bytes=args.max_bytes)
        except ValueError:
            raise InvalidInput("--max-bytes must be between 512 and 64000") from None
    as_of = getattr(args, "as_of", None)
    if as_of is not None:
        try:
            as_of = format_rfc3339(parse_rfc3339(as_of))
        except ValueError:
            raise InvalidInput("--as-of must be an RFC 3339 timestamp") from None
    if command in ("path", "context", "entity"):
        when = engine.clock()
        if command == "path":
            found = await graph_connections(engine, space, args.source, args.target, max_hops=args.max_hops,
                                            as_of=when)
        elif command == "context":
            if not args.names and not args.question:
                raise InvalidInput("give names or --question")
            found = await graph_context(engine, space, names=args.names, question=args.question, as_of=when,
                                        limits=ContextLimits(max_bytes=args.max_bytes), similar=args.similar,
                                        min_similarity=args.min_similarity)
        else:
            found = await graph_context(engine, space, names=[args.name], as_of=when,
                                        limits=ContextLimits(max_hops=1))
        if getattr(args, "json", False):
            print(_ledger_json(found.record(space, "current", when)), file=out)
        else:
            print(found.text, file=out)
        return 0 if found.status == "prepared" else 1
    if command == "match":
        from ..entities.match import MatchQueryError, graph_match

        patterns = [dict(zip(("subject", "predicate", "object"), pattern)) for pattern in args.pattern]
        try:
            when = as_of or engine.clock()
            matched = await graph_match(engine, space, patterns, returns=args.returns, limit=args.limit,
                                        status=args.status, as_of=when, together=not args.apart,
                                        follows=args.follows, max_bytes=args.max_bytes)
        except MatchQueryError as refused:
            raise InvalidInput(str(refused)) from None
        if getattr(args, "json", False):
            print(_ledger_json(matched.record(space, status=args.status, as_of=when, together=not args.apart,
                                              limit=args.limit, follows=args.follows)), file=out)
        else:
            print(matched.text, file=out)
        return 0 if matched.status == "matched" else 1
    if command == "health":
        from ..entities.health import DEFAULT_EXAMPLES, HealthError, MAX_BYTES as HEALTH_BYTES, graph_health

        when = engine.clock()
        try:
            found_health = await graph_health(
                engine, space, limit=args.limit if args.limit is not None else DEFAULT_EXAMPLES, as_of=when,
                max_bytes=args.max_bytes if args.max_bytes is not None else HEALTH_BYTES)
        except HealthError as refused:
            raise InvalidInput(str(refused)) from None
        print(_ledger_json(found_health.record(space, status="current", as_of=when))
              if getattr(args, "json", False) else found_health.text, file=out)
        return 0

    if command == "meanings":
        meanings = engine.relation_meanings
        if getattr(args, "json", False):
            print(_ledger_json({"space": space, "meanings": meanings.record() if meanings else None}), file=out)
            return 0
        if meanings is None:
            print(f"meanings: space {space} — nothing is configured, so the graph holds only what was said",
                  file=out)
            return 0
        from ..entities.meanings import MAX_IMPLIED, MAX_STEPS, MAX_WALKED

        pairs = sorted({tuple(sorted((one, other))) for one, other in meanings.inverse.items()})
        said = [f"meanings: space {space}, as this process is configured"]
        said += [f"  opposites: {one} ↔ {other}" for one, other in pairs]
        said += [f"  reads both ways: {name}" for name in meanings.symmetric]
        said += [f"  carries through: {name}" for name in meanings.transitive]
        said.append(f"  bounds: a chain is followed {MAX_STEPS} claims at most, a projection holds "
                    f"{MAX_IMPLIED} implications, and a walk examines {MAX_WALKED} claims")
        said.append("  what follows is never a claim: it is kept apart, carries the claims under it, "
                    "and its id begins imp:")
        print("\n".join(said), file=out)
        return 0

    if command == "duplicates":
        from ..entities.duplicates import DuplicatesError, likely_duplicates

        when = engine.clock()
        try:
            found_pairs = await likely_duplicates(engine, space, limit=args.limit, min_score=args.min_score, as_of=when,
                                                  max_bytes=args.max_bytes)
        except DuplicatesError as refused:
            raise InvalidInput(str(refused)) from None
        print(_ledger_json(found_pairs.record(space, status="current", as_of=when)) if getattr(args, "json", False)
              else found_pairs.text, file=out)
        return 0
    if command == "changes":
        from ..entities.changes import ChangesError, graph_changes

        try:
            changed = await graph_changes(engine, space, since=args.since, until=args.until or engine.clock(),
                                          limit=args.limit, max_bytes=args.max_bytes)
        except ChangesError as refused:
            raise InvalidInput(str(refused)) from None
        print(_ledger_json(changed.record(space)) if getattr(args, "json", False) else changed.text, file=out)
        return 0 if changed.status == "changed" else 1
    if command == "overview":
        from ..entities.overview import OverviewError, graph_overview

        when = engine.clock()
        try:
            overview = await graph_overview(engine, space, question=args.question, limit=args.limit,
                                            facts_each=args.facts, as_of=when, resolution=args.resolution,
                                            max_bytes=args.max_bytes)
        except OverviewError as refused:
            raise InvalidInput(str(refused)) from None
        if getattr(args, "json", False):
            print(_ledger_json(overview.record(space, status="current", as_of=when, question=args.question)), file=out)
        else:
            print(overview.text, file=out)
        return 0
    if command == "timeline":
        try:
            view = await timeline_view(engine, space, args.name, as_of=as_of)
        except (TimelineEntityAmbiguous, TimelineEntityMissing) as unresolved:
            print(_ledger_json(unresolved.record(args.name), indent=None), file=out)
            return 1
        print(_ledger_json(view), file=out)
        return 0
    when = engine.clock()
    if command == "schema":
        print(_ledger_json(await schema_record(engine, space, as_of=when, limit=args.limit, max_bytes=args.max_bytes)),
              file=out)
        return 0
    if command == "report":
        report = await report_record(engine, space, as_of=when, resolution=args.resolution, usage=args.usage)
        print(render_markdown(report) if args.markdown else _ledger_json(report), file=out)
        return 0
    projection, coverage = await load_projection(engine, space, mode="current", as_of=when)
    if command == "walk":
        try:
            seeds = walk_seeds(projection, args.names, coverage)
        except SeedRefused as refused:
            print(_ledger_json(refused.answer, indent=None), file=out)
            return 1
        print(_ledger_json(knowledge_view(projection, mode="current", as_of=when, limit=args.limit,
                                          attribute_limit=300, coverage=coverage, seeds=seeds,
                                          direction=args.direction, hops=args.hops)), file=out)
        return 0
    reasons = coverage.get("reasons") or []
    exported = export_graph(projection, args.format, about={"status": "current", "as_of": when,
                                                           "coverage": {**coverage, "truncated": bool(reasons)}})
    if args.out:
        with open(args.out, "wb") as file:
            file.write(exported.body)
        print(json.dumps({"written": args.out, "bytes": len(exported.body), "media_type": exported.media_type}),
              file=out)
    elif exported.media_type == "application/zip":
        raise InvalidInput(f"{args.format} is a zip; give --out FILE")
    else:
        print(exported.body.decode("utf-8", "backslashreplace"), file=out)
    return 0


async def run(args: argparse.Namespace, engine: MemoryEngine, stdin, out, settings=None) -> int:
    space = args.space
    emit = lambda obj: print(json.dumps(obj, ensure_ascii=False), file=out)  # noqa: E731
    if args.command == "graph":
        return await graph_command(args, engine, out)

    if args.command == "sync-directory":
        from .directory_cli import run_directory_sync
        return await run_directory_sync(args, engine, out)

    if args.command == "remember":
        if args.image is not None and args.jsonl:
            raise InvalidInput("--image cannot be combined with --jsonl; select a single source note")
        raw = read_source(args.file, stdin)
        attachment = None
        if args.jsonl:
            records = [Record.from_dict(json.loads(line)) for line in raw.splitlines() if line.strip()]
            added = await engine.remember_many(space, records)
        else:
            metadata = parse_pairs(args.meta, "--meta")
            if args.image is not None:
                if not raw.strip():
                    raise InvalidInput("--image needs a nonempty source note for retrieval")
                from ..memory.engine import MAX_ATTACHMENT_BYTES

                data, media_type, name = read_original_image(args.image, min(engine.max_attachment_bytes, MAX_ATTACHMENT_BYTES))
                try:
                    attachment = await engine.attach(space, data, media_type, filename=name)
                    if attachment.attachment_id != hashlib.sha256(data).hexdigest() or attachment.bytes != len(data) or attachment.media_type != media_type:
                        raise InvalidInput("attachment receipt does not match the selected original")
                except Exception as exc:
                    raise InvalidInput("image save unconfirmed; bytes may already be stored. Inspect memory before retrying") from exc
            try:
                added = [await engine.remember(
                    space, raw, kind=args.kind, source=args.source, tags=args.tag,
                    created_at=args.created_at, metadata=metadata,
                    attachment_ids=[attachment.attachment_id] if attachment else [],
                    dedup_key=args.dedup_key, replace=args.replace,
                )]
                if attachment:
                    episode = await engine.episode(space, added[0].episode_id)
                    if not any(item.attachment_id == attachment.attachment_id for item in episode.attachments):
                        raise InvalidInput("saved episode is missing the original attachment link")
            except Exception as exc:
                if attachment:
                    raise InvalidInput("source save unconfirmed; an image or episode may already be stored. Inspect memory before retrying") from exc
                raise
        if args.json:
            for a in added:
                emit(a.model_dump() | ({"attachments": [attachment.model_dump()]} if attachment else {}))
        else:
            fresh = sum(1 for a in added if a.outcome == "accepted")
            updated = sum(1 for a in added if a.outcome == "updated")
            dup = len(added) - fresh - updated
            print(f"remembered {fresh} episode(s)" + (f", {updated} replaced" if updated else "")
                  + (f", {dup} already known" if dup else ""), file=out)
            if attachment:
                print(f"original image linked: {attachment.attachment_id} ({attachment.bytes} bytes)", file=out)
        return 0

    if args.command == "when":
        from ..retrieval.temporal import (DEFAULT_LIMIT as TEMPORAL_LIMIT, MAX_BYTES as TEMPORAL_BYTES,
                                          temporal_answer)

        answered = await temporal_answer(engine, space, args.question, now=args.now,
                                         limit=args.limit if args.limit is not None else TEMPORAL_LIMIT,
                                         max_bytes=args.max_bytes if args.max_bytes is not None else TEMPORAL_BYTES)
        if args.json:
            emit(answered.record(space))
        else:
            print(answered.text, file=out)
        # Computed and recalled are both answers; the rest say why there is none.
        return 0 if answered.status in ("computed", "recalled") else 1

    if args.command == "recall":
        policy: tuple[str, ...] = ()
        if args.withhold:
            from ..retrieval.withhold import chosen_kinds

            # Checked before the search, as the HTTP route does: a policy
            # naming a kind that does not exist is a mistake in the
            # request, and searching first spends the work for an answer
            # nobody receives.
            policy = chosen_kinds(tuple(k.strip() for k in args.withhold.split(",") if k.strip()))
            # Everything refused here re-reads the episode from the store
            # *after* withholding and prints source verbatim, which hands
            # back what was just withheld: --merge and --code-context both
            # quote the raw episode, and --parts answers without passing
            # through withholding at all. The HTTP route refuses the same
            # class of combination; the CLI refusing a different set would
            # be the same hole wearing different clothes.
            clashes = [name for name, asked_for in (
                ("--merge", args.merge), ("--code-context", args.code_context),
                ("--parts", args.parts)) if asked_for]
            if clashes:
                raise InvalidInput(
                    f"--withhold cannot be combined with {', '.join(clashes)}: each of those "
                    f"quotes the episode again after withholding, which would hand back what "
                    f"was withheld; ask for one or the other")
        if args.parts:
            from ..retrieval.parts import recall_parts

            parted = await recall_parts(
                engine, space, args.query, limit=args.limit, as_of=args.as_of, tags=args.tag,
                where=parse_pairs(args.where, "--where"), kind=args.kind,
                source_prefix=args.source_prefix, since=args.since, until=args.until,
                rerank=not args.no_rerank, graph_boost=args.graph_boost)
            if args.json:
                emit(parted.record())
                return 0
            print(parted.text(), file=out)
            for lane in {lane for part in parted.per_part for lane in part.degraded}:
                print(f"degraded: {lane}", file=sys.stderr)
            return 0
        result = await engine.recall(
            space, args.query, limit=args.limit, as_of=args.as_of, tags=args.tag, where=parse_pairs(args.where, "--where"),
            history=args.history, kind=args.kind, source_prefix=args.source_prefix, since=args.since, until=args.until,
            conditions=read_conditions(args.conditions), candidate_limit=args.candidate_limit,
            rerank=not args.no_rerank, graph_boost=args.graph_boost,
        )
        kept = None
        opened = None
        if args.window:
            from ..retrieval.window import widen

            opened = await widen(engine, space, result.items,
                                 before=args.window, after=args.window)
            result = result.model_copy(update={"items": list(opened.items)})
        if policy:
            from ..retrieval.withhold import withhold

            # Facts and history too. Fixing this on the HTTP route and
            # not here left the same address reachable through the CLI.
            kept = withhold(result.items, facts=list(result.facts) + list(result.history),
                            kinds=policy)
            result = result.model_copy(update={
                "items": list(kept.items),
                "facts": list(kept.facts[:len(result.facts)]),
                "history": list(kept.facts[len(result.facts):])})
        joined = None
        if args.merge:
            from ..retrieval.merging import merge_neighbours

            joined = await merge_neighbours(engine, space, result.items)
            result = result.model_copy(update={"items": list(joined.items)})
        inside = None
        if args.code_context:
            from ..retrieval.code_context import code_context

            # **Last**, after every stage that re-reads a source. Code
            # context quotes the file -- a signature, an import line --
            # and a stage running after it can find that source deleted,
            # leaving the answer correctly empty while the context still
            # prints the text. Dropping the receipt's copy of the items
            # cannot fix that, because the quoted signature is a separate
            # copy of the same source. Running last is what makes the
            # context describe passages that survived.
            inside = await code_context(engine, space, result.items)
            # A source this stage finds gone is dropped from the answer
            # too, not only from the context.
            result = result.model_copy(update={"items": list(inside.items)})
        if args.json:
            said = result.model_dump() | {"context_reduction": result.context_reduction}
            emit(said | ({"merged": _staged(joined.record())} if joined else {})
                      | ({"widened": _staged(opened.record())} if opened else {})
                      | ({"withheld": _staged(kept.record())} if kept else {})
                      | ({"code_context": _staged(inside.record())} if inside else {}))
            return 0
        if kept is not None:
            print(kept.why, file=out)
        if opened is not None:
            print(opened.why, file=out)
        if inside is not None:
            print(inside.why, file=out)
            for chunk, context in inside.by_chunk.items():
                for holder in context.holders:
                    cut = " (quoted to the line bound, so incomplete)" if holder.clipped else ""
                    print(f"  #{chunk} inside {holder.name} (line {holder.line}){cut}", file=out)
                    for line in holder.text.splitlines():
                        print(f"      {line}", file=out)
                for brought in context.imports:
                    print(f"  #{chunk} line {brought.line}: {brought.text}", file=out)
        if joined is not None:
            print(joined.why, file=out)
            for chunk, absorbed in joined.from_chunks.items():
                print(f"joined {len(absorbed)} chunk(s) into #{chunk}: "
                      f"{', '.join(str(c) for c in absorbed)}", file=out)
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

    if args.command == "source-key":
        episode = await engine.episode_by_key(space, args.dedup_key)
        if args.json:
            emit(episode.model_dump())
        else:
            print(f"episode {episode.episode_id} in {space}: {len(episode.attachments)} attachment(s)", file=out)
            print(json.dumps(episode.content, ensure_ascii=True), file=out)
        return 0

    if args.command == "attachments":
        if not 0 < args.episode_id < 2**63:
            raise InvalidInput("episode_id must be a positive 64-bit signed integer")
        episode = await engine.episode(space, args.episode_id)
        if args.json:
            emit({"space": space, "episode_id": episode.episode_id,
                  "attachments": [attachment.model_dump() for attachment in episode.attachments]})
        else:
            print(f"episode {episode.episode_id} in {space}: {len(episode.attachments)} attachment(s)", file=out)
            for attachment in episode.attachments:
                # Quote untrusted filenames so terminal control characters stay inert.
                name = json.dumps(attachment.filename, ensure_ascii=True) if attachment.filename is not None else "(unnamed)"
                print(f"{attachment.attachment_id}  {attachment.media_type}  {attachment.bytes} bytes  {name}", file=out)
            if not episode.attachments:
                print("no attachments", file=out)
        return 0

    if args.command in ("jobs", "job", "cancel-job"):
        if args.command == "jobs":
            found = await engine.jobs(space, args.limit)
            if args.json:
                for job in found:
                    emit(job_payload(job))
            elif not found:
                print("no ingest jobs in this space yet", file=out)
            else:
                for job in found:
                    print(job_line(job), file=out)
            return 0
        job = await (engine.cancel_job(space, args.job_id) if args.command == "cancel-job"
                     else engine.job(space, args.job_id))
        emit(job_payload(job)) if args.json else print(job_line(job), file=out)
        return 0

    if args.command == "merge-space":
        if args.dry_run:
            preview = await engine.merge_space(space, into=args.into, preview=True)
            emit(preview.record()) if args.json else print(
                f"would move {preview.episodes} episode(s) and {preview.facts} claim(s) "
                f"from {space} into {args.into}", file=out)
            return 0
        if args.confirm != space:
            print(f"refusing: --confirm must repeat the space being merged {space!r}; nothing moved", file=out)
            return 2
        moved = await engine.merge_space(space, into=args.into, confirm=args.confirm)
        emit(moved.record()) if args.json else print(
            f"moved {moved.episodes} episode(s) and {moved.facts} claim(s) from {space} "
            f"into {args.into}; {space} is closed", file=out)
        return 0

    if args.command == "delete-space":
        if args.dry_run:
            receipt = await engine.space_impact(space)
            emit(receipt.model_dump()) if args.json else print(f"would delete space {space}: {space_line(receipt)}", file=out)
            return 0
        if args.confirm != space:
            print(f"refusing: --confirm must repeat the space name {space!r}; nothing was deleted", file=out)
            return 2
        receipt = await engine.delete_space(space)
        emit({"deleted": space, **receipt.model_dump()}) if args.json else print(f"deleted space {space}: {space_line(receipt)}", file=out)
        return 0

    if args.command == "forget":
        if args.dry_run:
            forget_receipt = await engine.impact(space, args.episode_id)
            emit(forget_receipt.model_dump()) if args.json else print(f"would forget episode {args.episode_id}: {receipt_line(forget_receipt)}", file=out)
            return 0
        forget_receipt = await engine.forget(space, args.episode_id)
        emit({"forgotten": args.episode_id, **forget_receipt.model_dump()}) if args.json else print(f"forgot episode {args.episode_id}: {receipt_line(forget_receipt)}", file=out)
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
            origin=args.origin, proposed=args.propose, extends=args.extends, derived_from=args.derived_from,
        )
        emit(fact.model_dump()) if args.json else print(fact_line(fact), file=out)
        return 0

    if args.command == "link":
        link = await engine.link_facts(space, args.from_fact, args.to_fact, args.kind,
                                       source_episode_id=args.source, quote=args.quote)
        emit(link.model_dump()) if args.json else print(link_line(link), file=out)
        return 0

    if args.command == "answer":
        from ..retrieval.router import answer_question

        routed = await answer_question(engine, space, args.question, now=args.now, limit=args.limit,
                                       route=args.route)
        if getattr(args, "json", False):
            print(_ledger_json(routed.record(space)), file=out)
            return 0
        print(f"route: {routed.route} — {routed.why}", file=out)
        print(routed.text, file=out)
        return 0

    if args.command == "sync":
        return await sync_command(args, engine, out)

    if args.command == "map":
        return await map_command(args, engine, out)
    if args.command == "fs":
        return await filesystem_command(args, engine, stdin, out)
    if args.command == "doctor":
        report = await engine.doctor(space)
        if args.json:
            emit(report.model_dump())
        else:
            print(("healthy" if report.healthy else "orphans found") + f": {report.episodes} episode(s), {report.chunks} chunk(s), "
                  f"{report.facts} fact(s), {report.links} link(s), {report.tombstones} tombstone(s)", file=out)
            for name in ("chunks_without_episode", "vectors_without_chunk", "facts_citing_forgotten", "facts_citing_unknown",
                         "links_with_missing_ends", "attachments_unlinked"):
                found = getattr(report, name)
                if found:
                    print(f"  {name}: {', '.join(str(i) for i in found)}", file=out)
            if report.not_inspected:
                print(f"  not inspected: {', '.join(report.not_inspected)}", file=out)
        return 0

    if args.command == "vectors":
        reembedded = None
        if args.reembed:
            reembedded = await engine.reembed_vectors()
        elif args.adopt:
            await engine.adopt_vector_identity()
        identity = engine.vector_identity
        vector_state: dict[str, object] = {
            "state": identity.state if identity else "unsettled", "writer": identity.writer if identity else None,
            "recorded": identity.recorded if identity else None, "blocked": engine.vector_block}
        if reembedded is not None:
            vector_state.update(spaces=list(reembedded.spaces), chunks=reembedded.chunks,
                                orphans_removed=reembedded.orphans_removed)
        if args.json:
            emit(vector_state)
        else:
            print(f"vectors {vector_state['state']}: written by {vector_state['recorded'] or 'unrecorded'}, "
                  f"this engine writes {vector_state['writer']}", file=out)
            if reembedded is not None:
                print(f"  re-embedded {reembedded.chunks} chunk(s) in {len(reembedded.spaces)} space(s); "
                      f"removed {reembedded.orphans_removed} orphan vector(s)", file=out)
            if vector_state["blocked"]:
                print(f"  vector lane off: {vector_state['blocked']}", file=out)
        return 0

    if args.command == "expire":
        from .config import parse_retention

        expiry = await engine.expire(space, parse_retention(",".join(args.keep)), limit=args.limit, dry_run=args.dry_run)
        if args.json:
            emit(expiry.model_dump())
        elif args.dry_run:
            print(f"would forget {expiry.remaining} episode(s) under {expiry.policy}", file=out)
        else:
            print(f"forgot {len(expiry.forgotten)} episode(s) under {expiry.policy}; {expiry.remaining} left for the next pass", file=out)
        return 0

    if args.command == "links":
        links = await engine.fact_links(space, args.fact_id)
        if args.json:
            emit([link.model_dump() for link in links])
        elif not links:
            print("no links", file=out)
        else:
            for link in links:
                print(link_line(link), file=out)
        return 0

    if args.command == "audit-grounding":

        from ..observability.audit import audit_grounding

        findings = await audit_grounding(engine, space, statuses=tuple(args.status or ("active",)))
        shown = [f for f in findings if f.flagged] if args.flagged_only else findings
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
        flagged = sum(1 for f in findings if f.flagged)
        counted = "claim needs" if flagged == 1 else "claims need"
        print(f"{flagged} of {len(findings)} {counted} a person" if flagged
              else f"nothing flagged in {len(findings)} extracted claims", file=out)
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

    if args.command in ("approve", "decline", "exclude", "include", "reconsider", "reopen"):
        import getpass

        actor = f"cli:{getpass.getuser()}"
        if args.command == "approve":
            fact = await engine.approve(space, args.fact_id, actor=actor)
        elif args.command == "decline":
            fact = await engine.decline(space, args.fact_id, args.reason, actor=actor)
        elif args.command == "exclude":
            fact = await engine.exclude(space, args.fact_id, args.reason, actor=actor)
        elif args.command == "reconsider":
            fact = await engine.reconsider(space, args.fact_id, args.reason, actor=actor)
        elif args.command == "reopen":
            fact = await engine.reopen(space, args.fact_id, args.reason, actor=actor)
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
            emit({"static_facts": [f.model_dump() for f in profile.static_facts], "dynamic": profile.dynamic,
                  "recent": [asdict(r) for r in profile.recent]})
        else:
            for f in profile.static_facts:
                print(fact_line(f), file=out)
            for line in profile.dynamic:
                print(f"- {line}", file=out)
        return 0

    if args.command == "derive":
        from . import config as runtime_config
        from ..ingestion.derive import Deriver

        chat = runtime_config.build_chat(settings)
        if chat is None:
            print("error: no consolidation model configured (SCONE_CHAT_URL and SCONE_CHAT_MODEL)", file=sys.stderr)
            return 2
        outcome = await Deriver(engine, chat).derive(space, limit_groups=args.limit)
        if args.json:
            emit({"space": space, **outcome.as_payload()})
        else:
            print(f"derived over {outcome.sent} of {outcome.groups} group(s): {len(outcome.proposed)} proposed, "
                  f"{outcome.restated} restated, {len(outcome.rejected)} rejected", file=out)
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
        summary = await engine.import_records(space, [json.loads(line) for line in raw.splitlines() if line.strip()],
                                              resurrect=args.resurrect)
        emit(summary.__dict__) if args.json else print(
            f"imported {summary.episodes} episode(s), {summary.facts} fact(s); already known: "
            f"{summary.deduplicated} episode(s), {summary.facts_skipped} fact(s)"
            + (f"; forgotten here and left so: {summary.tombstoned}" if summary.tombstoned else ""), file=out
        )
        return 0

    raise SystemExit(f"unknown command {args.command}")


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None, stdin=None, out=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args, rest = build_parser().parse_known_args(raw)
    env = os.environ if env is None else env
    if args.command == "agent-hook":
        from ..capture.agent_hook import run_hook

        return run_hook(rest, (stdin or sys.stdin).read(), env, stdout=out or sys.stdout)
    if rest:
        build_parser().error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "serve-conversations":
        try:
            settings = settings_for_cli(env)
        except (ValueError, SconeError):
            print("invalid conversation server settings; check SCONE_* configuration", file=sys.stderr)
            return 2
        try:
            from ..api.conversation_server import main as serve_conversations
        except ImportError:
            print("conversation serving needs pip install 'scone-memory[api]'", file=sys.stderr)
            return 2

        return serve_conversations(settings, journal=args.journal,
                                   model_factory=args.model_factory)
    if args.command == "bench-graph":
        return graph_bench_command(args, out or sys.stdout)
    settings = settings_for_cli(env)
    if args.command == "serve":
        from ..api.__main__ import main as serve

        serve(settings)  # same SQLite default as the other commands
        return 0
    if args.command in ("bench", "bench-conflicts", "bench-temporal", "bench-code", "bench-route",
                        "bench-parts", "calibrate", "tune"):
        command = {"bench": bench_command, "bench-conflicts": conflicts_command,
                   "bench-temporal": temporal_command, "bench-code": bench_code_command,
                   "bench-route": route_command, "bench-parts": parts_command,
                   "calibrate": calibrate_command, "tune": tune_command}[args.command]
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
