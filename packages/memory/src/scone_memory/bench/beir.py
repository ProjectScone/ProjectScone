"""Retrieval measured on a BEIR dataset, read from its own files.

A BEIR dataset is a directory: ``corpus.jsonl`` (``_id``, ``title``,
``text``), ``queries.jsonl`` (``_id``, ``text``) and ``qrels/<split>.tsv``
(``query-id``, ``corpus-id``, ``score``), the score a relevance grade
where 0 is judged not relevant. Nothing else is needed to read one, so no
package is: the files are read here, any split is chosen, and a
judgement naming a document or query the files do not hold is counted.

The corpus is stored in a fresh in-process memory, one episode per
document under ``beir:<id>``, each query recalled, and the passages
returned mapped back to documents in rank order, a document chunked many
times counted once at its best rank. Scores are graded nDCG (gain equal
to the grade, as trec_eval computes it), recall and precision of judged
relevant documents, and reciprocal rank, at each k asked for.

A corpus too large for the machine may be cut to ``max_documents``; every
document judged for the queries run is kept and the rest filled in file
order. A cut corpus has fewer distractors and scores higher, and the
report says it was cut.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..core.errors import InvalidInput

#: Passages recalled per k asked for, so documents chunked many times still fill k places.
OVERSAMPLE = 3
#: Documents a run stores at most unless the caller cuts the corpus further.
MAX_DOCUMENTS = 200_000


@dataclass(frozen=True)
class BeirSet:
    name: str
    split: str
    corpus: dict[str, tuple[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    #: Judgement rows naming a document the corpus does not hold.
    unknown_documents: int = 0
    #: Judgement rows naming a query the queries file does not hold.
    unknown_queries: int = 0
    #: Queries with no judgement at all; not run, since nothing could score them.
    unjudged_queries: int = 0


@dataclass(frozen=True)
class Sampled:
    data: BeirSet
    corpus_documents: int
    kept_documents: int
    reduced: bool
    seed: int
    why: str = ""


def _lines(path: Path, name: str) -> list[dict]:
    if not path.is_file():
        raise InvalidInput(f"a BEIR directory needs {name}; {path} is not there")
    rows = []
    with open(path, encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise InvalidInput(f"{name} line {number} is not JSON: {error}") from None
            if not isinstance(row, dict) or not isinstance(row.get("_id"), str):
                raise InvalidInput(f"{name} line {number} has no string _id")
            rows.append(row)
    return rows


def load_beir(directory: str | Path, split: str = "test") -> BeirSet:
    """The dataset in ``directory``, with the judgements of ``split``."""
    root = Path(directory)
    corpus = {row["_id"]: (str(row.get("title") or ""), str(row.get("text") or ""))
              for row in _lines(root / "corpus.jsonl", "corpus.jsonl")}
    queries = {row["_id"]: str(row.get("text") or "") for row in _lines(root / "queries.jsonl", "queries.jsonl")}
    judged_path = root / "qrels" / f"{split}.tsv"
    if not judged_path.is_file():
        raise InvalidInput(f"no judgements for the {split} split: qrels/{split}.tsv is not in {root}")
    qrels: dict[str, dict[str, int]] = {}
    unknown_documents = unknown_queries = 0
    with open(judged_path, encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            cells = line.rstrip("\n").split("\t")
            if number == 1 and cells[:2] == ["query-id", "corpus-id"]:
                continue
            if not line.strip():
                continue
            if len(cells) != 3:
                raise InvalidInput(f"qrels/{split}.tsv line {number} has {len(cells)} fields, not 3")
            query_id, doc_id, grade_text = cells
            try:
                grade = int(grade_text)
            except ValueError:
                raise InvalidInput(f"qrels/{split}.tsv line {number}: the grade {grade_text!r} is not a whole number") from None
            if grade < 0:
                raise InvalidInput(f"qrels/{split}.tsv line {number}: a grade is not negative")
            if doc_id not in corpus:
                unknown_documents += 1
                continue
            if query_id not in queries:
                unknown_queries += 1
                continue
            qrels.setdefault(query_id, {})[doc_id] = grade
    return BeirSet(name=root.name, split=split, corpus=corpus, queries=queries, qrels=qrels,
                   unknown_documents=unknown_documents, unknown_queries=unknown_queries,
                   unjudged_queries=sum(query not in qrels for query in queries))


def sampled(data: BeirSet, *, queries: Optional[int] = None, seed: int = 0,
            max_documents: Optional[int] = None) -> Sampled:
    """The judged queries to run, ``queries`` of them chosen by ``seed``, over a corpus
    cut to ``max_documents`` that keeps every document judged for them."""
    judged = sorted(data.qrels)
    if not judged:
        raise InvalidInput(f"no query is judged against a document the corpus holds: {data.unknown_documents} "
                           f"judgements name unknown documents and {data.unknown_queries} unknown queries")
    if queries is not None:
        if queries < 1:
            raise InvalidInput(f"queries is a count from 1, not {queries}")
        judged = sorted(judged, key=lambda query: hashlib.sha256(f"{seed}:{query}".encode()).hexdigest())[:queries]
    limit = MAX_DOCUMENTS if max_documents is None else max_documents
    needed = [doc for query in judged for doc in data.qrels[query]]
    keep = dict.fromkeys(needed)
    if len(keep) > limit:
        raise InvalidInput(f"the queries chosen judge {len(keep)} documents, more than max_documents {limit}")
    for doc in data.corpus:
        if len(keep) >= limit:
            break
        keep.setdefault(doc)
    reduced = len(keep) < len(data.corpus)
    why = (f"corpus cut from {len(data.corpus)} to {len(keep)} documents, every judged one kept; "
           "scores on a cut corpus run higher than on the whole" if reduced else "the whole corpus")
    subset = BeirSet(name=data.name, split=data.split, corpus={doc: data.corpus[doc] for doc in keep},
                     queries={query: data.queries[query] for query in judged},
                     qrels={query: data.qrels[query] for query in judged},
                     unknown_documents=data.unknown_documents, unknown_queries=data.unknown_queries,
                     unjudged_queries=data.unjudged_queries)
    return Sampled(data=subset, corpus_documents=len(data.corpus), kept_documents=len(keep), reduced=reduced,
                   seed=seed, why=why)


def _check_k(k: int) -> None:
    if type(k) is not int or k < 1:
        raise InvalidInput(f"k is a whole number from 1, not {k!r}")


def graded_ndcg_at(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """Graded nDCG at ``k``: each document's grade over ``log2(rank + 1)``, against the judged grades in the best order."""
    _check_k(k)
    gained = sum(judged.get(doc, 0) / log2(rank + 1) for rank, doc in enumerate(ranked[:k], start=1))
    ideal = sum(grade / log2(rank + 1)
                for rank, grade in enumerate(sorted(judged.values(), reverse=True)[:k], start=1))
    return gained / ideal if ideal else 0.0


def recall_at_graded(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """The share of the documents judged relevant (grade above 0) among the first ``k``."""
    _check_k(k)
    relevant = {doc for doc, grade in judged.items() if grade > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def precision_at_graded(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """The share of the first ``k`` places held by documents judged relevant."""
    _check_k(k)
    return sum(judged.get(doc, 0) > 0 for doc in ranked[:k]) / k


def reciprocal_rank_at(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """One over the rank of the first document judged relevant within ``k``; 0 when none is."""
    _check_k(k)
    return next((1 / rank for rank, doc in enumerate(ranked[:k], start=1) if judged.get(doc, 0) > 0), 0.0)


def fitted_query(text: str) -> str:
    """``text`` as recall takes it: whole when it fits, otherwise cut at the
    last space within the limit, or where the limit falls inside one token."""
    from ..core.validation import MAX_QUERY

    if len(text) <= MAX_QUERY:
        return text
    head = text[:MAX_QUERY + 1]
    space = head.rfind(" ")
    return head[:space].rstrip() if space > 0 else text[:MAX_QUERY]


@dataclass
class BeirReport:
    sample: Sampled
    ks: tuple[int, ...]
    metrics: dict[str, float]
    per_query: list[dict] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)
    short: int = 0
    #: Queries longer than recall takes, scored as cut at a word.
    cut: int = 0
    #: Queries with no text, which retrieve nothing and score zero.
    empty: int = 0

    def record(self) -> dict:
        data = self.sample.data
        return {"dataset": {"name": data.name, "split": data.split, "unknown_documents": data.unknown_documents,
                            "unknown_queries": data.unknown_queries, "unjudged_queries": data.unjudged_queries},
                "corpus": {"documents": self.sample.corpus_documents, "stored": self.sample.kept_documents,
                           "reduced": self.sample.reduced, "why": self.sample.why},
                "queries_run": len(data.queries), "seed": self.sample.seed, "ks": list(self.ks),
                "metrics": self.metrics, "queries_short": self.short, "queries_cut": self.cut,
                "queries_empty": self.empty, "config": self.config,
                "per_query": self.per_query, "measured": True}

    def text(self) -> str:
        lines = [f"BEIR {self.sample.data.name} ({self.sample.data.split}): {len(self.sample.data.queries)} queries, "
                 f"{self.sample.kept_documents} documents stored; {self.sample.why}"]
        lines += [f"  {name}: {value:.4f}" for name, value in self.metrics.items()]
        if self.short:
            lines.append(f"  {self.short} queries returned fewer documents than the largest k")
        if self.cut:
            lines.append(f"  {self.cut} queries were longer than recall takes and were cut at a word")
        if self.empty:
            lines.append(f"  {self.empty} queries were empty, retrieved nothing and scored zero")
        return "\n".join(lines)


async def run_beir(sample: Sampled, *, ks: Sequence[int] = (1, 3, 10), engine=None) -> BeirReport:
    """Store the sample's corpus, recall each query, and score the documents returned."""
    from .. import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from ..core.validation import MAX_LIMIT
    from ..ingestion.records import Record

    ks = tuple(sorted(set(ks)))
    for k in ks:
        _check_k(k)
    if not ks:
        raise InvalidInput("ask for at least one k")
    owned = engine is None
    memory = engine or await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    data = sample.data
    try:
        records = [Record(content=f"{title}\n\n{text}".strip(), source=f"beir:{doc}", dedup_key=doc)
                   for doc, (title, text) in data.corpus.items() if f"{title}{text}".strip()]
        for start in range(0, len(records), 500):
            await memory.remember_many("beir", records[start:start + 500])
        deepest = ks[-1]
        per_query: list[dict] = []
        totals = {name: 0.0 for name in _names(ks)}
        short = cut = empty = 0
        for query_id, text in data.queries.items():
            asked = fitted_query(text)
            marks: dict[str, bool] = {}
            if not asked:
                ranked: list[str] = []
                marks["empty"] = True
                empty += 1
            else:
                if asked != text:
                    marks["cut"] = True
                    cut += 1
                result = await memory.recall("beir", asked, limit=min(MAX_LIMIT, deepest * OVERSAMPLE))
                ranked = list(dict.fromkeys(item.source.removeprefix("beir:") for item in result.items
                                            if item.source and item.source.startswith("beir:")))
            judged = data.qrels[query_id]
            short += len(ranked) < deepest
            scores: dict[str, float] = {}
            for k in ks:
                scores[f"ndcg@{k}"] = graded_ndcg_at(ranked, judged, k)
                scores[f"recall@{k}"] = recall_at_graded(ranked, judged, k)
                scores[f"precision@{k}"] = precision_at_graded(ranked, judged, k)
            scores[f"mrr@{deepest}"] = reciprocal_rank_at(ranked, judged, deepest)
            for name, value in scores.items():
                totals[name] += value
            per_query.append({"query_id": query_id, "ranked": ranked[:deepest], **marks,
                              "scores": {name: round(value, 4) for name, value in scores.items()}})
        count = len(data.queries)
        metrics = {name: round(total / count, 4) if count else 0.0 for name, total in totals.items()}
        config = {"embedder": memory.embedder.id, "oversample": OVERSAMPLE, "store": "in-process" if owned else "given"}
        return BeirReport(sample=sample, ks=ks, metrics=metrics, per_query=per_query, config=config, short=short,
                          cut=cut, empty=empty)
    finally:
        if owned:
            await memory.close()


def _names(ks: Sequence[int]) -> list[str]:
    names = [f"{metric}@{k}" for k in ks for metric in ("ndcg", "recall", "precision")]
    return names + [f"mrr@{max(ks)}"]
