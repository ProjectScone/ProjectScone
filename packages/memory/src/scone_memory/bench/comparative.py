"""The same questions, the same passages, the same embedder: ours beside LlamaIndex.

The north star names LlamaIndex as the framework to beat, and a claim of
that kind is worth exactly the measurement behind it. This runs the
reference framework itself -- installed, unmodified -- over the same bench
items we score ourselves with: every session becomes one of its Documents,
its ``SentenceSplitter`` cuts them, its ``VectorStoreIndex`` embeds them
through an adapter around **our** embedder, and its retriever ranks the
nodes for the question. Each side's ranking is folded to distinct
sessions in rank order -- its nodes, our passages -- and our recall is
asked for enough passages that the top ``k`` sessions are reachable, so
the two sides are scored by the same rule (session recall at k, MRR,
precision and NDCG at k) on the same items. No model, reranker or query
rewriting runs on either side, and the configuration both ran under is
written into the report.

What this measures is retrieval over conversation sessions with one
embedder. It does not measure answer quality, and it is not the reference
framework at its best configuration; it is the two frameworks' default
retrieval on equal footing, which is the honest starting line.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Coroutine, Iterable, Mapping, Optional, Protocol, Sequence, TypeVar

from ..core.ports import Embedder
from ..core.embedding import embed_queries
from ..ingestion.embedding_cache import cache_key
from ..ingestion.vectors import validated_vectors
from ..providers.llm import ChatModel
from . import metrics as bench_metrics
from .runner import BenchItem, ItemResult, run

T = TypeVar("T")


def _blocking(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code, even when a loop
    is already running in this thread (LlamaIndex calls embedders
    synchronously from inside our async bench)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _adapter_class() -> Any:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.embeddings import BaseEmbedding

    class SconeEmbeddingAdapter(BaseEmbedding):
        """LlamaIndex's embedding interface over our ``Embedder`` port, so
        both sides embed with the same vectors."""

        _scone: Any = PrivateAttr()

        def __init__(self, embedder: Embedder) -> None:
            super().__init__(model_name=embedder.id, embed_batch_size=64)
            self._scone = embedder

        @classmethod
        def class_name(cls) -> str:
            return "SconeEmbeddingAdapter"

        def _get_query_embedding(self, query: str) -> list[float]:
            return _blocking(embed_queries(self._scone, [query]))[0]

        def _get_text_embedding(self, text: str) -> list[float]:
            return _blocking(self._scone.embed([text]))[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return _blocking(self._scone.embed(list(texts)))

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return (await embed_queries(self._scone, [query]))[0]

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return (await self._scone.embed([text]))[0]

        async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return await self._scone.embed(list(texts))

    return SconeEmbeddingAdapter


def SconeEmbedding(embedder: Embedder) -> Any:  # noqa: N802 - reads as the adapter's name at call sites
    """An instance of the LlamaIndex embedding adapter around ``embedder``."""
    return _adapter_class()(embedder)


class VectorCache(Protocol):
    """What a bench's embedding wrapper needs of a cache: the part of
    ``ingestion.embedding_cache.EmbeddingCache`` that reads, writes and says what it holds."""

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]: ...

    def keep(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None: ...

    def record(self) -> dict[str, object]: ...


class CachedEmbedder:
    """A model that is asked only for texts the cache does not hold.

    A real model's run is spent embedding. The sweep's engines, the
    reference's index and every recall's question go through wrappers over
    one cache, keyed as the engine's own embedding cache keys a chunk (the
    model's id, its width and the exact text), so a text is embedded once
    for the whole run and a rerun reads every vector back. The wrapper is
    the model to whoever holds it: the same id and width, and the model's
    input window and tokenizer where it has them, so a token chunk is
    counted by the model and the engine's default voice reads the model's
    id. Each wrapper counts for itself: texts asked for, served from the
    cache, sent to the model, and the wall seconds the model took. A cache
    that fails raises: a bench that quietly embedded again would misreport
    what it cost.

    The model's window drops the tokens past it without saying so, so where
    the model has both a window and a tokenizer the wrapper counts the texts
    asked for that ran past it and the tokens it cut, a text read back as
    well as one embedded, since its vector was cut when it was made. Where it
    cannot count, both are None rather than a zero that would read as a fact."""

    def __init__(self, inner: Embedder, cache: VectorCache) -> None:
        self.inner = inner
        self.cache = cache
        self.id = inner.id
        self.dim = inner.dim
        self.max_input_tokens: Optional[int] = getattr(inner, "max_input_tokens", None)
        counter = getattr(inner, "count_tokens", None)
        self.count_tokens: Optional[Callable[[str], int]] = counter if callable(counter) else None
        self.asked = self.reused = self.embedded = 0
        self.seconds = 0.0
        counts_window = self.count_tokens is not None and self.max_input_tokens is not None
        self.over_window: Optional[int] = 0 if counts_window else None
        self.tokens_past_window: Optional[int] = 0 if counts_window else None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=False)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=True)

    async def _embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        identity = self.id + ":query-cache" if query else self.id
        keys = [cache_key(identity, self.dim, text) for text in texts]
        found = self.cache.take(keys, self.dim)
        first_text: dict[str, str] = {}
        for key, text in zip(keys, texts):
            first_text.setdefault(key, text)
        wanted = [key for key in first_text if key not in found]
        fresh: dict[str, list[float]] = {}
        if wanted:
            started = time.perf_counter()
            pending = [first_text[key] for key in wanted]
            answered = await embed_queries(self.inner, pending) if query else await self.inner.embed(pending)
            self.seconds += time.perf_counter() - started
            fresh = dict(zip(wanted, validated_vectors(answered, len(wanted), self.dim)))
            self.cache.keep(fresh, self.dim)
        self.asked += len(texts)
        self.reused += sum(1 for key in keys if key in found)
        self.embedded += len(wanted)
        self._count_past_window(texts)
        return [list(fresh[key]) if key in fresh else list(found[key]) for key in keys]

    def _count_past_window(self, texts: Sequence[str]) -> None:
        if self.count_tokens is None or self.max_input_tokens is None:
            return
        past = [tokens - self.max_input_tokens for tokens in map(self.count_tokens, texts)
                if tokens > self.max_input_tokens]
        self.over_window = (self.over_window or 0) + len(past)
        self.tokens_past_window = (self.tokens_past_window or 0) + sum(past)

    def record(self) -> dict[str, Any]:
        return {"asked": self.asked, "reused": self.reused, "embedded": self.embedded,
                "embed_seconds": round(self.seconds, 3), "over_window": self.over_window,
                "tokens_past_window": self.tokens_past_window}


class OneThreadCache:
    """An embedding cache every call of which runs on one thread of its own.

    The reference builds its index in a worker thread and embeds from
    there, and a SQLite connection may be used only by the thread that
    opened it; so the cache is opened on this thread and asked there,
    whichever thread asks."""

    def __init__(self, open_cache: Callable[[], VectorCache]) -> None:
        self._thread = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-cache")
        self._cache = self._thread.submit(open_cache).result()

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]:
        return self._thread.submit(self._cache.take, keys, dim).result()

    def keep(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None:
        self._thread.submit(self._cache.keep, vectors, dim).result()

    def record(self) -> dict[str, object]:
        return self._thread.submit(self._cache.record).result()


def bench_embedder(name: str, *, cache_dir: Optional[str] = None) -> Embedder:
    """The embedder a comparison runs on both sides, by name: ``hash`` for
    the hashed-token embedder every earlier row used, or a local model
    (``embedders.local.MODELS``) run in-process through fastembed, its
    files kept in ``cache_dir``."""
    from ..embedders import local
    from ..embedders.hash import HashEmbedder

    if name == "hash":
        return HashEmbedder()
    if name not in local.MODELS:
        raise ValueError(f"unknown bench embedder {name!r}; known: hash, {', '.join(sorted(local.MODELS))}")
    return local.LocalEmbedder(name, cache_dir=cache_dir)


def _llm_class() -> Any:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    class SconeChatAdapter(CustomLLM):
        """LlamaIndex's LLM interface over our ``ChatModel`` port, so both
        synthesizers write with the same local model. Every prompt the
        reference builds goes through as the user turn, unchanged."""

        _scone: Any = PrivateAttr()
        _calls: int = PrivateAttr(default=0)

        def __init__(self, model: ChatModel) -> None:
            super().__init__()
            self._scone = model
            self._calls = 0

        @classmethod
        def class_name(cls) -> str:
            return "SconeChatAdapter"

        @property
        def calls(self) -> int:
            return self._calls

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(model_name="scone-chat", context_window=8_192, num_output=1_024, is_chat_model=False)

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            self._calls += 1
            return CompletionResponse(text=_blocking(self._scone.complete("", prompt)))

        @llm_completion_callback()
        async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            self._calls += 1
            return CompletionResponse(text=await self._scone.complete("", prompt))

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            yield self.complete(prompt, formatted, **kwargs)

    return SconeChatAdapter


@dataclass(frozen=True)
class ReferenceSummary:
    """What the reference's synthesizer wrote, and what it cost."""

    text: str
    synthesizer: str
    model_calls: int
    #: The reference's summary names no passage per sentence, so nothing in it can be checked the way ours is.
    cites: bool = False
    failed: Optional[str] = None


async def llamaindex_summary(model: ChatModel, question: str, passages: Sequence[str]) -> ReferenceSummary:
    """The reference's ``TreeSummarize`` over ``passages`` with our model.

    A model that fails is reported, not raised, so one failed item does not
    end a run; the failure is on the item's record."""
    from llama_index.core.response_synthesizers import TreeSummarize

    llm = _llm_class()(model)
    synthesizer = TreeSummarize(llm=llm, use_async=True)
    try:
        text = await synthesizer.aget_response(question, list(passages))
    except Exception as error:
        return ReferenceSummary("", "TreeSummarize", llm.calls, failed=type(error).__name__)
    return ReferenceSummary(str(text), "TreeSummarize", llm.calls)


def _documents(item: BenchItem) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for index, session in enumerate(item.sessions):
        text = "\n".join(session)
        if text.strip() and index < len(item.session_ids):
            kept.append((item.session_ids[index], text))
    return kept


def reference_splitter(embedder: Embedder, *, chunk_size: int, chunk_overlap: int) -> Any:
    """The reference's ``SentenceSplitter``, counting the tokens the embedding
    model reads where the model can count them.

    LlamaIndex counts tiktoken's tokens by default, and a model such as BGE
    reads more of its own for the same text and drops the tokens past its
    window: at 512 tiktoken tokens, 297 of the reference's 729 nodes over
    three LongMemEval-S items ran past BGE's 512. So with a model that counts
    (``count_tokens``), the splitter counts with it, and a chunk's start and
    end markers (the count of an empty text) are paid once a chunk, as the
    engine's token chunks count them: ``chunk_size`` is what the model reads.
    An embedder that cannot count, the hashed one, keeps LlamaIndex's default."""
    from llama_index.core.node_parser import SentenceSplitter

    counter = getattr(embedder, "count_tokens", None)
    if not callable(counter):
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    markers = counter("")

    def tokens(text: str) -> range:
        # The splitter sums its pieces' counts, so each is counted without the markers.
        return range(max(0, counter(text) - markers))

    return SentenceSplitter(chunk_size=chunk_size - markers, chunk_overlap=chunk_overlap, tokenizer=tokens)


def reference_tokenizer(embedder: Embedder) -> str:
    """What counts the reference's chunk tokens, as ``reference_splitter`` chooses."""
    if callable(getattr(embedder, "count_tokens", None)):
        return f"{embedder.id}, a chunk's markers included"
    return "tiktoken gpt-3.5-turbo (LlamaIndex's default)"


async def llamaindex_session_ranking(item: BenchItem, embedder: Embedder, *, k: int,
                                     chunk_size: int = 512, chunk_overlap: int = 0, hybrid: bool = False) -> list[str]:
    """Sessions in the order LlamaIndex ranks their nodes for the item's
    question, at most ``k`` distinct: its default vector retrieval, or with
    ``hybrid`` its own BM25 retriever fused with the vector retriever by
    reciprocal rank -- the reference at its best rather than its default."""
    from llama_index.core import Document, VectorStoreIndex

    if hybrid:
        try:
            from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover - depends on the optional package
            raise ImportError("the hybrid reference needs llama-index-retrievers-bm25; install it beside llama-index-core") from None
        from llama_index.core.llms import MockLLM
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    documents = [Document(text=text, metadata={"session_id": session_id},
                          excluded_embed_metadata_keys=["session_id"], excluded_llm_metadata_keys=["session_id"])
                 for session_id, text in _documents(item)]
    if not documents:
        return []
    adapter = SconeEmbedding(embedder)
    splitter = reference_splitter(embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def build_and_retrieve() -> list[Any]:
        index = VectorStoreIndex.from_documents(documents, embed_model=adapter, show_progress=False,
                                                transformations=[splitter])
        # Every node is ranked, so a session that splits into many nodes
        # cannot crowd the others out of a fixed top-N before the folding.
        nodes_in_index = max(1, len(index.docstore.docs))
        vector = index.as_retriever(similarity_top_k=nodes_in_index)
        if not hybrid:
            return list(vector.retrieve(item.question))
        lexical = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=nodes_in_index)
        # One query, no LLM: the fusion here is the reference's rank fusion
        # of its two retrievers, not its model-written query variants.
        # A mock model satisfies the fusion retriever's constructor; with one
        # query it is never asked anything, and no hosted model is touched.
        fused = QueryFusionRetriever([vector, lexical], llm=MockLLM(), similarity_top_k=nodes_in_index, num_queries=1,
                                     mode=FUSION_MODES.RECIPROCAL_RANK, use_async=False, verbose=False)
        return list(fused.retrieve(item.question))

    nodes = await asyncio.to_thread(build_and_retrieve)
    ranked: list[str] = []
    for node in nodes:
        session_id = node.node.metadata.get("session_id")
        if isinstance(session_id, str) and session_id not in ranked:
            ranked.append(session_id)
            if len(ranked) == k:
                break
    return ranked


@dataclass(frozen=True)
class SideScores:
    """One side's numbers at every k: whether any answer session was in
    the top k, whether all were, the mean reciprocal rank, and -- the
    reference framework's own retrieval measures -- precision at k (the
    share of the top k that answer) and NDCG at k (the answers' places,
    discounted by rank)."""

    recall_any: dict[int, float]
    recall_all: dict[int, float]
    mrr: float
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemComparison:
    question_id: str
    question_type: str
    answer_sessions: tuple[str, ...]
    scone: tuple[str, ...]
    llamaindex: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    n: int
    excluded_without_evidence: int
    sides: dict[str, SideScores]
    delta: SideScores
    items: tuple[ItemComparison, ...]
    config: dict[str, Any] = field(default_factory=dict)

    def per_item_at(self, k: int) -> dict[str, int]:
        """At ``k``: items where only we found an answer session (wins), only
        LlamaIndex did (losses), or both or neither did (ties)."""
        wins = losses = ties = 0
        for item in self.items:
            ours = any(session in item.answer_sessions for session in item.scone[:k])
            theirs = any(session in item.answer_sessions for session in item.llamaindex[:k])
            if ours and not theirs:
                wins += 1
            elif theirs and not ours:
                losses += 1
            else:
                ties += 1
        return {"k": k, "wins": wins, "losses": losses, "ties": ties}

    def record(self) -> dict[str, Any]:
        def side(scores: SideScores) -> dict[str, Any]:
            return {"recall_any": {str(k): v for k, v in scores.recall_any.items()},
                    "recall_all": {str(k): v for k, v in scores.recall_all.items()}, "mrr": scores.mrr,
                    "precision": {str(k): v for k, v in scores.precision.items()},
                    "ndcg": {str(k): v for k, v in scores.ndcg.items()}}

        ks = sorted(self.delta.recall_any)
        return {"protocol": "comparative-retrieval-v1", "n": self.n,
                "excluded_without_evidence": self.excluded_without_evidence,
                "sides": {name: side(scores) for name, scores in self.sides.items()}, "delta": side(self.delta),
                "per_item": [self.per_item_at(k) for k in ks],
                "items": [{"question_id": item.question_id, "question_type": item.question_type,
                           "answer_sessions": list(item.answer_sessions), "scone": list(item.scone),
                           "llamaindex": list(item.llamaindex)} for item in self.items],
                "config": self.config}


def _scores(rankings: Sequence[tuple[Sequence[str], set[str]]], ks: Sequence[int]) -> SideScores:
    if not rankings:
        return SideScores({k: 0.0 for k in ks}, {k: 0.0 for k in ks}, 0.0, {k: 0.0 for k in ks}, {k: 0.0 for k in ks})
    recall_any = {k: round(bench_metrics.hit_rate(rankings, k), 4) for k in ks}
    recall_all = {k: round(sum(1.0 for retrieved, relevant in rankings if relevant <= set(retrieved[:k])) / len(rankings), 4)
                  for k in ks}
    precision = {k: round(sum(bench_metrics.precision_at(retrieved, relevant, k) for retrieved, relevant in rankings)
                          / len(rankings), 4) for k in ks}
    ndcg = {k: round(sum(bench_metrics.ndcg_at(retrieved, relevant, k) for retrieved, relevant in rankings)
                     / len(rankings), 4) for k in ks}
    return SideScores(recall_any, recall_all, round(bench_metrics.mrr(rankings), 4), precision, ndcg)


def distinct_sessions(ranked: Sequence[str], k: int) -> tuple[str, ...]:
    """The first ``k`` distinct sessions of a ranking of passages, in
    the order their first passage held: a session cut into many passages
    is one session in the ranking, as it is on the reference's side."""
    return tuple(dict.fromkeys(session for session in ranked if session))[:k]


def side_delta(ours: SideScores, theirs: SideScores) -> SideScores:
    """Ours minus theirs, at every k for recall, precision and NDCG, and
    for MRR; positive means we did better. Two sides scored at different
    ks, or one with precision and NDCG and one without, are refused: a
    missing number is not a zero."""
    if set(ours.recall_any) != set(theirs.recall_any):
        raise ValueError("both sides must be scored at the same ks")
    for name in ("precision", "ndcg"):
        if set(getattr(ours, name)) != set(getattr(theirs, name)):
            raise ValueError(f"both sides must carry {name} at the same ks, or neither")
    return SideScores({k: round(ours.recall_any[k] - theirs.recall_any[k], 4) for k in ours.recall_any},
                      {k: round(ours.recall_all[k] - theirs.recall_all[k], 4) for k in ours.recall_all},
                      round(ours.mrr - theirs.mrr, 4),
                      {k: round(ours.precision[k] - theirs.precision[k], 4) for k in ours.precision},
                      {k: round(ours.ndcg[k] - theirs.ndcg[k], 4) for k in ours.ndcg})


async def compare(items: Iterable[BenchItem], make_engine: Callable[[], Any], embedder: Embedder, *,
                  ks: Sequence[int] = (5, 10, 15), chunk_size: int = 512, chunk_overlap: int = 0,
                  hybrid: bool = False) -> Comparison:
    """Both sides over the same items, scored by the same rule; ``hybrid``
    runs the reference with its BM25 retriever fused in, its best configuration."""
    import llama_index.core

    items = list(items)
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("ks must be positive integers")
    k_max = max(ks)
    # Our recall returns passages, up to PER_EPISODE_CAP of one session;
    # asked for that many times k, it can reach k distinct sessions, and
    # the ranking is folded to sessions before it is scored -- the same
    # rule the reference's nodes are folded by.
    from ..retrieval.fusion import PER_EPISODE_CAP

    recall_limit = k_max * PER_EPISODE_CAP
    ours = await run(make_engine, items, ks=tuple(ks), limit=recall_limit)
    by_id: dict[str, ItemResult] = {result.question_id: result for result in ours.results}
    compared: list[ItemComparison] = []
    excluded = 0
    for item in items:
        if not item.has_evidence:
            excluded += 1
            continue
        theirs = await llamaindex_session_ranking(item, embedder, k=k_max, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                  hybrid=hybrid)
        mine = by_id.get(item.question_id)
        compared.append(ItemComparison(item.question_id, item.question_type, tuple(item.answer_session_ids),
                                       distinct_sessions(mine.retrieved_sessions, k_max) if mine else (), tuple(theirs)))
    scone = _scores([(item.scone, set(item.answer_sessions)) for item in compared], ks)
    llama = _scores([(item.llamaindex, set(item.answer_sessions)) for item in compared], ks)
    delta = side_delta(scone, llama)
    config: dict[str, Any] = {
        "embedder": embedder.id, "llm": None, "reranker": None, "ks": list(ks),
        "scone": {"recall_limit": recall_limit, "retrieval": "engine defaults: vector + lexical lanes, reciprocal rank fusion",
                  "sessions_folded_from_passages": True},
        "llamaindex": {"version": llama_index.core.__version__, "index": "VectorStoreIndex",
                       "retriever": ("QueryFusionRetriever(VectorIndexRetriever + BM25Retriever)" if hybrid
                                     else "VectorIndexRetriever"),
                       "fusion": "reciprocal_rerank" if hybrid else None,
                       "splitter": "SentenceSplitter", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                       "tokenizer": reference_tokenizer(embedder),
                       "similarity_top_k": "every node in the item's index", "sessions_folded_from_nodes": True},
    }
    return Comparison(len(compared), excluded, {"scone": scone, "llamaindex": llama}, delta, tuple(compared), config)
