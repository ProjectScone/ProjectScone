# Document duplicate review

Scone can inspect incoming text or a stored episode for copied passages and
possible paraphrases. Original documents remain unchanged. Existing exact
episode and attachment deduplication still applies independently.

```python
from scone_memory.deduplication import (
    DeduplicationConfig, DocumentRevision, MemoryDuplicateInspector,
    project_document, review_notification,
)

# Reuse this inspector with an already-open, explicitly configured MemoryEngine.
inspector = MemoryDuplicateInspector(memory, DeduplicationConfig(
    overlap_threshold=0.25,
    semantic_enabled=True,
))
document = DocumentRevision("my_space", "manual.pdf", "revision-2", extracted_text)
report = await inspector.inspect(document)
notification = review_notification(report)
if notification is not None:
    # Return or persist this typed notification in your application's review queue.
    print(notification.message, notification.sources, notification.available_actions)

# Preserve all search content until a reviewer chooses a different action.
projection = project_document(document, report)
# Explicit alternatives:
projection = project_document(document, report, action="suppress_copied_passages")
projection = project_document(document, report, action="exclude_document")
```

`project_document` returns a search projection, not a database mutation. An
application's index writer applies the selected projection to its search index;
existing `MemoryEngine` indexes are not silently rewritten. Persist the trusted
report and decision in the application's review queue before applying a change.
Do not accept a client-constructed report as authoritative evidence.

Each retained passage has text and half-open UTF-8 byte offsets into the original.
Index these passages separately: joining disconnected ranges would invent a
contiguous quotation and break citations. Whole-document exclusion returns no
retained search passages but never deletes the original. Reports are bound to
scope, source, revision, content digest, and byte offset; stale reports cannot be
applied to modified documents.

## What the report means

- Copied coverage is the union of matching incoming byte ranges divided by its
  UTF-8 byte length. Overlapping matches count once. At least 25% triggers review
  by default; configure another threshold as needed.
- Matching preserves Unicode source offsets and recognizes normalized whitespace
  and case. Short common phrases are bounded by `min_match_chars`, default 24.
- Semantic matches contain cosine scores and the embedder identity. The default
  threshold is 0.90. These are review candidates, not copied percentages or proof
  that the passages say the same thing. Semantic matches alone never identify
  bytes safe to suppress.
- Matches include source episode/chunk IDs and source ranges. Native candidates
  are checked against their retained episode content and requested memory space.
- A native search uses bounded lexical and vector candidates. `complete=False`
  means the report cannot establish that every duplicate in the corpus was found.
  Absence of a match is not proof of originality.

Inspect an existing document with
`await inspector.inspect_episode("my_space", episode_id)`. It excludes its own
source revision from matches. Native APIs require the caller to authorize the
requested space, just like other direct `MemoryEngine` operations.

## Latency and provider configuration

Semantic inspection defaults on and reuses the engine's configured embedder and
stored vector index. It embeds only incoming passages, batches their embeddings,
and caches query vectors in a bounded LRU. Every inspection searches candidates
again so forgotten sources or changed access rights are not cached search results.
Use `inspector.clear_cache(scope="my_space")` when disposing of that scope.

Set `semantic_enabled=False` to avoid both embedding and vector-search calls.
An absent semantic model or `HashEmbedder` reports `semantic_status="unavailable"`;
hash-vector overlap is not presented as paraphrase understanding. No provider is
downloaded or connected implicitly by this feature.

Defaults bound input to 256 KiB, candidates to 32, aggregate candidate bytes to
2 MiB, and semantic input to 32 passages. Larger inputs fail the document limit
explicitly; candidate, passage, and matching-work truncations appear in report
limitations. Applications can tune these budgets, but larger budgets cost work.
Backend lexical tokenization and approximate vector search affect candidate
coverage, especially for languages the configured text index does not support.

`report.metrics` records total, lexical, embedding and semantic search milliseconds,
candidate counts, embedded passages, and cache hits. Measure these on your actual
corpus and hardware; tiny integration fixtures do not establish retrieval accuracy
or production throughput.

Custom stores can implement `CandidateProvider` and use
`DocumentDuplicateDetector` directly. Providers must authorize and scope all
candidates, return genuine cosine scores from the named embedder, and disclose
partial candidate coverage. They should query existing indexes rather than scan
or re-embed the corpus for each document.
