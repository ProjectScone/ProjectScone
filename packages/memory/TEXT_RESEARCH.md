# Temporary text research

`scone_memory.text_network` exports two pure, bounded helpers in addition to
the unchanged `build_text_network` version 1 API. Callers supply and authorize
the text. Neither helper reads stores, persists results, or calls a model.

`build_section_coverage(documents, terms, sections, *, limits=None,
should_cancel=None)` accepts `TextDocument`, `TextTerm`, and `TextSection`
values. Each section has `key`, `document_key`, `title`, `start`, and `end`.
Offsets are Unicode codepoints in the original, unmodified source. Section
keys are unique; sections must stay within their document, must not overlap,
and must not split a scanned prose passage. Sections may cover only part of a
document. Applications decide what their sections mean.

The `SectionCoverage` result includes document fingerprints, term IDs,
section counts, and scan coverage. `network_digest` and term IDs match the
version 1 network built from the same documents and terms. The result's own
`digest` also binds the selected sections and scan outcome. Counts measure
matched **passages**, not the number of mentions inside each passage. They
are calculated from raw text before graph citation sampling. A section's
`complete` flag distinguishes a complete scan from one stopped by the passage
budget. Missing sparse counts mean zero only in a complete section.

Each nonzero count includes `first_passage_start` and `first_passage_end` for
opening context directly, including passages absent from the network's
sampled citations. `first_start`/`first_end` and `last_start`/`last_end` locate
the canonical literal match in the first and last matching passages. They
are not an enumeration of every occurrence.

`surrounding_context(documents, requests, *, limits=None, should_cancel=None)`
accepts `ContextRequest` values with a unique `key`, `document_key`, expected
`content_sha256`, whole-passage `start`/`end`, and optional `before`/`after`
neighbor counts (default 1; each 0–3). Wrong fingerprints, arbitrary substring
ranges, code-block targets, and unverified targets beyond the scan budget are
rejected. Windows retain source identity, target bounds, exact returned text
bounds, and explicit truncation flags. Source gaps between neighboring prose
passages remain unmodified; windows are source excerpts, not rewritten prose.

`ResearchLimits` bounds input to 15 documents, 60 terms, 512 KiB of UTF-8 text,
4,000 scanned prose passages, and 512 KiB of serialized output. It also bounds
sections to 128, context requests to 10, and each context window to 12,000
Unicode codepoints. Callers may lower these limits. Excess sections and
requests are rejected rather than silently dropped. Both helpers check
`should_cancel` before processing, while scanning/matching, and before
returning. Cancellation raises `TextNetworkCancelled` without a partial result.
