# Passage continuity development comparison — 8 September 2026

An optional `neighbor_chunks=1` reads the adjacent stored chunks around ranked
passages. Ranked anchors retain priority, every source keeps its original ID
and text, and the complete source block stays within 8,000 bytes. This follows
the parent/child retrieval idea in the RAGFlow reference, using Scone's existing
verified passage reader without importing RAGFlow.

The comparison prepared **400 contexts**: the same 200 original development
questions from the [public QA baseline](public-qa-v1.results.md), each with the
option disabled and enabled. It used a copied SQLite ledger and copied Qdrant
1.19.1 storage, cached `bge-small-en-v1.5`, five ranked anchors and otherwise
unchanged preparation settings. No generation model ran in this comparison.

| Metric | Disabled | One neighboring chunk on each side |
|---|---:|---:|
| Hotpot: all annotated sentences present | 72/100 | 74/100 |
| Hotpot: mean annotated-sentence coverage | 86.8% | 87.5% |
| SQuAD: reference answer present within its source | 99/100 | 100/100 |
| Hotpot: median preparation latency | 26.51 ms | 26.88 ms |
| SQuAD: median preparation latency | 22.38 ms | 22.99 ms |
| Hotpot: median context bytes | 3,363.5 | 3,675 |
| SQuAD: median context bytes | 3,758.5 | 5,015.5 |

No question lost annotated evidence. All 400 contexts stayed within the byte
budget, and every enabled window preparation reported success. The disabled
configuration reproduced all 200 frozen baseline source sets exactly. The
reader validates retained chunk text and source scope before inclusion; the
experiment also checked supplied passages against the original corpus.

This addresses one observed passage-boundary failure and two Hotpot evidence
gaps. It does not repair missing supporting documents or prove improved answer
accuracy. These are already-inspected development questions; the additional
200 reserved questions remain unrun. The timing comparison always prepared
the disabled configuration first, so cache/order effects and ordinary machine
activity prevent attributing the small latency difference solely to expansion.

Original questions, gold references and the original 600 model responses were
not changed. The preparation script does not read gold labels. Coverage was
computed afterward. Separate requests, receipts, code hashes and results are
retained under the ignored `bench-runs/passage-window-dev-2026-09-08/` directory.
The context artifact SHA256 is
`34c3ccc8493141e9a6f3dabcf70e037750f15e395d1c25e500b7e52ed608935b`.
The context builder and window reader match their recorded experiment hashes;
subsequent `TextConversation` edits only add state type annotations.
The option remains disabled by default pending generation-quality evaluation.
