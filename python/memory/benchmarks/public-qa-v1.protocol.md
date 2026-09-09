# Public QA benchmark v1 — preregistered 8 September 2026

Purpose: measure the quality, latency, and memory tradeoff on real source text,
without selecting questions or changing prompts after observing model answers.
This protocol is fixed before its first model response. Changes require a new
run/protocol and must preserve the old observations.

## Data and sampling

- HotpotQA development distractor v1: human-authored multi-hop questions over
  Wikipedia paragraphs. Community mirror supplied by the user:
  https://huggingface.co/datasets/namlh2004/hotpotqa/blob/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json
  Mirror SHA256: e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8.
  Official CMU endpoint failed TLS; do not claim canonical-byte verification.
- SQuAD 1.1 development: human-authored answerable questions over Wikipedia.
  https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
- Preserve downloaded bytes and record their checksums, URLs, counts, and license
  attribution. Do not commit downloaded corpora or generated outputs.
- Within each dataset rank question IDs by SHA256 of
  `scone-public-qa-v1:20260908:` followed by the namespaced ID. First100 are
  evaluation questions; next100 are reserved, unrun questions. Do not select by
  difficulty, question/answer content, or preliminary model performance.
- Shared corpus: all paragraphs supplied with these400 questions, including all
  Hotpot distractor paragraphs, deduplicated by original title+text. Reserved
  question texts/answers are excluded from inference. This is a question-ID
  holdout, not a guarantee of disjoint source articles/topics.
- Index source title+text only. No questions, answers, supporting-fact labels,
  handcrafted facts, per-question gold filters, or generated replacement text.
  Retain support labels separately for scoring.

## Frozen engine and model settings

- Python3.14; actual self-managed Qdrant1.19.1 server, SQLite ledger,
  cached bge-small-en-v1.5 embeddings. No cloud inference or browser retrieval.
- Existing MemoryContext, namespace `public-qa-v1`, no source/project filter,
  limit5, context8000bytes, recall deadline30s, structured paths enabled with
  existing defaults. Raw ingestion only: no manually seeded fact graph.
- Additional standalone native recall at limit10 for document recall@5/@10.
  Actual generation-input coverage is measured separately after context packing.
- Freeze and hash all200 prepared requests before generation. All models use
  exactly those requests; no gold-source control is included in headline scores.
- Models: gemma4-e4b-ctx8k:latest, llama3.2-ctx8k:latest,
  llama3.1-ctx8k:latest. Record installed digests and parameters;8192context,
  temperature0, reasoning_effort=none, max_output_tokens256, deadline120s.
- One response per model/question:600 scored responses. Blocks of20 questions;
  rotate model order by block number. No simultaneous inference requests.
  Record cold/block-first observations separately; do not assume isolated timing.
- Use DEFAULT_SYSTEM_PROMPT plus this fixed response-format instruction:
  `Answer the latest question with only the shortest complete answer (a name,
  date, number, phrase, or yes/no). Do not add explanation, citations, or source
  IDs. If the supplied memory does not contain sufficient evidence, answer
  exactly INSUFFICIENT_EVIDENCE.` Question text remains unchanged.
- Gold files are consumed only by the separate scorer. Inference reads corpus,
  query and prepared-request files only. Source code/manifest hashes are checked
  before and after the run; substantive changes invalidate pooled comparison.

## Scoring and reporting

- Answer exact match and token F1 with standard lowercase, ASCII-punctuation,
  article and whitespace normalization. SQuAD takes the maximum over its
  published reference answers. Hotpot preserves its special yes/no/noanswer
  mismatch rule. No answer rewriting, answer extraction, or LLM-judge correction.
- Failures, interrupted/unfinished replies, and timeouts score0 and remain in
  the denominator. Keep raw public output and terminal status. Abstentions are
  separate, and score0 on these answerable sets. Never replace a failed response
  with a retry. An interrupted experiment retains started rows as failures.
- Report support-document recall@5/@10; Hotpot annotated supporting-sentence
  coverage in actual requests; SQuAD answer-span presence within its source.
  These are annotation coverage metrics, not proof of semantic faithfulness.
- Report dataset-specific and overall scores, sample counts, p50/p95 elapsed and
  first-token latency, completion/abstention/failure rates, and observed resident
  model memory. Preserve every request, response, score, and retrieval receipt.
- Record limitations: pooled paragraph corpus is not full-Wikipedia retrieval;
  public data may have appeared in model training; single samples do not measure
  repeat variance; answer EM/F1 are not full semantic correctness; source-citation
  quality and unanswerable-question handling are not evaluated by this protocol.
- No tuning during the600-response run. Later diagnosis/tuning is separately
  labeled; its changes must be evaluated on the reserved questions.

## Measurement details fixed before inference

- Recall@k means the fraction of annotated supporting source documents found
  within the first k ranked native-recall chunks (k=5,10). Deduplicate their
  document IDs for the numerator; do not refill the list to k unique documents.
  Document identity is SHA256 of original title, NUL, and original paragraph
  text. Average per-question fractions; also report all-support-documents found.
- Generation timers cover inference only, including model loading and prompt
  processing when incurred. Record MemoryContext preparation separately. Their
  sum is an estimated sequential request cost, not an observed live HTTP latency.
- Memory collector: Ollama `/api/ps` before and after each20-question model block.
  Save every loaded model's `size` and `size_vram` in bytes, its digest and context.
  These are Ollama-reported loaded-model sizes, not process RSS or peak system
  RAM. Before a block, unload other models in this benchmark's three-model set
  using native keep_alive=0. Do not unload unrelated models; record them.
- The manifest contains all600 planned observations. Interrupted started calls
  are terminal failures; unattempted calls remain explicitly unattempted until
  resumed. Do not rank models from different completed prefixes or call an
  incomplete run complete. Full-run scoring requires every planned observation
  to be terminal, with failures retained in each model's200-question denominator.
- The reserved questions are evaluated only after tuned settings are frozen.
  Once their results influence tuning, they are no longer an untouched holdout.
