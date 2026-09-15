# The question lane: a chunk found by the questions it answers

A passage is written in its author's words and asked about in the
reader's. "Is there a winter mooring ban?" shares no word with "The
harbour closes to sailing boats every November", so the text lane cannot
find the answer and a hashed vector barely can. The question lane asks a
chat model, once per chunk, for the questions the chunk answers, keeps
only those it can check, and indexes them beside the chunk so a query
phrased like one of them finds the chunk.

It is off by default, and it costs one model call per chunk.

## What is kept

Each reply is a list of `{"question", "quote"}` pairs, where the quote is
the sentence of the chunk that answers the question. A pair is kept only
when:

- the question is not empty and at most 400 characters;
- the quote is at least four words and stands in the chunk verbatim,
  whitespace aside (the same anchoring `bench.questions` uses for its
  measurement sets, `bench.questions.anchored`);
- it is among the first `per_chunk` pairs of the reply;
- the chunk has not already had the same question (case and whitespace
  aside) in this reply.

Everything else is dropped and counted in the report: `dropped_unquoted`,
`dropped_unasked`, `dropped_extra`, `dropped_repeated`, and
`dropped_unparsed` for a reply that is not such a list. A call that fails
is counted in `calls_failed` and the pass goes on to the next chunk.

A small model often writes every object of the list and stops before the
closing bracket (llama3.2-ctx8k did on the documents measured below). Such
a reply is read object by object up to the first that is not whole
(`bench.questions.partial_pairs`) and counted in `read_partial`; each pair
read that way still has to pass every check above. The bench's own
question sets keep the strict reader.

## Where the questions go

Into the context index (`core.ports.ContextIndex`), the one the context
lane searches — never into the chunk's text, its vectors or its offsets.
The passage a recall returns is always the chunk's own text. The index
keeps one text per chunk and replaces it on write, so when the context
lane is also on, the words the chunk is under (headings, title, source
name) are derived again and written back beside the questions. A chunk
with no kept question is not written at all, so its context words stay
as ingestion left them.

At recall the index is searched when `SCONE_QUESTION_LANE` or
`SCONE_CONTEXT_LANE` is on, with the query the text lane got, and fused
by rank at the context lane's weight (`retrieval.recall.CONTEXT_WEIGHT`,
2.0). An item the lane placed says so in `lanes.context`; the recall
event carries `question_lane` beside `context_lane`. Because the two
lanes share one index, a space whose questions were written stays
searchable through them while `SCONE_CONTEXT_LANE` is on even if
`SCONE_QUESTION_LANE` is later turned off; forget the episodes, or write
the lane again, to take them out.

This differs from the references on purpose. RAGFlow writes questions
per chunk at ingestion, searches them at many times the content's weight
and, when a chunk has questions, embeds the questions instead of the
content. LlamaIndex's `QuestionsAnsweredExtractor` adds them to the text a
node is embedded from. Both take the model's questions on trust; here a
question without its quote is not kept, and the lane is lexical only.

## Running it

The pass runs by hand, over chunks in id order, and refuses while the
lane is off.

```python
engine = await MemoryEngine(documents, vectors, embedder, question_lane=True).open()
report = await engine.build_chunk_questions("space", chat, per_chunk=3)
print(report.text())
```

```sh
SCONE_QUESTION_LANE=1 SCONE_CHAT_URL=http://127.0.0.1:11434/v1 SCONE_CHAT_MODEL=llama3.2-ctx8k \
  scone --space notes chunk-questions --per-chunk 3
```

```http
POST /v1/chunk-questions?per_chunk=3&max_chunks=2000&after_chunk=120&episode_id=7
```

The route uses the synthesis model and answers 501 without one; the
command needs `SCONE_CHAT_URL` and `SCONE_CHAT_MODEL`. Both return the
report (`--json` for the command).

## Bounds, and what the report says when they cut

| Bound | Default | When it cuts |
|---|---|---|
| `per_chunk` | 3, at most 5 | pairs past it are `dropped_extra` |
| `max_chunks` | 2000, at most 2000 | `chunks_cut` counts the chunks not looked at and `resume_after` names the chunk id to pass as `after_chunk` next |
| `MAX_CHUNK_BYTES` | 8000 | a longer chunk is not shown to the model and is counted in `skipped_long` |

`episode_ids` (`--episode`, `episode_id`) limits a pass to named
episodes. A store without the context index gets no model calls: the
report has `kept_lane: false` and a reason, and a recall with the lane on
over that store says `question lane: not kept by …` in `degraded` and
answers from the other lanes. The SQLite and in-memory stores keep it.

## Forgetting

The questions live in the chunk's row of the context index, which goes
with the chunk: SQLite by cascade (and its full-text index by trigger),
the in-memory store when it deletes the episode's chunks. Forgetting an
episode leaves no question that can find anything.

## Measured

See `benchmarks/chunk-question-lane-v1.results.md`.
