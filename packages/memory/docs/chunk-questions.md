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

The list read is the first in the reply that opens with an object, so a
bracket in the prose around it is not the list, and neither is a `[]`
quoted inside one of its questions. A reply with no object in it that
says `[]` has no questions.

A small model often writes a list that is not valid JSON as a whole: it
stops before the closing bracket (llama3.2-ctx8k often did on the
documents measured below), leaves out the commas, or copies a quote mark
from the passage without escaping it. Such a reply is read object by
object (`bench.questions.loose_pairs`) and counted in `read_loosely`: an
object that does not decode, or an item that is not a question and a
quote, is passed over to the next object and counted in
`dropped_unread`, and reading stops at the closing bracket or the end of
the reply. Each pair read that way still has to pass every check above.
The bench's own question sets keep the strict reader.

## Where the questions go

Into a question index of their own (`core.ports.QuestionIndex`) — never
into the chunk's text, its vectors, its offsets or the context index
that holds the words the chunk is under. The passage a recall returns is
always the chunk's own text, and the context lane's words are exactly as
ingestion left them, whichever lanes the engine running the pass has on.

At recall the question index is searched only when `SCONE_QUESTION_LANE`
is on, with the query the text lane got, and fused by rank at
`retrieval.recall.QUESTION_WEIGHT` (2.0, the context lane's). An item
the lane placed says so in `lanes.question`; the recall event carries
`question_lane` beside `context_lane`, and each says whether its own
index was searched. Turning `SCONE_QUESTION_LANE` off stops the
questions ranking anything; they stay stored until their episode is
forgotten.

Writing the lane again replaces a chunk's questions with those the new
pass kept, whenever the chunk's reply was read — none included: a chunk
whose reply kept no question holds none afterwards and is counted in
`chunks_kept_none`. A chunk whose call failed, whose reply could not be
read, or that was too long to show keeps what an earlier pass wrote.

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
episodes. A store without the question index gets no model calls: the
report has `kept_lane: false` and a reason, and a recall with the lane on
over that store says `question lane: not kept by …` in `degraded` and
answers from the other lanes. The SQLite and in-memory stores keep it.

## Forgetting

The questions live in the chunk's row of the question index, which goes
with the chunk: SQLite by cascade (and its full-text index by trigger),
the in-memory store when it deletes the episode's chunks or the space.
The pass reads the chunks up front and then waits on the model for each;
the store writes a chunk's questions only if the chunk is still there
when the reply comes back, so an episode forgotten or replaced meanwhile
gets none, and the report counts those chunks in `chunks_gone`.
Forgetting an episode leaves no question that can find anything.

## Measured

On 13 of this repository's documents (233 chunks), with llama3.2-ctx8k
writing both the lane and, by a different prompt, 43 evaluation questions,
the lane made recall worse: R@5 0.953 off and 0.837 on, MRR 0.868 and
0.817. The questions that moved down were overtaken by chunks the lane
placed high for questions about something else; a fusion weight low enough not to
do that (0.01) only ties the lane off on held-out questions. Writing the
lane cost 100 model calls and about 1,090 s per 100 chunks. The
questions were then kept in the context index; moved into their own
index, the same questions give the same scores for every question and
every weight. The
evaluation questions share the chunk's words, which is not the case the
lane is for, so this is not a verdict on questions asked in other words.
Details, the weight sweep and the caveats:
[`benchmarks/chunk-question-lane-v1.results.md`](../benchmarks/chunk-question-lane-v1.results.md).
