# Follow-up queries: a turn searched with what the conversation named

Memory is searched for the latest user message. In a conversation that
is often not enough: after "Where does Alice Chen work?", the turn
"since when?" names nothing, and a search for it finds whatever says
"since", about anyone. `retrieval/followup.py` searches such a turn a
second time, with what the earlier turn named, and says what it did.
It is off unless asked for.

```python
from scone_memory.integrations.chat import recall_context
from scone_memory.realtime.context import MemoryContext

messages = [{"role": "user", "content": "Where does Alice Chen work?"},
            {"role": "assistant", "content": "At Acme Robotics."},
            {"role": "user", "content": "since when?"}]

prepared, receipt = await recall_context(engine, "notes", messages, followup="carry")
receipt.followup["carried"]      # ["Alice Chen"]
receipt.followup["from_message"] # 0
receipt.followup["query"]        # "Alice Chen since when?"

context = MemoryContext(engine, "notes", session_id, followup_queries="carry")
request, receipt = await context.prepare(messages)
receipt["followup"]              # the same block
```

Served text conversations take it from the environment:
`SCONE_FOLLOWUP_QUERIES=off | carry | rewrite` (default `off`), on both
`scone serve` and `scone serve-conversations --model-factory`.
`serve-conversations` without a model factory refuses the setting at
startup: history-only serving searches no memory, so it would do nothing.
Voice sessions and `pipeline.memory.MemoryStage` do not take it; they
search the latest question alone.

## carry: no model

A latest user turn **leans on the conversation** when any of these holds,
and the receipt's `cues` lists which:

- it holds one of the phrases "since when", "what about", "how about",
  or ends in "too" or "as well" ("Globex too?");
- it names nothing (see below) and
  - refers back: it, its, they, them, their, he, him, his, she, her,
    that, this, those, these, there, then;
  - or is three words or fewer (`short`);
  - or has fewer than two telling words of its own — words of four
    letters or more that are not common words (`names nothing`).

A turn that names something is taken to be about what it names, and
only the phrases above make it lean. "Is there a meeting with Bob Smith
on Friday?" and "When did Kestrel Bank open its Leeds branch?" hold a
referring word, but it points into the question, so they are searched
as asked and the reason lists the names ("standalone: the question names
Bob Smith, Friday; ..."). The cost: "Is it open on Sundays?" names
"Sundays", so it is not carried either.

Then the **named or rare terms** of the most recent earlier *user* turn
that has any are carried forward verbatim — never the assistant's words,
which the person did not say — and a second query is searched: those
terms, then the question. A named or rare term is read from the text as
written: a run of capitalised words that are not common words (so a
sentence may open with a name, but not with "Where"), a quoted phrase,
or a word shaped like an identifier (a digit, a symbol inside it as in
`node.js`, or a capital after its first letter as in `iPhone`). "Rare"
is a shape, not a count over the corpus; no store is read. A listed
word that opens a sentence capitalised is not a name and does not join
the name after it: verbs of asking (Explain, Remind, Summarize,
Describe, Compare, Check, Recall ...), discourse adverbs (Actually,
Maybe ...) and Yesterday, Today, Tomorrow, Tonight. So "Actually,
Summarize Alice Chen's role." names `Alice Chen`. Shape cannot tell
"Globex hired Carol" from "Explain what Carol does", so an opening verb
missing from that list is still read as a name.

Nothing is carried, and the record says why, for a first turn, a
standalone question, earlier turns that name nothing, terms the question
already names, and a carried query over the 1,000-character query bound.
A term counts as already named only as whole words, whatever the case
and spacing: "Ann" is carried into "And since when has she run Annex?",
and "Io" into "since when is the ratio tracked?".

## rewrite: a model, with a fallback

`followup="rewrite"` with a `followup_model` (any `ChatModel`) asks the
model, in the pattern of [query rewrite](query-rewrite.md), to restate
the turn as a standalone question from the conversation's user and
assistant turns. The restatement is taken only when it reads as a JSON
query, is within the query bound, and shares a word with the question
or the history it was shown; a restatement equal to the question means
the model found it standalone. When the model fails, passes its
deadline (`followup_timeout`, 5 s by default, at most 60), or returns
what cannot be trusted, **carry** runs instead — or the question is
searched as asked — and `fallback` names the cause. A first turn asks no
model.

## Fusion

The question as asked is always searched. The follow-up's list is fused
with it **by rank**: interleaved, the question's first, each passage
once, cut at the recall limit. The question's best passage keeps the
lead; the follow-up's best is second. Facts are interleaved the same way
and cut to as many as the longer of the two recalls gave, so a carried
query never doubles the facts: `recall_context` puts facts ahead of
passages inside its character budget, and ten more facts could push out
the passage the question found on its own. The receipt's
`facts_dropped` counts the facts the cut left out. The cut bounds the
number of facts, not their length. Reciprocal rank fusion was tried
first: because the second query contains the question's own words, the
question's matches rank in both lists and collect credit twice, which
buried the passage only the carried terms found. On the development
pairs it moved no second turn at weights 1, 2 or 4.

In `MemoryContext` the second recall runs under the same fixed scope as
the first (the same `where`, kind, source and dates). A message that
alone reads as an overview request ("What's the status of it?") is
searched instead when its carried query names a subject; if the carried
query reads as an overview too, nothing is carried and the reason says
so. Adaptive retrieval plans its own queries and tool mode chooses its
own searches, so neither is combined with follow-up queries.

## The receipt

`ContextReceipt.followup` (chat) and `receipt["followup"]` (context);
absent when the setting is off.

| Field | Meaning |
|---|---|
| `mode` | carry or rewrite, as configured |
| `method` | how the second query was made: carry, rewrite, or none |
| `applied` | whether a second query was searched |
| `reason` | why, in every case |
| `query` | the second query exactly as searched, or null |
| `carried`, `from_message` | the verbatim terms and the index of the user message they came from |
| `cues` | what made the turn read as leaning on the conversation |
| `terms_found`, `cut` | new terms that turn named; `cut` is true when `MAX_CARRIED_TERMS` (6) dropped some |
| `turns_unread` | earlier user turns not read when `MAX_LOOKBACK_TURNS` (4) was reached without a term |
| `facts_dropped` | facts left out when the two recalls were fused, to hold no more than the longer one gave |
| `model_calls`, `fallback` | rewrite only: calls made, and why the model's answer was not used |
| `history_omitted` | rewrite only: turns left out by the 6,000-byte history bound |

## Settings

| Variable | Meaning |
|---|---|
| `SCONE_FOLLOWUP_QUERIES` | off (default), carry, or rewrite; requires `SCONE_CONVERSATIONS_JOURNAL`, not with `SCONE_ADAPTIVE_RETRIEVAL` or tool mode; `serve-conversations` needs `--model-factory` for it |
| `SCONE_FOLLOWUP_URL`, `SCONE_FOLLOWUP_MODEL` | rewrite only, required: a self-hosted OpenAI-compatible endpoint and model |
| `SCONE_FOLLOWUP_API_KEY` | rewrite only, optional; never borrowed from another setting |
| `SCONE_FOLLOWUP_TIMEOUT` | rewrite only: seconds for the model, default 5, at most 60 |

## Measured

[`benchmarks/followup-queries-v1.results.md`](../benchmarks/followup-queries-v1.results.md):
on 26 synthetic two-turn pairs (hash embedder, in-memory stores, limit
5), second-turn recall at 5 went from 20/26 with the setting off to
26/26 with carry, on both surfaces, and every first turn was found
either way (26/26). A carried query costs one more recall: median
second-turn preparation went from 2.28 ms to 4.22 ms in memory. The
pairs are small and authored; the numbers show
the mechanism, not a rate on real conversations. What carry misses is
listed there too.
