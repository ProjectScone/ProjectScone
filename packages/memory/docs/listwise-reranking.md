# Listwise reranking: a model orders the candidates, a window at a time

A scorer — the offline cross encoder, or a model asked for a number per
passage — judges each passage alone. The listwise reranker
([`retrieval/listwise.py`](../src/scone_memory/retrieval/listwise.py))
shows a chat model several passages together, numbered, and asks for their
order: `[2] > [1] > [3]`. It is RankGPT's shape, the one LlamaIndex ships as
`RankGPTRerank`. It is off unless asked for, and the default recall path
calls no model.
With a 3B local model it made retrieval worse on the measurement below; the
example shows the wiring, not a recommendation.

```python
from scone_memory.providers.llm import OpenAICompatibleChat
from scone_memory.retrieval.listwise import ListwiseReranker

chat = OpenAICompatibleChat("http://127.0.0.1:11434/v1", "llama3.2-ctx8k")
memory = await MemoryEngine(documents, vectors, embedder,
                            reranker=ListwiseReranker(chat, window=20, step=10,
                                                      passage_bytes=1024, timeout=60)).open()
result = await memory.recall("notes", "When did I move the bills to autopay?", limit=10)
result.rerank.listwise   # the receipt
```

Or from the environment, on the chat model the server already has:

```
SCONE_CHAT_URL=http://127.0.0.1:11434/v1
SCONE_CHAT_MODEL=llama3.2-ctx8k
SCONE_RERANKER_LISTWISE=1
# SCONE_RERANKER_LISTWISE_WINDOW=20         passages per call, 2..20
# SCONE_RERANKER_LISTWISE_STEP=10           1..window-1, default half the window
# SCONE_RERANKER_LISTWISE_PASSAGE_BYTES=1024  64..8192
# SCONE_RERANKER_LISTWISE_TIMEOUT=60        seconds for the whole pass, up to 600
```

It cannot be combined with `SCONE_RERANKER_FACTORY` or the cross encoder;
either would set the order again.

## What it does

The candidates are the ones every reranker gets: retained, in scope,
at most `SCONE_RERANK_LIMIT` of them within `SCONE_RERANK_MAX_BYTES`, in
fused order.

1. **Windows, bottom first.** With 32 candidates, a window of 20 and a
   step of 10, the model is asked three times: places 12–31, then 2–21,
   then 0–19. A passage the model puts first in a low window is in the
   next window up, so a relevant passage found at place 30 can reach the
   top in one pass. The last window is always the top one, full when
   there are enough candidates.
2. **Bounded calls.** Each call shows at most `window` passages, each on
   one line and cut to `passage_bytes` UTF-8 bytes on a character
   boundary. Every cut is counted, per call and per pass.
3. **A strict reading.** A reply is taken only when it is the ranking and
   nothing else: `[3] > [1] > [2]`, spaces allowed around `>`. A reply
   that names some passages validly ranks those first and leaves the rest
   in the order they were shown (`partial`). Anything else — prose around
   the ranking, an identifier outside the window, one named twice — leaves
   the whole window as it was (`unparseable`, with `unparseable`,
   `out_of_range` or `repeated` as the reason). The reply's text is never
   kept. Recall's `degraded` says how many windows were partial or
   unparseable.
4. **One deadline.** The whole pass has `timeout` seconds. When it passes,
   or the model fails, fused order stands, `rerank.status` is `failed`,
   `degraded` carries the reason (`listwise timeout after 60s; fused order
   kept`, or `listwise model failed: ChatError; fused order kept` — the
   type, never the message), and the receipt shows how far the pass got.
   The engine's `rerank_timeout` (at most 10 s) is a scorer's budget and
   does not apply to a listwise pass.

The `rerank_score` on each item is its place as a number (the first of
32 gets 32.0): an order, never a confidence.

## The receipt

`result.rerank.listwise` (and the recall event's `rerank.listwise`):

| field | meaning |
|---|---|
| `window`, `step`, `passage_bytes`, `timeout` | the bounds the pass ran under |
| `calls` | one per model call: `start`/`end` (places in the list as it stood), `passages`, `clipped`, `outcome` (`ranked`, `partial`, `unparseable`, `failed`, `timeout`), `ranked`, `moved` (places in the window that changed), `duration_ms`, `reason` |
| `model_calls` | calls started, including one a deadline interrupted |
| `moved` | candidates whose final place differs from fused order; 0 on a fallback |
| `clipped` | candidates longer than the byte bound |
| `fallback` | why fused order was kept, or null |

A scorer's trace has no `listwise` key at all.

## Where it differs from the reference

LlamaIndex's `RankGPTRerank` sends every node in one call, cuts each to
300 words, and reads the reply by keeping every digit in it, so a sentence
of prose becomes a ranking of whatever numbers it contains, and nothing
says so. Its RankLLM integration slides a window, through the `rank_llm`
package. Here the window is built in, every call is bounded by bytes, the
reading is strict and reported per window, the pass has one deadline with
a fallback that names its reason, and the receipt counts calls and moves.
The strict reading has a cost the measurement below shows: a window that
names one identifier twice is refused whole, where the reference keeps the
first occurrence.

## Measured

**With `llama3.2-ctx8k` it made retrieval worse, and it is slow. Keep it off
with a 3B model.**

[`benchmarks/listwise_reranking.py`](../benchmarks/listwise_reranking.py) ran
20 LongMemEval-S items (`stratified_sample(seed=42)`), hashed-token embedder,
in-memory stores, engine defaults, recall limit 30 folded to sessions. Each
query ran fused (`rerank=False`) and listwise (`rerank=True`) on the same
stored passages back to back, the order alternating, three repeats. Listwise:
32 candidates (`SCONE_RERANK_LIMIT` default), window 20, step 10, 1024 bytes
per passage (none was cut), Ollama at temperature 0, **timeout 300 s, not the
60 s default**. The box was shared: load average 50–75, another model (`llama3.1-ctx8k`) was
loaded beside it when the run began.

| | fused | listwise |
|---|---:|---:|
| R@5 (any evidence session), median of 3 repeats | 0.95 | 0.75 |
| MRR, median of 3 repeats | 0.858 | 0.656 |
| seconds per query, median of the 3 repeats' medians | 0.13 | 63.5 |
| seconds per query, median of all 60 | 0.15 | 61.0 |

- R@5 and MRR were the same in all three repeats. At k=5, listwise won 1 item
  and lost 5, each repeat.
- Of 179 model calls, 35 windows were ranked, 88 partial, 54 unparseable, and
  2 hit the deadline. All 54 unparseable windows were refused for naming an
  identifier twice (`repeated`). The reference would have kept each one's
  first occurrence and used the rest.
- 105 of the 177 replies began with the last passage shown (`[20]`). With
  windows sliding bottom first, that carries the bottom candidate of each
  window upward.
- A model call took 19.7 s at the median. 33 of the 60 listwise queries took
  more than 60 s, so under the default timeout they would have fallen back
  to fused order (`listwise timeout after 60s; fused order kept`). At 300 s,
  2 of 60 did, and each returned exactly the fused sessions.

The fused R@5 of 0.95 comes from this 20-item sample with stem prefixes on
(the default). It is not the scoreboard's 50-item number and does not move it.
Not measured here: a larger model (`llama3.1-ctx8k`), smaller windows, a
lenient reading of repeated identifiers, and LlamaIndex's `RankGPTRerank` with
the same model on the same candidates.
