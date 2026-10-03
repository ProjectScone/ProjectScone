# Scone's agent loop on held-out HotpotQA: 61.2% EM, and why

## Run

- **Date and questions:** 2 October 2026, the held-out half of HotpotQA dev (3,676 questions, by SHA-256 of id; see
  [`hop_rule.py`](hop_rule.py)).
- **Corpus:** the pooled corpus of 66,635 paragraphs.
- **Model:** Gemma 4 31B through OpenRouter, reasoning off, for every arm.
- **Agent arms:** [`agent.py`](agent.py) runs `EvidenceToolLoop` with `initial_search=True`, up to 4 tool calls and
  4 rounds, over `ScopedMemoryTools`. The `+ hop` arm gives `search_memory` the gated second hop.
- **Single-pass arms:** these give the top five documents to the same model with **the agent's own system prompt**
  (`run.py --prompt agent`): Scone's default assistant prompt, the short-answer format and the agent instruction.
  So agent and single-pass differ only in the tools.
- **Scoring:** the official normalization. Every question was answered by every arm.

## Results

| Arm | EM | F1 | Gave up (INSUFFICIENT_EVIDENCE) |
| --- | ---: | ---: | ---: |
| single-pass LlamaIndex (vector + BM25) | 47.9% | 58.7% | 24.2% |
| single-pass Scone | 48.4% | 59.4% | 24.2% |
| single-pass Scone + gated hop | 51.8% | 63.0% | 20.8% |
| Scone agent loop | 60.4% | 74.3% | 8.1% |
| **Scone agent loop + hop** | **61.2%** | **75.1%** | 7.4% |

Paired on exact match (left side's wins vs right side's wins):

| Comparison | Wins–losses | Sign test |
| --- | --- | --- |
| agent + hop vs single-pass LlamaIndex | 601–112 | p = 1e-81 |
| agent + hop vs single-pass Scone + gated hop | 466–120 | p = 5e-49 |
| agent vs single-pass Scone | 557–113 | p = 2e-71 |
| agent + hop vs agent | 71–42 | p = 0.008 |
| single-pass Scone + gated hop vs single-pass LlamaIndex | 237–94 | p = 2e-15 |

**Agent behaviour:**
- The agent searched again on 673 questions with the hop, about 18%; on the rest it answered from the host's first
  search.
- With the hop, the median turn took 4.3 s (p95 11.1 s).
- The two agent arms read 7.5 million and 7.7 million prompt tokens.

## What it shows

- **The agent loop answers far more questions correctly than one search does, with the same model and
  instructions.** It gains 12 points of EM over single-pass Scone and 13 over LlamaIndex. Most of the gain is that it
  rarely gives up: 8% against 24%.
- **A first comparison against single-pass under the original "answer from the numbered sources" prompt showed a
  +14-point gap.** Part of that was prompt, not tools: single-pass gave up on 28% of questions under that prompt.
  The table above removes that difference.
- **This design cannot separate two causes:**
  - the agent's ability to search again;
  - the different shape of its evidence: top passages and claims as tool output, against whole documents.

  Both are part of what the loop is.
- **The hop helps inside the loop too, by a small margin** (71–42). As a single pass, it adds 3.4 points over Scone.

## Cost

- **Single-pass, agent prompt (11,028 answers):** $2.78, read from the provider's usage records. That is $0.29 per
  million prompt tokens, about three times the listed price: OpenRouter routed these requests to a pricier provider.
- **Agent arms:** their rows record tokens but not cost. Estimated from the account's monthly spend minus the
  measured runs, the two came to about $1.77.
- **Next:** future runs should record the provider's cost for every call.

## Where the agent's gain came from, and a reranker that recovers half of it without an agent loop

### The loss is the second paragraph

For single-pass Scone (agent prompt), split by how many gold paragraphs were in its top five:

| Gold paragraphs in context | Questions | Gave up | EM |
| --- | ---: | ---: | ---: |
| both | 2,382 (65%) | 4.2% | 65.8% |
| one | 1,250 (34%) | 60.3% | 16.8% |
| none | 44 (1%) | 81.8% | 2.3% |

- **With both paragraphs, single-pass and the agent read alike:** 65.8% vs 65.6%.
- **The agent's gain is the questions where only the first hop was found.** It searched again on 656 of them and
  answered 63.0%.

### Candidates were already there; the order was not

On the development half, the first pass's top ten plus the text hop's top ten hold both gold paragraphs for 87.7% of
questions (90.5% with entity-name searches). Rule-based merging put both in the top five for only 71.3%.

Reranking the pool with local cross-encoders (`pool_rerank.py`) on the development half:

| Reranker | Both in top 5 | Rerank time |
| --- | ---: | --- |
| MiniLM-L6 (ms-marco) | 67.8% | |
| MiniLM, bridge-aware (rescored against question + leading paragraph) | 69.4% | |
| **bge-reranker-base** | **82.5%** (both in top 2: 64.0%) | 2.2 s |

Changes for speed:

- Dropping the entity searches (5.6 per question) cost 1.5 points.
- The 8-bit model (`onnx/model_quantized.onnx`) held quality at 2.3x the speed.
- Reading 600 characters of each paragraph brought reranking to 373 ms (480 ms p95), within 0.7 points of the best
  configuration.

**The configuration chosen on the development half:** the first pass's top 10 plus the text hop's top 10, about 15
candidates, reranked by 8-bit bge reading 600 characters.

### Held-out half (3,676), scored once

**Retrieval:**

| Arm | hit@1 | all@2 | all@5 | all@10 |
| --- | ---: | ---: | ---: | ---: |
| LlamaIndex | 87.4% | 36.8% | 64.2% | 75.9% |
| Scone first pass | 87.2% | 37.3% | 64.8% | 76.4% |
| Scone + gated text hop | 87.2% | 37.3% | 70.3% | 83.2% |
| **Scone hop pool + bge-int8 rerank** | **93.9%** | **61.2%** | **79.9%** | **86.7%** |

Reranking took 369 ms median (475 ms p95) on CPU.

**Answers**, from the same reader and the agent prompt:

| Arm | EM | F1 | Gave up |
| --- | ---: | ---: | ---: |
| single-pass LlamaIndex | 47.9% | 58.7% | 24.2% |
| single-pass Scone + gated hop | 51.8% | 63.0% | 20.8% |
| **single-pass Scone hop pool + bge-int8 rerank** | **55.3%** | **67.7%** | 15.5% |
| agent loop + hop | 61.2% | 75.1% | 7.4% |

Paired on exact match (rerank arm's wins vs the other side's wins):

| Against | Wins–losses | Sign test |
| --- | --- | --- |
| LlamaIndex | 372–101 | p = 2e-37 |
| gated hop | 268–140 | p = 2e-10 |
| agent loop + hop | 162–380 | p = 3e-21 |

- **A reranker is needed for the top of the list, in this form:** over the hop's candidate pool, with a strong model.
- **One search plus a 0.4 s rerank closes about half of the gap** between single-pass and the agent loop.
- **Not measured:** the agent loop with the same reranker on its searches. This answer run cost $0.70, from the
  provider's usage records.
