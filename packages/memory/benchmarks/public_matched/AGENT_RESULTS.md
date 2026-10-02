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
