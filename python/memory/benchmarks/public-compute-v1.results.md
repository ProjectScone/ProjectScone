# Public computation-tool comparison v1 — 9 September 2026

Offering the optional computation tool produced **115/200 exact matches with
computation disabled (57.5%) and 113/200 enabled (56.5%)**. The model never selected
`compute_memory`. This experiment therefore measures exposing its schema and
guidance, including subsequent retrieval behavior; it provides no empirical
evidence about calculation accuracy, operand selection, rejection handling, or
the benefit of actually using the calculator. Computation remains optional and
disabled by default.

This is one paired run on reused development data and a shared workstation. The
observed difference is not evidence of a reliable causal regression or a general
limit on tool-assisted generation.

## Frozen setup and integrity

The [protocol](public-compute-v1.protocol.md) was committed and pushed as
`3816873` before inference: commit time 07:58:19 UTC, first turn 07:59:16 UTC.
The run finished at 09:56:55 UTC. All 400 planned turns are terminal: 398 completed
and two failed. No turn was retried, replaced, or excluded from scoring.

- The same 200 original questions, 100 HotpotQA and 100 SQuAD 1.1, ran once per
  arm with alternating off/on order. Questions retain their original wording.
  The separate 200 reserved questions remain unrun.
- The corpus contains 2,176 original paragraphs pooled across development and
  reserved sources, including distractors. SQLite and self-managed Qdrant 1.19.1
  served the copied ledger and 2,835 vectors; cached BGE-small-en-v1.5 generated
  query embeddings. This is not the official per-question distractor setting
  or full-Wikipedia retrieval.
- Python 3.14.7 ran `EvidenceToolLoop(initial_search=True)` with
  `SelfHostedStructuredToolChat`. The only arm switch was
  `ScopedMemoryTools(enable_computation=...)`.
- The installed `gemma4-e4b-ctx8k:latest` digest was
  `858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`:
  Q4_K_M, 8,192-token context, temperature zero, reasoning disabled, and
  2,048 maximum output tokens. Ollama reports this alias's parameter size as
  8.0B; the alias name is not a measurement of parameter count.
- Production limits allowed four tool calls including initial search, four
  rounds, and 120 seconds per turn. Scoped retrieval retained its two-second
  budget. No fine-tuning, question-specific prompts, gold evidence, answer
  revision, forced calculation, or manual answer repair was applied.

The separate scorer ran only after terminal completion and verified frozen
code, inputs, and stores. Independent review checked every scheduled row, raw
final answer, score, and evidence-coverage record. All 4,106 offered passage
occurrences match original source text; no authored facts or relations were
injected. The upstream Hotpot checker agrees on answer EM/F1 for all 198
completed Hotpot answers; its two failed turns receive zero. Supporting-fact
and joint metrics were not measured. Scoring parity does not establish parity
with the official benchmark's retrieval setup.

## Answer scores

F1 values below are percentages of the mean token F1. Each dataset has 100
questions per arm; the overall denominator is 200 per arm, including failures.

| Dataset | Off EM | On EM | Off F1 | On F1 | EM gains / losses / ties |
| --- | ---: | ---: | ---: | ---: | --- |
| All | 57.5% (115) | 56.5% (113) | 68.8275% | 67.9748% | 1 / 3 / 196 |
| HotpotQA | 46% (46) | 45% (45) | 57.6304% | 56.1230% | 1 / 2 / 97 |
| SQuAD 1.1 | 69% (69) | 68% (68) | 80.0246% | 79.8266% | 0 / 1 / 99 |

Token-F1 pairs comprise four gains, eight losses, and 188 ties. Twenty raw
answers differ, including wording changes with equal scores and two failure
pairs. Exact `INSUFFICIENT_EVIDENCE` responses increased from 14 to 17
(Hotpot 11 to 13; SQuAD three to four). This count does not include every
natural-language expression of uncertainty.

Three of the four EM changes concern answer wording: the enabled arm shortened
the fuel-economy answer to the reference phrase, added a sentence around
"the Middle Rhine," and added the artist's name to "Strange Fruit." The other
change answered "Sammy Hagar" where the question requested an instrument and
the off arm answered "rhythm guitar." These are qualitative observations about
the unchanged outputs, not alternate scores. All 20 changed answers and their
recorded evidence IDs appear in the appendix.

The historical single-generation baseline scored 64% EM. Its pipeline differs
from this structured tool loop, so it is context rather than a controlled
estimate of the effect of adding tools.

## Retrieval and tool behavior

Of 200 initial-search pairs, 199 succeeded in both arms: 132 packets are exactly
equal and 67 differ. A recursive comparison found only 103 changed
`items[].score` values, with maximum absolute difference approximately
0.000001. All 199 successful pairs have identical ordered passages, source IDs,
text, and other packet fields. One pair has an initial retrieval failure.
Scores were visible to the model; the requests must not be described as
identical. The numeric differences' cause was not established by this audit.

Across all passages offered over a turn, literal evidence coverage was:

| Dataset | Off annotation coverage | On annotation coverage | Off all annotations | On all annotations | Off support-document recall | On support-document recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 95.3167% | 95.5667% | 90.5% (181/200) | 91% (182/200) | 96.25% | 96% |
| HotpotQA | 91.6333% | 91.1333% | 82% | 82% | 92.5% | 92% |
| SQuAD 1.1 | 99% | 100% | 99% | 100% | 100% | 100% |

Hotpot coverage checks annotated supporting sentences inside the matching
source document. SQuAD checks a literal reference answer inside a matching
document. These are evidence-availability proxies, not verified reasoning or
answer correctness; all failures remain in their denominators.

| Parsed model action | Off attempts | On attempts |
| --- | ---: | ---: |
| `search_memory` | 342 | 367 |
| `read_memory` | 106 | 129 |
| `trace_memory` | 0 | 0 |
| `compute_memory` | 0 | 0 |

Initial automatic searches are additional to these model-selected actions.
The calculator was genuinely available in all 542 enabled-arm structured
requests across 199 turns that reached the model. It was absent from all 510
off-arm structured requests. Final prose requests correctly omit action tools.
Raw action JSON confirms zero computation selections, dispatches, successes,
operations, or argument/quote rejections. There is no calculation sample from
which to estimate accuracy. No fact graph was indexed, so this run also does
not evaluate graph traversal.

| Provider calls in a turn | Off turns | On turns |
| --- | ---: | ---: |
| 0 | 0 | 1 |
| 1 | 39 | 23 |
| 2 | 12 | 9 |
| 3 | 11 | 14 |
| 4 | 138 | 153 |

For successful turns, tool-call counts (including initial search) were
off: 39 with one, 12 with two, 11 with three, 137 with four;
on: 23 with one, nine with two, 14 with three, 153 with four.
The on arm made more search/read attempts without improving answer scores.

## Failures, timing, and context

- Off, `hotpotqa:5abbd7985542992ccd8e7fb8`: whole-turn `TimeoutError` after
  120,006.582 ms. Three provider calls completed; the fourth was cancelled
  without a response. The aggregate scorer stores an empty error string for
  this exception; its raw type and trace identify the timeout.
- On, `hotpotqa:5a754fc35542996c70cfaedc`: initial retrieval timed out;
  the turn failed after 1,508.295 ms with zero provider calls. The retrieval
  budget reserves time for verification, so failure can occur before the outer
  two-second tool deadline. The underlying slowdown's cause was not established.

Both failures are preserved with zero answer credit. Shared-host contention
may affect timings; these measurements do not isolate its cause.

| Timing | Off p50 / p95 | On p50 / p95 | Off / on observations |
| --- | ---: | ---: | ---: |
| All turns | 14.505 / 34.968 s | 16.036 / 40.783 s | 200 / 200 |
| Hotpot turns | 16.841 / 34.628 s | 17.770 / 45.388 s | 100 / 100 |
| SQuAD turns | 12.655 / 34.968 s | 14.201 / 38.000 s | 100 / 100 |
| Provider calls | 4.671 / 11.338 s | 4.568 / 11.685 s | 648 / 695 |

The first off/on turns took 24.676/12.909 seconds, with four provider calls each.
The launch residency snapshot contained the background Llama model; the final
snapshot contained both it and Gemma. All 22 scheduled snapshots were saved.
At completion, Ollama reported 9,636,843,355 resident bytes for Gemma and
5,917,649,141 for Llama. These are reported model-residency sizes, not peak
process RSS. The memory server and unrelated services stayed running; development
checks also ran on this workstation. Timing is not isolated throughput.

The off arm made 648 provider calls (647 complete, one cancelled), versus 695
completed calls on: 47 additional calls. Token-usage receipts cover 647/695 calls:

| Reported tokens | Off | On |
| --- | ---: | ---: |
| Total prompt tokens across calls | 1,217,393 | 1,362,483 |
| Total completion tokens across calls | 16,109 | 17,396 |
| Maximum prompt tokens in one call | 4,198 | 4,289 |
| Maximum total tokens in one call | 4,234 | 4,325 |

All completed calls reported finish reason `stop`; none reported `length`.
These receipts do not show the 8,192-token window being reached. They do not
prove that the server never trims input, and byte caps are not token budgets.

## Artifacts and next experiment

Raw requests, responses, decisions, source packets, final answers, scores,
residency records, frozen scripts and copied stores are retained in the ignored
`bench-runs/public-compute-dev-2026-09-09/` directory. Raw datasets and generated
run artifacts are not committed. The hashes below identify the audited run:

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `2d14d003d421f3fe4ab844fef468e30fafa450f40a3a4129113b68aea4354a5d` |
| `observations.json` | `310b9f601767ce6541d2d88fdade4abf05a743280f55a620921dd26d12911cd9` |
| Upstream Hotpot evaluator | `d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0` |

The immediate generation issues are answer-type selection, concise answer
formatting, and knowing when enough evidence has already been found. A future
calculator evaluation also needs a separately frozen sample with genuine
computation requirements. Any changes need a new recorded experiment; this
run's questions, outputs, failures, and scores remain unchanged. Neither this
run nor its unit tests establish universal accuracy or a held-out improvement.

## Appendix: every changed raw answer

There are 20 changed raw answers; the other 180 are byte-for-byte equal.
The entries below preserve exact question and answer strings, scores, and final
retained evidence IDs. `null` evidence IDs indicate a failed turn without a
final receipt; its partial tool packets remain in the raw artifact. Evidence
coverage is computed from all offered packets, which may differ from final
retained evidence. Chunk IDs refer to the frozen copied store, not to external
dataset IDs. Empty strings are failures, not generated abstentions.

### hotpotqa:5abbd7985542992ccd8e7fb8

```json
{
  "question": "Thor Equities owns properties in Soho, which became known for what type of rental properties?",
  "off": {
    "answer": "",
    "status": "failed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": null,
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "trendy upscale boutiques to national and international chain store outlets",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:103",
      "chunk:104",
      "chunk:1246",
      "chunk:2206",
      "chunk:234",
      "chunk:1052",
      "chunk:1247"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:5727526cdd62a815002e9b0f

```json
{
  "question": "What is PPP?",
  "off": {
    "answer": "Public-Private Partnering (PPPs) is a form of procurement in construction that involves relationship contracting, emphasizing a co-operative relationship between the principal and contractor and other stakeholders within a construction project.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.14285714285714285
    },
    "retained_evidence_ids": [
      "chunk:1312",
      "chunk:1708",
      "chunk:1421",
      "chunk:873",
      "chunk:294"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "Public-Private Partnering (PPPs) is a form of partnering in construction that involves a co-operative relationship between the principal and contractor and other stakeholders within a construction project.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.16
    },
    "retained_evidence_ids": [
      "chunk:1312",
      "chunk:1708",
      "chunk:1421",
      "chunk:873",
      "chunk:294"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5ab9753055429970cfb8eb0b

```json
{
  "question": "What movie genre did Emmanuelle Vaugier play a supporting role in?",
  "off": {
    "answer": "\"Secondhand Lions\"",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:211",
      "chunk:2619",
      "chunk:53",
      "chunk:947",
      "chunk:1861",
      "chunk:2790"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 0.5,
      "all_annotations": false,
      "support_document_recall": 0.5
    }
  },
  "on": {
    "answer": "\"40 Days and 40 Nights\"",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:211",
      "chunk:2619",
      "chunk:53",
      "chunk:947",
      "chunk:1861"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 0.5,
      "all_annotations": false,
      "support_document_recall": 0.5
    }
  }
}
```

### hotpotqa:5a80274e5542992e7d278dfb

```json
{
  "question": "Follow the Yellow Brick Road Tour is taking place to commemorate the album that was recorded where?",
  "off": {
    "answer": "Château d'Hérouville in France",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.6666666666666666
    },
    "retained_evidence_ids": [
      "chunk:1022",
      "chunk:2253",
      "chunk:321",
      "chunk:1248",
      "chunk:2230"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "The album was recorded at the Château d'Hérouville in France.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.4
    },
    "retained_evidence_ids": [
      "chunk:1022",
      "chunk:2253",
      "chunk:321",
      "chunk:1248",
      "chunk:2230"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5a88fa145542997e5c09a6e1

```json
{
  "question": "What was the benefit of reintroducing piston power and replacing the most popular turboprop engine in history?",
  "off": {
    "answer": "The piston engine offered much better fuel economy, which Orenda Aerospace felt would be attractive for older aircraft whose engines were reaching the end of their lifespan.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.27586206896551724
    },
    "retained_evidence_ids": [
      "chunk:1703",
      "chunk:66",
      "chunk:2215",
      "chunk:349",
      "chunk:1817"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "much better fuel economy",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:1703",
      "chunk:66",
      "chunk:2215",
      "chunk:349",
      "chunk:1817"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:5705edcd52bb8914006896cb

```json
{
  "question": "How many miles south of San Jose is the north - south midway point located?",
  "off": {
    "answer": "11 miles",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.6666666666666666
    },
    "retained_evidence_ids": [
      "chunk:1877",
      "chunk:921",
      "chunk:1750",
      "chunk:2409",
      "chunk:2756"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "around 11 miles",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.5
    },
    "retained_evidence_ids": [
      "chunk:1877",
      "chunk:921",
      "chunk:1750",
      "chunk:2409",
      "chunk:2756"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5ac4e593554299076e296e37

```json
{
  "question": "OU812 was the second album to feature a vocalist that also played what instrument in \"Finish What Ya Started\"?",
  "off": {
    "answer": "rhythm guitar",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:1010",
      "chunk:2442",
      "chunk:2776",
      "chunk:2347",
      "chunk:2156",
      "chunk:108",
      "chunk:361",
      "chunk:1845",
      "chunk:1011"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 0.5,
      "all_annotations": false,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "Sammy Hagar",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:1010",
      "chunk:2442",
      "chunk:2776",
      "chunk:2347",
      "chunk:2156",
      "chunk:108",
      "chunk:361",
      "chunk:1845",
      "chunk:2270"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 0.5,
      "all_annotations": false,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:57264e2f708984140094c1e5

```json
{
  "question": "What does the plos pathogen paper claim?",
  "off": {
    "answer": "The authors concluded that this new research, together with prior analyses from the south of France and Germany, [the text was cut off].",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.1935483870967742
    },
    "retained_evidence_ids": [
      "chunk:1398",
      "chunk:1201",
      "chunk:2643",
      "chunk:661",
      "chunk:994"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 0.0,
      "all_annotations": false,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "The paper published in PLoS Pathogens concluded that the new research, together with prior analyses from the south of France and Germany, \"ends the debate about the etiology of the Black Death, and unambiguously demonstrates that Y. pestis was the causative agent of the epidemic plague that devastated Europe during the Middle Ages.\"",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.4827586206896552
    },
    "retained_evidence_ids": [
      "chunk:1398",
      "chunk:1201",
      "chunk:2643",
      "chunk:661",
      "chunk:994",
      "chunk:1399"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:5728de0e2ca10214002da9dd

```json
{
  "question": "How is the climate near the savannah grasslands?",
  "off": {
    "answer": "The climate in the savannah grasslands around Nairobi, Kenya, is cooler.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.5
    },
    "retained_evidence_ids": [
      "chunk:826",
      "chunk:2386",
      "chunk:221",
      "chunk:1495",
      "chunk:2385"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "The climate is cooler in the savannah grasslands around the capital city, Nairobi.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.46153846153846156
    },
    "retained_evidence_ids": [
      "chunk:826",
      "chunk:2386",
      "chunk:221",
      "chunk:1495",
      "chunk:2385",
      "chunk:827"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5ac3a7f75542993915413890

```json
{
  "question": "The actor who starred in \"The Hearse\" gained worldwide film by starring in whose series of 1940s films?",
  "off": {
    "answer": "Joseph Cotten",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:1449",
      "chunk:894",
      "chunk:2226",
      "chunk:62",
      "chunk:849",
      "chunk:2227"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:1449",
      "chunk:894",
      "chunk:2226",
      "chunk:62",
      "chunk:849",
      "chunk:2227",
      "chunk:229",
      "chunk:899",
      "chunk:1849",
      "chunk:2216",
      "chunk:234"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5a7a94f155429941d65f26ca

```json
{
  "question": "What is the altitude of the town in England that has Leadgate between it and Garrigill?",
  "off": {
    "answer": "The town is Alston, and its altitude is about 1,000 ft above sea level.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.631578947368421
    },
    "retained_evidence_ids": [
      "chunk:1126",
      "chunk:2742",
      "chunk:988",
      "chunk:2813",
      "chunk:743"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:1126",
      "chunk:2742",
      "chunk:988",
      "chunk:2813",
      "chunk:743"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:572ffc99947a6a140053cefa

```json
{
  "question": "How was Sadat rewarded by the Islamists for his attempts to bring Egypt into modern times and civilization?",
  "off": {
    "answer": "Sadat released Islamists from prison and welcomed home exiles in exchange for political support.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.09090909090909091
    },
    "retained_evidence_ids": [
      "chunk:1354",
      "chunk:1355",
      "chunk:1761",
      "chunk:2242",
      "chunk:886"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "Sadat released Islamists from prison and welcomed home exiles in tacit exchange for political support in his struggle against leftists.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.07142857142857142
    },
    "retained_evidence_ids": [
      "chunk:1354",
      "chunk:1355",
      "chunk:1761",
      "chunk:2242",
      "chunk:886"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:5737a7351c456719005744f5

```json
{
  "question": "What is the force called rgarding a potential field between two locations?",
  "off": {
    "answer": "A conservative force",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:2751",
      "chunk:1559",
      "chunk:643",
      "chunk:2814",
      "chunk:905"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "The force is related directly to the difference in potential energy between two different locations in space, and can be considered to be an artifact of the potential field.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.27586206896551724
    },
    "retained_evidence_ids": [
      "chunk:2751",
      "chunk:1559",
      "chunk:643",
      "chunk:2814",
      "chunk:905"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:572ff12e04bcaa1900d76eff

```json
{
  "question": "What flows between Bingen and Bonn?",
  "off": {
    "answer": "the Middle Rhine",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:178",
      "chunk:563",
      "chunk:2329",
      "chunk:629",
      "chunk:665"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "The Middle Rhine flows between Bingen and Bonn.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.4444444444444444
    },
    "retained_evidence_ids": [
      "chunk:178",
      "chunk:563",
      "chunk:2329",
      "chunk:629",
      "chunk:665"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:56e7752337bdd419002c3fd9

```json
{
  "question": "What types of teachers are retiring the most?",
  "off": {
    "answer": "secondary school teachers",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:1667",
      "chunk:1671",
      "chunk:1513",
      "chunk:1628",
      "chunk:997"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "Secondary school teachers",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:1667",
      "chunk:1671",
      "chunk:1513",
      "chunk:1628",
      "chunk:997"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### squad:573099ee8ab72b1400f9c5dc

```json
{
  "question": "What advancements besides military technology did Europe achieve?",
  "off": {
    "answer": "The provided evidence mentions that European chemists made deadly explosives and that innovations in machinery allowed for improved firearms, but it does not detail other advancements besides military technology.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:615",
      "chunk:311",
      "chunk:2794",
      "chunk:2793",
      "chunk:2803",
      "chunk:312",
      "chunk:313",
      "chunk:1014"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:615",
      "chunk:311",
      "chunk:2794",
      "chunk:2793",
      "chunk:2803",
      "chunk:1014"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5adf72335542992d7e9f935f

```json
{
  "question": "Mehdi Hashemi was thought to be guilty of opposition to the Iranian regime's dealings during what President's administration?",
  "off": {
    "answer": "The evidence does not specify which president's administration the opposition to the regime's dealings occurred during.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:2327",
      "chunk:1077",
      "chunk:252",
      "chunk:797",
      "chunk:1161",
      "chunk:1145"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 0.5,
      "all_annotations": false,
      "support_document_recall": 0.5
    }
  },
  "on": {
    "answer": "The evidence does not specify which president's administration the opposition to the Iranian regime's dealings occurred during.",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:2327",
      "chunk:1077",
      "chunk:252",
      "chunk:797",
      "chunk:1161",
      "chunk:1715",
      "chunk:1739"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5a754fc35542996c70cfaedc

```json
{
  "question": "Who was the last king of the Shahiya?",
  "off": {
    "answer": "Trilochanapala",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": [
      "chunk:2557",
      "chunk:2522",
      "chunk:89",
      "chunk:207",
      "chunk:1544"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "",
    "status": "failed",
    "score": {
      "em": 0.0,
      "f1": 0.0
    },
    "retained_evidence_ids": null,
    "offered_evidence_coverage": {
      "annotation_coverage": 0.0,
      "all_annotations": false,
      "support_document_recall": 0.0
    }
  }
}
```

### hotpotqa:5a909a0355429916514e7505

```json
{
  "question": "Are Dorothy Parker and Maxine Kumin the same nationality?",
  "off": {
    "answer": "Yes",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:1579",
      "chunk:896",
      "chunk:1903",
      "chunk:1586",
      "chunk:1130"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "yes",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:1579",
      "chunk:896",
      "chunk:1903",
      "chunk:1586",
      "chunk:1130"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```

### hotpotqa:5ae47ae05542995ad6573d4f

```json
{
  "question": "The hip hop group Strange Fruit Project relates its name to what song by a famous female jazz musician and singer known for her vocal delivery and improvisational skills?",
  "off": {
    "answer": "Strange Fruit",
    "status": "completed",
    "score": {
      "em": 1.0,
      "f1": 1.0
    },
    "retained_evidence_ids": [
      "chunk:668",
      "chunk:1417",
      "chunk:295",
      "chunk:1587",
      "chunk:2423"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  },
  "on": {
    "answer": "Billie Holiday's \"Strange Fruit\"",
    "status": "completed",
    "score": {
      "em": 0.0,
      "f1": 0.6666666666666666
    },
    "retained_evidence_ids": [
      "chunk:668",
      "chunk:1417",
      "chunk:295",
      "chunk:1587",
      "chunk:2423",
      "chunk:1401"
    ],
    "offered_evidence_coverage": {
      "annotation_coverage": 1.0,
      "all_annotations": true,
      "support_document_recall": 1.0
    }
  }
}
```
