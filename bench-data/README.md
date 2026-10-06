# Benchmark datasets

The public datasets Scone's benchmarks run on, committed so that results can be reproduced from a clone. Run outputs
are not here; each benchmark's results file records its own numbers.

```bash
python3 bench-data/prepare.py              # unpack LongMemEval-S; no network needed
python3 bench-data/prepare.py --m          # also download LongMemEval-M (2.7 GB) and convert it to JSONL
python3 bench-data/prepare.py --ontonotes  # also download the OntoNotes 5 test split
python3 bench-data/prepare.py --check      # verify every committed file against MANIFEST.json
```

`prepare.py` uses only the Python standard library.

## Committed here

| File | Dataset | Size | Licence |
| --- | --- | ---: | --- |
| `longmemeval_s.json.gz` | LongMemEval-S, 500 questions (unpacks to `longmemeval_s.json`, 278 MB) | 78 MB | MIT |
| `longmemeval_oracle.json` | LongMemEval oracle sessions | 15 MB | MIT |
| `temporal-40.json`, `preference-30.json`, `preference-10.json` | question subsets drawn from LongMemEval | 41 MB | MIT |
| `syn-7.json` | 40 small questions in LongMemEval format, for smoke tests | 0.1 MB | MIT |
| `public-qa/hotpot_dev_distractor_v1.json` | HotpotQA dev, distractor setting (7,405 questions) | 61 MB | CC BY-SA 4.0 |
| `public-qa/squad_dev_v1.1.json` | SQuAD v1.1 dev | 4.9 MB | CC BY-SA 4.0 |
| `qasper/qasper-test-v0.3.json` | QASPER v0.3 test | 18 MB | CC BY 4.0 |
| `ner/fewnerd-supervised-test.parquet`, `ner/fewnerd-labels.*` | Few-NERD supervised test split | 4.8 MB | CC BY-SA 4.0 |
| `memoryagentbench/Conflict_Resolution.parquet` | MemoryAgentBench conflict-resolution split | 1.5 MB | MIT |
| `public-qa/hotpot_evaluate_v1.py`, `qasper/qasper_evaluator.py` | the datasets' official scoring scripts | | Apache-2.0 |

The files are stored as downloaded, without edits. LongMemEval-S is gzipped and unpacks to the original bytes; it
and the oracle file match the sha256 that Hugging Face publishes for them. The temporal and preference files are
question subsets selected from LongMemEval, not publisher files. `MANIFEST.json` records each file's source, licence,
size and sha256.

Each dataset keeps its own licence, listed above. Scone's licence does not apply to them. The CC BY-SA files are
redistributed unchanged under CC BY-SA 4.0. Cite the datasets' authors when you use them.

## Downloaded, not committed

| File | Why | Source |
| --- | --- | --- |
| `longmemeval_m_cleaned.jsonl` (2.4 GB) | too large for the repository | Hugging Face `xiaowu0162/longmemeval-cleaned`, pinned by sha256 and rewritten as one question per line |
| `ner/ontonotes5-test.parquet` | its licence does not allow redistribution | Hugging Face `tner/ontonotes5`, pinned by sha256 |

## Where the benchmarks expect them

- **LongMemEval harness:** `--dataset bench-data/longmemeval_s.json` or `bench-data/longmemeval_m_cleaned.jsonl`.
- **Public QA (HotpotQA, SQuAD):** `--data-dir bench-data/public-qa`.
- **Entity recognition:** `--data-dir bench-data/ner`.
