# Reranker models

Reranker and multimodal retrieval weights, committed with the code so that `git clone` or `git pull` is enough to
get them. No Hugging Face access, Git LFS or download script is needed. They add about 5 GB to a clone.

```bash
git pull                              # or: git clone https://github.com/ProjectScone/ProjectScone.git
python3 models/combine.py             # rejoin every weight file and verify it
python3 models/combine.py --check     # verify only, change nothing
```

To clone the code without the weights:

```bash
git clone --filter=blob:none --sparse https://github.com/ProjectScone/ProjectScone.git
cd ProjectScone && git sparse-checkout set .github deploy packages scripts terraform tests
```

CI checks out the code directories the same way, so it never downloads `models/`.

GitHub refuses files over 100 MB, so each weight file is committed as 90 MiB chunks under `models/<name>/parts/`.
`combine.py` needs only the Python standard library. It joins the chunks into `models/<name>/model.safetensors` and
checks every chunk and the whole file against `MANIFEST.json`. The whole-file hash is Hugging Face's own sha256 of the
original, so a joined file is byte-for-byte the published model. Without Python:

```bash
cat models/BGE-VL-large/parts/model.safetensors.part-* > models/BGE-VL-large/model.safetensors
shasum -a 256 models/BGE-VL-large/model.safetensors    # compare with MANIFEST.json
```

## Models

| Folder | What it is | Inputs | Size | License | Hugging Face revision |
| --- | --- | --- | ---: | --- | --- |
| `llama-nemotron-rerank-vl-1b-v2` | NVIDIA multimodal cross-encoder reranker: SigLIP 2 vision + Llama 3.2 1B (~1.7B parameters) | text, image, image+text | 3.4 GB | NVIDIA Open Model License; Llama 3.2 Community License. Built with Llama. | `b8a9987` |
| `granite-embedding-reranker-english-r2` | IBM Granite cross-encoder (ModernBERT, 149M), English | text | 0.6 GB | Apache-2.0 | `d09d3d6` |
| `BGE-VL-large` | BAAI multimodal **embedder** on CLIP ViT-L/14: first-stage retrieval, not a reranker | text, image, image+text | 0.9 GB | MIT | `40fb482` |

Each folder keeps the model's own `README.md` (its model card, with benchmarks and usage) and the license files
it ships.

- **Multimodal pipeline:** BGE-VL-large embeds and retrieves; `llama-nemotron-rerank-vl-1b-v2` reorders the top 20 to 50.
- **Text only:** Granite is the small, fast cross-encoder.

## Hardware

These were measured on 2026-10-03 on an Apple M3 Max (14 cores), CPU only and 8 threads unless noted. Each run reranks
one query against 20 passages of about 110 words, after a warm-up.

| Model | Precision | Peak memory | 20 passages |
| --- | --- | ---: | ---: |
| Granite reranker r2 | float32 | 1.4 GB | 0.44 s |
| BGE-VL-large (embed query and passages) | float32 | 3.0 GB | 0.20 s |
| BGE-VL-large | bfloat16 | 0.8 GB | 1.1 s |
| NVIDIA rerank-vl-1b-v2 | float32 | 10.7 GB | 2.3 s |
| NVIDIA rerank-vl-1b-v2 | bfloat16 (its default) | 3.3 GB | 23.5 s |
| NVIDIA rerank-vl-1b-v2, Apple GPU (MPS) | default | — | 0.74 s |

- **On a CPU, load NVIDIA and BGE-VL in float32.** Pass `model_kwargs={"torch_dtype": torch.float32}`. Their configs
  default to bfloat16, which most CPUs run 5 to 10 times slower.
- **Text-only on a modest CPU:** 4 cores and 4 GB free run Granite well.
- **NVIDIA:** plan for 12 GB of free RAM on CPU, or use a GPU. With CUDA, its card recommends bfloat16 and
  flash-attention.
- **An x86 office machine** has no MPS. Expect the CPU rows, scaled by core count and speed.

## Use, fully offline

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r models/requirements.txt
python models/rerank_example.py nvidia     # granite | bge-vl | all
```

`rerank_example.py` loads each model from this folder with `HF_HUB_OFFLINE=1`, so it fails loudly instead of reaching
the network. The NVIDIA and BGE-VL folders ship their own modelling code (`trust_remote_code=True` runs that local
code). Read it before running it on a locked-down machine.

## With Scone

- Scone's in-process reranker (`SCONE_RERANKER_CROSS_ENCODER_DIR`) accepts ONNX cross-encoders from a fixed list.
- These models need a Python reranker factory (`SCONE_RERANKER_FACTORY=module:factory`) that wraps one of the loaders
  in `rerank_example.py`.

## Integrity

`MANIFEST.json` records, for every model:

- the Hugging Face repository and commit;
- the license;
- every small file committed;
- the weight file's size and sha256;
- each chunk's path and sha256, in order.

Before commit, every small file was checked against Hugging Face's listed size and sha256. Each weight file was
hashed as it streamed from Hugging Face, and the hash matched Hugging Face's LFS sha256.
