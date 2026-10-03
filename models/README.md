# Reranker models (the `models` branch)

This branch holds model weights only. It shares no history with `main`, so cloning or building Scone never downloads
them. Fetch it when you want the models:

```bash
git fetch origin models
git worktree add ../scone-models models    # or: git checkout models
cd ../scone-models
python3 models/combine.py                  # rejoin every weight file and verify it
python3 models/combine.py --check          # verify only, change nothing
```

To get only this branch without the rest of the project:
`git clone --single-branch --branch models https://github.com/ProjectScone/ProjectScone.git scone-models` (about 5 GB).

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
