"""Load each model from this repo's models/ folder, fully offline, and score a tiny example.

    python models/rerank_example.py nvidia     # or: granite, bge-vl, all

Run models/combine.py first so models/<name>/model.safetensors exists. HF_HUB_OFFLINE=1 makes any attempt to reach
the network fail loudly instead of quietly downloading something else.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

MODELS = Path(__file__).resolve().parent
QUERY = 'Which company makes the H100 GPU?'
DOCS = ['NVIDIA designs the H100 data-center GPU.', 'Basketball is popular in the United States.',
        'The A100 and H100 are NVIDIA accelerators used to train language models.']


def nvidia() -> None:
    """Multimodal: documents may be text, PIL images, or {'text': ..., 'image': ...}."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(str(MODELS / 'llama-nemotron-rerank-vl-1b-v2'), trust_remote_code=True)
    print(model.rank(QUERY, DOCS))


def granite() -> None:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(str(MODELS / 'granite-embedding-reranker-english-r2'))
    print(model.rank(QUERY, DOCS))


def bge_vl() -> None:
    """An embedder, not a reranker: retrieve with it, then rerank with nvidia."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODELS / 'BGE-VL-large'), trust_remote_code=True)
    query, docs = model.encode([QUERY]), model.encode(DOCS)
    print(model.similarity(query, docs))


RUNNERS = {'nvidia': nvidia, 'granite': granite, 'bge-vl': bge_vl}

if __name__ == '__main__':
    choice = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, run in RUNNERS.items():
        if choice in ('all', name):
            print(f'== {name}')
            run()
