---
license: mit
language:
- en
base_model:
- openai/clip-vit-large-patch14
library_name: sentence-transformers
tags:
- sentence-transformers
- multimodal-retrieval
- embedding-model
pipeline_tag: sentence-similarity
---

<h1 align="center">MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2412.14475">
        <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2412.14475-B31B1B.svg">
    </a>
    <a href="https://github.com/VectorSpaceLab/MegaPairs">
        <img alt="Build" src="https://img.shields.io/badge/Github-Code-blue">
    </a>
    <a href="https://huggingface.co/datasets/BAAI/MegaPairs">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Datasets-MegaPairs-yellow">
</p>

<p align="center">
</a>
    <a href="https://huggingface.co/BAAI/BGE-VL-base">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-BGE_VL_base-yellow">
    </a>
    <a href="https://huggingface.co/BAAI/BGE-VL-large">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-BGE_VL_large-yellow">
    </a>
    <a href="https://huggingface.co/BAAI/BGE-VL-MLLM-S1">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-BGE_VL_MLLM_S1-yellow">
    </a>
    <a href="https://huggingface.co/BAAI/BGE-VL-MLLM-S2">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-BGE_VL_MLLM_S2-yellow">
    </a>
</p>

## News
```2024-3-4``` 🚀🚀 We have released the BGE-VL-MLLM models on Huggingface: [BGE-VL-MLLM-S1](https://huggingface.co/BAAI/BGE-VL-MLLM-S1) and [BGE-VL-MLLM-S2](https://huggingface.co/BAAI/BGE-VL-MLLM-S2). **BGE-VL-MLLM-S1** is trained exclusively on our MegaPairs dataset, achieving outstanding performance in composed image retrieval, with an 8.1% improvement on the CIRCO benchmark (mAP@5) over the previous state-of-the-art. **BGE-VL-MLLM-S2** builds on BGE-VL-MLLM-S1 with an additional epoch of fine-tuning on the MMEB benchmark training set, delivering enhanced performance across a broader range of multimodal embedding tasks.

```2024-12-27``` 🚀🚀 BGE-VL-CLIP models are released on Huggingface: [BGE-VL-base](https://huggingface.co/BAAI/BGE-VL-base) and [BGE-VL-large](https://huggingface.co/BAAI/BGE-VL-large).

```2024-12-19``` 🎉🎉 Release our paper: [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](https://arxiv.org/pdf/2412.14475).

## Release Plan
- [x] Paper
- [x] BGE-VL-base and BGE-VL-large models
- [x] BGE-VL-MLLM model
- [ ] MegaPairs Dataset
- [ ] Evaluation code
- [ ] Fine-tuning code


## Introduction
In this work, we introduce **MegaPairs**, a novel data synthesis method that leverages open-domain images to create *heterogeneous KNN triplets* for universal multimodal retrieval. Our MegaPairs dataset contains over 26 million triplets, and we have trained a series of multimodal retrieval models, **BGE-VL**, including BGE-VL-CLIP (base and large) and BGE-VL-MLLM.

BGE-VL achieve state-of-the-art performance on four popular zero-shot composed image retrieval benchmarks and the massive multimodal embedding benchmark (MMEB). Extensive experiments demonstrate the ***efficiency, scalability, and generalization*** features of MegaPairs. Please refer to our [paper](https://arxiv.org/abs/2412.14475) for more details.

## Model Usage

### Using Sentence Transformers

Install Sentence Transformers:
```bash
pip install sentence_transformers[image]
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/BGE-VL-large", trust_remote_code=True)

query_image = "https://huggingface.co/BAAI/BGE-VL-large/resolve/main/assets/cir_query.png"
candidate_1 = "https://huggingface.co/BAAI/BGE-VL-large/resolve/main/assets/cir_candi_1.png"
candidate_2 = "https://huggingface.co/BAAI/BGE-VL-large/resolve/main/assets/cir_candi_2.png"

# Encode text
text_embeddings = model.encode(["A dog sitting on a bench", "A cat sleeping on a couch"])
print(text_embeddings.shape)
# (2, 768)

# Encode images
image_embeddings = model.encode([query_image, candidate_1])
print(image_embeddings.shape)
# (2, 768)

# Compute similarities
similarities = model.similarity(text_embeddings, image_embeddings)
print(similarities)
# tensor([[0.1255, 0.1018],
#         [0.0161, 0.0271]])

# Composed image retrieval: encode image+text query, compare with image candidates
query_embeddings = model.encode([{
    "image": query_image,
    "text": "Make the background dark, as if the camera has taken the photo at night",
}])
candidate_embeddings = model.encode([candidate_1, candidate_2])
scores = model.similarity(query_embeddings, candidate_embeddings)
print(scores)
# tensor([[0.3696, 0.1714]])
```

You can pass string texts, images as PIL Images, local paths, URLs, or a combination of text and images (with a dictionary format) to the model's `encode` function. The model will automatically process the inputs and return the corresponding embeddings. You can then compute cosine similarities or perform retrieval tasks based on these embeddings.

### Using transformers
You can easily use BGE-VL-CLIP models based on ```transformers```
```python
import torch
from transformers import AutoModel

MODEL_NAME = "BAAI/BGE-VL-base" # or "BAAI/BGE-VL-large"

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True) # You must set trust_remote_code=True
model.set_processor(MODEL_NAME)
model.eval()

with torch.no_grad():
    query = model.encode(
        images = "./assets/cir_query.png", 
        text = "Make the background dark, as if the camera has taken the photo at night"
    )

    candidates = model.encode(
        images = ["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"]
    )
    
    scores = query @ candidates.T
print(scores)
```

See the [demo](./retrieval_demo.ipynb) for a complete example of using BGE-VL for multimodel retrieval.


### 2. BGE-VL-MLLM Models


```python
import torch
from transformers import AutoModel
from PIL import Image

MODEL_NAME= "BAAI/BGE-VL-MLLM-S1"

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.cuda()

with torch.no_grad():
    model.set_processor(MODEL_NAME)

    query_inputs = model.data_process(
        text="Make the background dark, as if the camera has taken the photo at night", 
        images="./assets/cir_query.png",
        q_or_c="q",
        task_instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
    )

    candidate_inputs = model.data_process(
        images=["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"],
        q_or_c="c",
    )

    query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
    candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
    
    query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
    candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)

    scores = torch.matmul(query_embs, candi_embs.T)
print(scores)
```


## Model Performance
### Zero-Shot Composed Image Retrieval

BGE-VL sets a new performance benchmark in zero-shot composed image retrieval tasks. On the CIRCO benchmark, our BGE-VL-base model, with only 149 million parameters, surpasses all previous models, including those with 50 times more parameters. Additionally, BGE-VL-MLLM achieves an 8.1% improvement over the previous state-of-the-art model.

<img src="./assets/res-zs-cir.png" width="800">

### Zero-Shot Performance on MMEB

BGE-VL-MLLM achieves state-of-the-art zero-shot performance on the Massive Multimodal Embedding Benchmark (MMEB), despite being trained only on the ImageText-to-Image paradigm. This demonstrates the excellent generalization capability of MegaPairs for multimodal embedding.

<img src="./assets/res-zs-mmeb.png" width="800">

### Fine-Tuning Performance on MMEB

After fine-tuning on downstream tasks, BGE-VL-MLLM maintains its leading performance. Notably, it surpasses the previous state-of-the-art by 7.1% on the MMEB out-of-distribution (OOD) set. These results demonstrate the robust generalization capability of BGE-VL-MLLM and highlight the potential of MegaPairs as foundational training data for universal multimodal embedding.

<img src="./assets/res-ft-mmeb.png" width="800">

### Performance Scaling
MegaPairs showcases **scalability**: BGE-VL-base improves as training data increases. It also demonstrates **efficiency**: with just 0.5M training samples, BGE-VL-base significantly outperforms MagicLens, which uses the same CLIP-base backbone and was trained on 36.7M samples.

<img src="./assets/res-scaling.png" width="800">


## License
The annotations for MegaPairs and the BGE-VL models are released under the [MIT License](LICENSE). The images in MegaPairs originate from the [Recap-Datacomp](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B), which is released under the CC BY 4.0 license.



## Citation
If you find this repository useful, please consider giving a star ⭐ and citation

```
@article{zhou2024megapairs,
  title={MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval},
  author={Zhou, Junjie and Liu, Zheng and Liu, Ze and Xiao, Shitao and Wang, Yueze and Zhao, Bo and Zhang, Chen Jason and Lian, Defu and Xiong, Yongping},
  journal={arXiv preprint arXiv:2412.14475},
  year={2024}
}
```
