---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
tags:
  - reranker
  - cross-encoder
  - visual-document-retrieval
  - question-answering retrieval
  - multimodal reranking
  - semantic-search
  - rag
  - transformers
  - sentence-transformers
pipeline_tag: text-ranking
language:
  - multilingual
library_name: transformers
---


## **Model Overview**


### Description:

The `llama-nemotron-rerank-vl-1b-v2` was developed by NVIDIA for multimodal question-answering retrieval. It is optimized for providing a logit score that represents how relevant a document page is to a given query. The model can process documents in the form of image, text, or image and text combined. The expected images are screenshots of document pages or slides. Documents are ranked given a user query in text form. The model supports images containing text, tables, charts, and infographics. We report the model's performance by evaluating it on the popular ViDoRe V1, V2 and the new Vidore V3 (see [Vidore LB](https://huggingface.co/spaces/vidore/vidore-leaderboard) for details) multimodal retrieval benchmarks, and on two internally curated visual retrieval datasets.

The reranking model serves as a key component of a **multimodal** retrieval system, such as a vision RAG pipeline, where it helps improve overall accuracy. A multimodal retrieval system often uses a multimodal embedding model (dense) to return relevant documents given the input. A reranking model can be used to rerank the potential candidates into a final order. The reranking model takes the query and document pairs as input, and its self-attention can perform deeper interaction between their tokens. It’s not scalable to apply a ranking model to all documents in the knowledge base for a given query; therefore, ranking models are often deployed to rerank top candidate documents retrieved by embedding models.

This model is ready for commercial use. <br>

### Use Case: <br>
The llama-nemotron-rerank-vl-1b-v2 is most suitable for users who want to build a multimodal question-and-answer application over a large corpus, leveraging the latest dense retrieval technologies. <br>

### License/Terms of Use:

The use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
Additional Information: [Llama 3.2 Community Model License Agreement](https://www.llama.com/llama3_2/license/). Built with Llama.

### Release Date
12/18/2025 via https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2

## Model Architecture:

**Architecture Type:** Transformer<br>
**Network Architecture:** : Eagle VLM architecture with SigLIP 2 400M vision encoder and [llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) model as language model. <br>

The llama-nemotron-rerank-vl-1b-v2 is a cross-encoder model with approximately 1.7B parameters. It is a fine-tuned version of an NVIDIA Eagle-family model, which consists of the SigLIP 2 400M vision encoder and the Llama 3.2 1B language model. The final embedding output by the decoder is aggregated using a mean pooling strategy, and a binary classification head is fine-tuned for the ranking task. The CrossEntropy loss is used to maximize the likelihood of (visual) documents containing information to answer the question and minimize the likelihood for (negative) documents that do not contain information to answer the question.

The vision-language model reranker incorporates key innovations from NVIDIA, including [Eagle 2 work](https://arxiv.org/abs/2501.14818) which uses a tiling-based VLM architecture, and [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse). The Eagle 2 architecture, available on [Hugging Face](https://huggingface.co/collections/nvidia/eagle-2-6764ba887fa1ef387f7df067), significantly enhances multimodal understanding through its dynamic tiling and mixture of vision encoders design. It particularly improves performance on tasks that involve high-resolution images and complex visual content.


## Input(s): <br>
**Input Type(s):** Image, Text<br>

**Input Format(s):** <br>
- Image: Red, Green, Blue (RGB) <br>
- Text: String <br>

**Input Parameters:** 
- Image: Two-Dimensional (2D) <br>
- Text: One-Dimensional (1D) <br>

**Other Properties Related to Input:** <br>

The model was fine-tuned exclusively on image data, using max_input_tiles = 4 and the maximum context length of 2048 tokens. For evaluation, it was tested on image-only, image+text, and text-only inputs, with max_input_tiles = 6 and the maximum context length of 10240 tokens. Inputs exceeding the maximum length are truncated. <br>


## Output(s)

**Output Type(s):** Floats <br>

**Output Format(s):** List of Floats <br>

**Output Parameters:** 1D <br>

**Other Properties Related to Output:** Each value corresponds to a raw logit. Users can choose to apply a Sigmoid activation function to the logits to convert them into probabilities during model usage.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br> 


## Installation

The model requires transformers version >= 4.56.0, and optionally flash-attention.

`pip install "transformers>=4.56.0"` <br>
`pip install "flash-attn>=2.6.3,<2.8" --no-build-isolation`


## Usage

### Using Sentence Transformers

Install Sentence Transformers:
```bash
pip install sentence_transformers
```

```python
from sentence_transformers import CrossEncoder
from transformers.image_utils import load_image

model = CrossEncoder("nvidia/llama-nemotron-rerank-vl-1b-v2", trust_remote_code=True)

query = "How is AI improving the intelligence and capabilities of robots?"
documents = [
    "AI enables robots to perceive, plan, and act autonomously.",
    "AI is transforming autonomous vehicles by enabling safer, smarter, and more reliable decision-making on the road.",
    "A biological foundation model designed to analyze and generate DNA, RNA, and protein sequences.",
]
images = [
    load_image("https://developer.download.nvidia.com/images/isaac/nvidia-isaac-lab-1920x1080.jpg"),
    load_image("https://blogs.nvidia.com/wp-content/uploads/2018/01/automotive-key-visual-corp-blog-level4-av-og-1280x680-1.png"),
    load_image("https://developer-blogs.nvidia.com/wp-content/uploads/2025/02/hc-press-evo2-nim-25-featured-b.jpg"),
]

# Text-only reranking
pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)
print(scores)
# [-0.738  -3.859  -7.75 ]

rankings = model.rank(query, documents)
print(rankings)
# [{'corpus_id': 0, 'score': -0.73828125}, {'corpus_id': 1, 'score': -3.859375}, {'corpus_id': 2, 'score': -7.75}]

# Image-only reranking
pairs = [(query, img) for img in images]
scores = model.predict(pairs)
print(scores)
# [-4.125  -5.156  -5.969]

rankings = model.rank(query, images)
print(rankings)
# [{'corpus_id': 0, 'score': -4.125}, {'corpus_id': 1, 'score': -5.15625}, {'corpus_id': 2, 'score': -5.96875}]

# Image+text reranking
multimodal_docs = [{"image": img, "text": doc} for img, doc in zip(images, documents)]
pairs = [(query, doc) for doc in multimodal_docs]
scores = model.predict(pairs)
print(scores)
# [-2.594  -4.531  -6.219]

rankings = model.rank(query, multimodal_docs)
print(rankings)
# [{'corpus_id': 0, 'score': -2.59375}, {'corpus_id': 1, 'score': -4.53125}, {'corpus_id': 2, 'score': -6.21875}]
```

### Using Transformers

```
import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor
from transformers.image_utils import load_image

modality = "image"

# Load model
model_path = "nvidia/llama-nemotron-rerank-vl-1b-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    device_map="auto"
).eval()

# Build processor kwargs (base settings)
processor_kwargs = {
    "trust_remote_code": True,
    "max_input_tiles": 6,
    "use_thumbnail": True
}

# Set rerank_max_length based on modality
if modality == "image":
    processor_kwargs["rerank_max_length"] = 2048
elif modality == "image_text":
    processor_kwargs["rerank_max_length"] = 10240
elif modality == "text":
    processor_kwargs["rerank_max_length"] = 8192

# Load processor with modality-specific kwargs
processor = AutoProcessor.from_pretrained(
    model_path,
    **processor_kwargs
)

query = "How is AI improving the intelligence and capabilities of robots?"
image_paths = [
    "https://developer.download.nvidia.com/images/isaac/nvidia-isaac-lab-1920x1080.jpg",
    "https://blogs.nvidia.com/wp-content/uploads/2018/01/automotive-key-visual-corp-blog-level4-av-og-1280x680-1.png",
    "https://developer-blogs.nvidia.com/wp-content/uploads/2025/02/hc-press-evo2-nim-25-featured-b.jpg"
]

# Load all images
images = [load_image(img_path) for img_path in image_paths]

# Text descriptions corresponding to each image/document
document_texts = [
    "AI enables robots to perceive, plan, and act autonomously.",
    "AI is transforming autonomous vehicles by enabling safer, smarter, and more reliable decision-making on the road.",
    "A biological foundation model designed to analyze and generate DNA, RNA, and protein sequences.",
]

if modality == "image":
    # Prepare inputs: same query, different images
    examples = [{
        "question": query,
        "doc_text": "",
        "doc_image": image
    } for image in images]

elif modality == "image_text":
    examples = [{
        "question": query,
        "doc_text": doc_text,
        "doc_image": image
    } for image, doc_text in zip(images, document_texts)]

elif modality == "text":
    # Prepare inputs: same query, different texts
    examples = [{
        "question": query,
        "doc_text": doc_text,
        "doc_image": ""
    } for doc_text in document_texts]

else:
    raise ValueError(f"Invalid modality: {modality}. Must be 'image', 'image_text', or 'text'")

# Process with processor
batch_dict = processor.process_queries_documents_crossencoder(examples)

# Move to device
batch_dict = {
    k: v.to(device) if isinstance(v, torch.Tensor) else v
    for k, v in batch_dict.items()
}

# Run inference
with torch.no_grad():
    outputs = model(**batch_dict, return_dict=True)

# Get logits
logits = outputs.logits
logits_flat = logits.squeeze(-1)
    
# Get sorted indices (highest to lowest)
sorted_indices = torch.argsort(logits_flat, descending=True)

print(f"\nRanking (highest to lowest relevance for the modality {modality}):")
for rank, idx in enumerate(sorted_indices, 1):
    doc_idx = idx.item()
    logit_val = logits_flat[doc_idx].item()
    if modality == "text":
        print(f"  Rank {rank}: logit={logit_val:.4f} | Text: {document_texts[doc_idx]}")
    else:  # image or image_text modality
        print(f"  Rank {rank}: logit={logit_val:.4f} | Image: {image_paths[doc_idx]}")

```


## vLLM Usage

The model can be served with [vLLM](https://github.com/vllm-project/vllm) for high-throughput scoring and reranking. The checkpoint includes a query-document score template in `chat_template.jinja`. Select it explicitly with `--chat-template` for serving or `chat_template` for `LLM.score()`. The Score/Rerank APIs do not automatically apply the checkpoint template; omitting it changes the model inputs and scores.

See the [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/specific_models/#llama-nemotron-multimodal) for full details.

### Online Serving

Download the bundled score template and start the server:

```bash
hf download nvidia/llama-nemotron-rerank-vl-1b-v2 chat_template.jinja \
  --local-dir nemotron-vl-rerank-config

vllm serve nvidia/llama-nemotron-rerank-vl-1b-v2 \
  --runner pooling \
  --trust-remote-code \
  --max-model-len 10240 \
  --chat-template nemotron-vl-rerank-config/chat_template.jinja
```

Note: Use `--max-model-len 10240` to support all modalities including image+text. A smaller value like `2048` can be used if only processing image-only documents.

Send a rerank request:
```python
import requests

url = "http://localhost:8000/rerank"

# Text-only reranking
response = requests.post(url, json={
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "query": "How is AI improving the intelligence and capabilities of robots?",
    "documents": [
        "AI enables robots to perceive, plan, and act autonomously.",
        "A biological foundation model designed to analyze DNA, RNA, and protein sequences.",
    ],
})
print(response.json())

# Image document reranking
response = requests.post(url, json={
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "query": "How is AI improving the intelligence and capabilities of robots?",
    "documents": [
        {"content": [{"type": "image_url", "image_url": {"url": "https://example.com/page.png"}}]},
        {"content": [
            {"type": "text", "text": "AI enables robots to perceive, plan, and act autonomously."},
            {"type": "image_url", "image_url": {"url": "https://example.com/page2.png"}},
        ]},
    ],
})
print(response.json())
```

### Offline / In-Process

```python
from vllm import LLM

llm = LLM(
    model="nvidia/llama-nemotron-rerank-vl-1b-v2",
    runner="pooling",
    max_model_len=10240,
    trust_remote_code=True,
)

# Reuse the template bundled with the checkpoint.
score_template = llm.get_tokenizer().get_chat_template()

query = "How is AI improving the intelligence and capabilities of robots?"
documents = [
    "AI enables robots to perceive, plan, and act autonomously.",
    "A biological foundation model designed to analyze DNA, RNA, and protein sequences.",
]

# Text-only scoring
outputs = llm.score(query, documents, chat_template=score_template)
for doc, output in zip(documents, outputs):
    print(f"Score: {output.outputs.score:.4f} | {doc}")

# Image document scoring
outputs = llm.score(
    query,
    [
        {"content": [{"type": "image_url", "image_url": {"url": "https://example.com/page.png"}}]},
        {"content": [
            {"type": "text", "text": "AI enables robots to perceive, plan, and act autonomously."},
            {"type": "image_url", "image_url": {"url": "https://example.com/page2.png"}},
        ]},
    ],
    chat_template=score_template,
)
for output in outputs:
    print(f"Score: {output.outputs.score:.4f}")
```


## Software Integration:

**Runtime Engine(s):**
- TensorRT <br>

**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere, NVIDIA Hopper, NVIDIA Lovelace, NVIDIA Blackwell <br>
**Preferred/Supported Operating System(s):** Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>


## Model Version(s):
llama-nemotron-rerank-vl-1b-v2

## Training and Evaluation Datasets:

## Training Dataset

The development of large-scale, public open-QA datasets has driven significant progress in powerful vision-language models, as well as vision embedding and reranking models. However, following issues limit the use of these models in commercial settings:

- Commercial licensing restrictions in popular public datasets, such as MS MARCO. <br>
- Many multimodal datasets rely on synthetic data generated with proprietary models.


NVIDIA's training dataset is based on public QA datasets, and only includes datasets that have a license for commercial applications.

**Properties:** The model was fine-tuned with publicly available image datasets. We also generated synthetic queries for the image corpora, whose original queries were produced using proprietary models.

**Data Modality**

- Image

**Image Training Data Size**
- Less than a Million Images

**Data Collection Method by dataset** Hybrid: Automated, Human, Synthetic <br>
**Labeling Method by dataset:** Hybrid: Automated, Human, Synthetic <br>


### Evaluation Dataset:

We evaluate the embedding + reranking pipeline on a set of evaluation benchmarks. We applied the ranking model to the candidates retrieved from the `llama-nemotron-embed-vl-1b-v2` model.

**Vision document retrieval benchmarks** <br>
We evaluated `llama-nemotron-rerank-vl-1b-v2` on five various visual document retrieval datasets: the popular ViDoRe V1, V2, the new Vidore V3, and two internal visual document retrieval datasets:

- **DigitalCorpora-10k:**  A dataset with questions based on a corpus of 10k documents from [DigitalCorpora](https://digitalcorpora.org/corpora/file-corpora/cc-main-2021-31-pdf-untruncated/) that have a good mixture of text, tables, and charts.
- **Earnings V2:** an internal retrieval dataset of 287 questions based on 500 PDFs, mostly consisting of earnings reports from big tech companies.

For those interested in reproducing our results, one of our internal datasets (DigitalCorpora-10k) can be created by following instructions in this [notebook](https://github.com/NVIDIA/nv-ingest/blob/main/evaluation/digital_corpora_download.ipynb) from the NeMo Retriever Extraction GitHub repository.


**Text retrieval benchmarks** <br>
We evaluated  llama-nemotron-rerank-vl-1b-v2 on 92 text retrieval datasets, from the benchmarks BEIR, MIRACL (multi-language), MLQA (cross-language) and MLDR (long-context).

### Evaluation Results <br>

#### Visual Document Retrieval (page retrieval)

In this section, we report the performance of `llama-nemotron-rerank-vl-1b-v2` on different input modalities. In the table below, we can see that compared to the VLM embedding baseline, the VLM reranking model increases the Avg Recall@5 by approximately 7.2% for `text` modality, 6.9% for `image` modality, and 6% for `image + text` modality on 5 evaluation datasets.

Note: `Image+Text` modality means that both the page image and its text (extracted using ingestion libraries like [NV-Ingest](https://github.com/NVIDIA/nv-ingest)) are fed as input to the reranking model for more accurate representation and retrieval.


<table>
<caption style="text-align: left;"><em>Visual Document Retrieval benchmarks- Avg Recall@5 on DC10k, Earnings V2, ViDoRe V1, V2, V3</em></caption>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Modality</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>Text</th>
    <th>Image</th>
    <th>Image + Text</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>llama-nemotron-embed-vl-1b-v2</td>
    <td>71.04%</td>
    <td>71.20%</td>
    <td>73.24%</td>
  </tr>
  <tr>
    <td>+ llama-nemotron-rerank-vl-1b-v2</td>
    <td>76.12%</td>
    <td>76.12%</td>
    <td>77.64%</td>
  </tr>
</tbody>
</table>
<br>

The table below demonstrates the llama-nemotron-rerank-vl-1b-v2's evaluation accuracy performance compared to two other publicly available multimodal reranker models: `jina-reranker-m0` and `MonoQwen2-VL-v0.1`. The Jina model does not have commercial license and it does not support `image+text` modality out of the box, thus, we report `image` only and `text` only evaluation scores for this model.

<table>
  <tr>
    <th></th>
    <th colspan="3">Modality</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>Text</th>
    <th>Image</th>
    <th>Image+Text</th>
  </tr>
  <tr>
    <td>llama-nemotron-rerank-vl-1b-v2</td>
    <td>76.12%</td>
    <td>76.12%</td>
    <td><b>77.64%</b></td>
  </tr>
  <tr>
    <td>jina-reranker-m0</td>
    <td>69.31%</td>
    <td><b>78.33%</b></td>
    <td>NA</td>
  </tr>
  <tr>
    <td>MonoQwen2-VL-v0.1</td>
    <td>74.70%</td>
    <td>75.80%</td>
    <td><b>75.98%</b></td>
  </tr>
</table>

#### Text Retrieval (chunk retrieval)

The llama-nemotron-rerank-vl-1b-v2 demonstrates competitive retrieval accuracy on text retrieval benchmarks, comparable to NVIDIA's text-only reranking model llama-nemotron-rerank-1b-v2. This means you can deploy NVIDIA's VLM-based [llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) embedding model along with `llama-nemotron-rerank-vl-1b-v2` reranking model, regardless of whether your retrieval corpus consists of images, text, or both.

<table>
<caption style="text-align: left;"><em>Text Retrieval benchmarks (chunk retrieval) - Avg. Recall@5</em></caption>
  <tr>
    <th>Model</th>
    <th>BEIR retrieval + TechQA</th>
    <th>MIRACL</th>
    <th>MLQA</th>
    <th>MLDR</th>
    <th>Average</th>
  </tr>
  <tr>
    <td>llama-nemotron-embed-1b-v2 + llama-nemotron-rerank-1b-v2</td>
    <td>73.64%</td>
    <td>65.80%</td>
    <td>86.83%</td>
    <td>68.49%</td>
    <td>73.69%</td>
  </tr>
  <tr>
    <td>llama-nemotron-embed-vl-1b-v2 + llama-nemotron-rerank-vl-1b-v2</td>
    <td>73.18%</td>
    <td>65.71%</td>
    <td>87.05%</td>
    <td>69.98%</td>
    <td>73.98%</td>
  </tr>
</table>
<br>

**Data Collection Method by dataset** Hybrid: Automated, Human, Synthetic <br>
**Labeling Method by dataset:** Hybrid: Automated, Human, Synthetic <br>
**Properties**  More details on ViDoRe benchmarks can be found on  their Hugging Face [page](https://huggingface.co/vidore).

## Inference

- Acceleration Engine: TensorRT <br>
- Test Hardware: H100, A100, L40S, A10G, B200, RTX PRO 6000


## Citation
```
@inproceedings{moreira2025_nvretriever,
author = {Moreira, Gabriel de Souza P. and Osmulski, Radek and Xu, Mengyao and Ak, Ronay and Schifferer, Benedikt and Oldridge, Even},
title = {Improving Text Embedding Models with Positive-aware Hard-negative Mining},
year = {2025},
isbn = {9798400720406},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746252.3761254},
doi = {10.1145/3746252.3761254},
pages = {2169–2178},
numpages = {10},
keywords = {contrastive learning, distillation, embedding models, hard-negative mining, rag, text retrieval, transformers},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}
```

## **Ethical Considerations**

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Explainability, Bias, Safety, and Privacy sections.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## Bias

| Field | Response |
| ----- | ----- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing | None |
| Measures taken to mitigate against unwanted bias | None |

## Explainability

| Field | Response |
| ----- | ----- |
| Intended Application & Domain: | Passage or document ranking for question and answer retrieval. |
| Model Type: | Transformer cross-encoder. |
| Intended User: | Generative AI developers working with conversational AI models. This is suitable for users building question-answering applications over large multimodal corpora and aiming to improve retrieval performance by reranking a set of candidate documents for a given question. The corpus may include visually rich document images (e.g., pages with text, figures, tables, charts, or infographics) and/or text extracted from documents. |
| Output: | List of Floats (Score/Logit indicating if a passage/document relevant to a question). |
| Describe how the model works: | The model outputs a relevance score (logit) reflecting how strongly it predicts that the passage or document contains the information required to answer the question. |
| Technical Limitations: | The model's max sequence length is 10240. Longer text inputs should be truncated. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | N/A |
| Verified to have met prescribed NVIDIA quality standards: | Yes |
| Performance Metrics: | Accuracy, Throughput, and Latency. |
| Potential Known Risks: | This model does not always guarantee to provide a meaningful ranking of document(s) for a given query. |
| Licensing & Terms of Use: | The use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt). Additional Information: [Llama 3.2 Community Model License Agreement](https://www.llama.com/llama3_2/license/). Built with Llama. |

## Privacy

Field | Response
:---- | :----
Generatable or reverse engineerable personal data? | No
Personal data used to create this model? | Yes
Was consent obtained for any personal data used? | Unknown (Externally-Sourced Data)
How often is dataset reviewed? | Before release and as needed in response to dataset, model, or policy changes.
Is a mechanism in place to honor data subject right of access or deletion of personal data? | Unknown (Externally-Sourced Data)
If personal data was collected for the development of the model, was it collected directly by NVIDIA? | No
If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects? | Not Applicable
If personal data was collected for the development of this AI model, was it minimized to only what was required? | Yes
Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | No
Is there provenance for all datasets used in training? | Yes
Does data labeling (annotation, metadata) comply with privacy laws? | Yes
Is data compliant with data subject requests for data correction or removal, if such a request was made? | No, not possible with externally sourced data.
Applicable Privacy Policy | [NVIDIA Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/)

## Safety

| Field | Response |
| ----- | ----- |
| Model Application(s): | Document Reranking for Retrieval. User queries can be text and documents can be text, document page images, charts, tables, and infographics. |
| Describe the physical safety impact (if present). | Not Applicable |
| Use Case Restrictions: | The use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt). Additional Information: [Llama 3.2 Community Model License Agreement](https://www.llama.com/llama3_2/license/). Built with Llama. |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |
