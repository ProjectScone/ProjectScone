# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union, Tuple
import dataclasses
from dataclasses import field

import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import ProcessorMixin

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclasses.dataclass
class Conversation:
    """Manages prompt construction with system messages and multi-turn dialogues."""

    # System instruction prepended to prompts
    system_message: str = ""
    # Role identifiers for dialogue turns
    roles: Tuple[str, str] = ("", "")
    # Message history as (role, content) pairs
    messages: List[List[str]] = field(default_factory=list)
    # Separator token between messages
    sep: str = ""
    # Token IDs that trigger generation stopping
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Construct the formatted prompt string from system message and dialogue history."""
        ret = self.system_message + self.sep
        for role, message in self.messages:
            if message:
                ret += role + message + self.sep
            else:
                ret += role
        return ret

    def append_message(self, role: str, message: str):
        """Add a message turn to the dialogue history."""
        self.messages.append([role, message])


def get_conv_template(name: str) -> Conversation:
    """Initialize a conversation instance with default configuration."""
    return Conversation(
        stop_token_ids=[128259, 128001],
    )


def load_image(image):
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
    elif isinstance(image, dict):
        if "disk_path" in image:
            return Image.open(image["disk_path"])
        elif "base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["base64"])))
        elif "url" in image:
            response = requests.get(image["url"])
            return Image.open(BytesIO(response.content))
        elif "bytes" in image:
            return Image.open(BytesIO(image["bytes"]))
        else:
            raise ValueError(f"Invalid image: {image}")
    else:
        raise ValueError(f"Invalid image: {image}")


def build_transform(input_size, norm_type="imagenet"):
    if norm_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif norm_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    previous version mainly foucs on ratio.
    We also consider area ratio here.
    """
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
        # new area > 60% of original image area is enough.
        factor_based_on_area_n_ratio = min(area_ratio, 0.6) * min(
            target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
        )

        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class LlamaNemotronVLRerankProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Any,
        padding: Union[bool, str] = True,
        rerank_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        max_input_tiles: int = 2,
        num_image_token: int = None,
        prompt_template: str = None,
        force_image_size: int = 512,
        template: str = "bidirectional-llama-retriever",
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        self.padding = padding
        self.rerank_max_length = rerank_max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        tokenizer.padding_side = "left"
        tokenizer.model_input_names = tokenizer.model_input_names + ["pixel_values"]
        self.tokenizer = tokenizer

        self.norm_type = "siglip"
        self.image_size = force_image_size
        self.max_input_tiles = max_input_tiles
        self.num_image_token = num_image_token
        self.system_message = ""
        self.prompt_template = prompt_template
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail

        super().__init__(self.tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Optional[List] = None,
        images: Optional[List[Any]] = None,
        text_kwargs: Optional[Dict[str, Any]] = None,
        images_kwargs: Optional[Dict[str, Any]] = None,
        common_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process inputs for Sentence Transformers CrossEncoder compatibility.

        For text-only pairs, ``text`` is a list of (query, document) tuples.
        For image-based inputs, ``images`` contains document images alongside
        ``text`` containing the query strings.

        Args:
            text: List of text strings or list of (query, document) tuples.
            images: Optional list of PIL Images for document pages.
            text_kwargs: Keyword arguments for text processing.
            images_kwargs: Keyword arguments for image processing (unused).
            common_kwargs: Common keyword arguments (e.g. return_tensors).
        """
        text_kwargs = text_kwargs or {}
        common_kwargs = common_kwargs or {}
        tokenizer_kwargs = {**common_kwargs, **text_kwargs}
        return_tensors = tokenizer_kwargs.pop("return_tensors", "pt")
        padding = tokenizer_kwargs.pop("padding", self.padding)
        truncation = tokenizer_kwargs.pop("truncation", True)

        if text is not None and len(text) > 0 and isinstance(text[0], (tuple, list)):
            # Text pairs from CrossEncoder: [("query1", "doc1"), ("query2", "doc2")]
            formatted = [
                self.prompt_template_question_passage(q, d) if self.prompt_template == "v1"
                else f"{q} \n {d}"
                for q, d in text
            ]
            doc_images = [None] * len(text)
        elif images is not None:
            # Image documents with optional text
            formatted = [
                self.prompt_template_question_passage("", t) if self.prompt_template == "v1"
                else t
                for t in (text or [""] * len(images))
            ]
            doc_images = images
        else:
            # Plain text (fallback)
            formatted = text or []
            doc_images = [None] * len(formatted)

        documents = [
            {"image": img if img is not None else "", "text": t}
            for img, t in zip(doc_images, formatted)
        ]
        return self.process_query_documents(
            documents,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
        )

    def apply_chat_template(
        self,
        messages: List,
        tokenize: bool = True,
        return_dict: bool = True,
        text_kwargs: Optional[Dict[str, Any]] = None,
        images_kwargs: Optional[Dict[str, Any]] = None,
        audio_kwargs: Optional[Dict[str, Any]] = None,
        videos_kwargs: Optional[Dict[str, Any]] = None,
        common_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """Process chat-formatted messages for multimodal CrossEncoder pairs.

        When Sentence Transformers encounters non-text pairs (e.g. image
        documents), it converts them to structured messages with ``"query"``
        and ``"document"`` roles and calls this method.

        Args:
            messages: A conversation or a batch of conversations. Each is a list of
                message dicts with ``"role"`` (``"query"``/``"document"``) and
                ``"content"`` keys.
            tokenize: Whether to tokenize or return the rendered prompt text.
            return_dict: Whether to return a dict (always True for ST).
            text_kwargs: Keyword arguments for text processing.
            images_kwargs: Keyword arguments for image processing (unused).
            audio_kwargs: Unused, for API compatibility.
            videos_kwargs: Unused, for API compatibility.
            common_kwargs: Common keyword arguments (e.g. return_tensors).
        """
        is_batched = bool(messages) and isinstance(messages[0], (list, tuple))
        if not tokenize:
            return self.tokenizer.apply_chat_template(
                messages,
                chat_template=kwargs.pop("chat_template", None) or self.chat_template,
                tokenize=False,
                prompt_template=self.prompt_template,
                **kwargs,
            )
        if not is_batched:
            messages = [messages]

        text_kwargs = text_kwargs or {}
        common_kwargs = common_kwargs or {}
        tokenizer_kwargs = {**common_kwargs, **text_kwargs}
        return_tensors = tokenizer_kwargs.pop("return_tensors", "pt")
        padding = tokenizer_kwargs.pop("padding", self.padding)
        truncation = tokenizer_kwargs.pop("truncation", True)

        documents = []
        for conversation in messages:
            query_text = ""
            doc_text = ""
            doc_image = None

            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "query":
                    if isinstance(content, str):
                        query_text = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                query_text = part.get("text", "")
                elif role == "document":
                    if isinstance(content, str):
                        doc_text = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    doc_text = part.get("text", "")
                                elif part.get("type") == "image":
                                    doc_image = part.get("image")

            if self.prompt_template == "v1":
                formatted = self.prompt_template_question_passage(query_text, doc_text)
            else:
                formatted = f"{query_text} \n {doc_text}"

            documents.append({
                "image": doc_image if doc_image is not None else "",
                "text": formatted,
            })

        return self.process_query_documents(
            documents,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
        )

    def process_query_documents(
        self,
        documents: Union[Dict, List[Dict]],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        max_length: Optional[int] = None,
        pixel_values_layout: Literal["per_image", "flat_tiles"] = "flat_tiles",
        **kwargs,
    ) -> Dict[str, Any]:
        """Process query-document pairs into model inputs with tokenized text and pixel values.

        Args:
            documents: Either a dict with "images" and "texts" lists, or a list of
                dicts each with "image" and "text" keys. Images can be PIL Images,
                file paths, or None/empty string for text-only documents.
            return_tensors: Output format — "pt" for PyTorch tensors, "np" for numpy arrays.
            padding: Padding strategy passed to the tokenizer. Defaults to the value
                set in the processor constructor.
            truncation: Whether to truncate sequences to the tokenizer's model_max_length.
            max_length: Optional override for the maximum sequence length. If None,
                the tokenizer's model_max_length is used when truncation is enabled.
            pixel_values_layout: How to structure the pixel values output:
                - "flat_tiles": All image tiles concatenated into a single tensor of shape
                  (total_tiles, C, H, W). Different images may contribute different numbers
                  of tiles. None if no images are present. This is the format expected by
                  the model's forward() method.
                - "per_image": A list aligned with the input documents, where each entry
                  is either a tensor of shape (num_tiles, C, H, W) or None.

        Returns:
            Dict with "input_ids", "attention_mask", and "pixel_values".
        """
        if return_tensors not in ("pt", "np"):
            raise ValueError(
                f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'."
            )

        if isinstance(documents, dict):
            images = documents["images"]
            texts = documents["texts"]
            assert len(texts) == len(images)
        elif isinstance(documents, list):
            images = [pair["image"] for pair in documents]
            texts = [pair["text"] for pair in documents]
        else:
            raise ValueError("The documents need to be a dict or list of dicts")

        contents = []
        pil_images_by_idx = {}
        max_input_tile_by_idx = {}
        for idx, (image, text) in enumerate(zip(images, texts)):
            prefix = ""
            if image is not None and image != "":
                pil_images_by_idx[idx] = load_image(image)
                prefix = "<image>"
                max_input_tile_by_idx[idx] = self.max_input_tiles

            # ToDo: Order is hardcoded and different than before. No \n after <image>
            content = text
            if prefix != "":
                content = prefix + " " + content
            contents.append(content)

        assert len(max_input_tile_by_idx) == len(pil_images_by_idx), (
            "The number of max_input_tile_by_idx and pil_images_by_idx should be the same."
        )

        transform = build_transform(
            input_size=self.image_size, norm_type=self.norm_type
        )

        template = get_conv_template(self.template)
        template.system_message = self.system_message

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        content_prompts = []
        pixel_values_list = []
        for i, content in enumerate(contents):
            pil_image = pil_images_by_idx.get(i)
            max_input_tiles = max_input_tile_by_idx.get(i)
            if pil_image is not None:
                if self.dynamic_image_size:
                    image_tiles = dynamic_preprocess(
                        pil_image,
                        image_size=self.image_size,
                        max_num=max_input_tiles,
                        use_thumbnail=self.use_thumbnail,
                    )
                else:
                    image_tiles = [pil_image]

                pixel_values = [transform(item) for item in image_tiles]
                pixel_values = torch.stack(pixel_values).to(dtype=torch.bfloat16)
            else:
                pixel_values = None
            pixel_values_list.append(pixel_values)

            if pixel_values is not None and "<image>" not in content:
                content = "<image> " + content

            # Reseting conversation messages
            template.messages.clear()

            # TODO: do we need this template?
            template.append_message(template.roles[0], content)  # user
            template.append_message(template.roles[1], None)  # assistant
            content_prompt = template.get_prompt()

            if pixel_values is not None:
                num_patches = pixel_values.shape[0]
                image_tokens = (
                    IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                    + IMG_END_TOKEN
                )
                content_prompt = content_prompt.replace("<image>", image_tokens, 1)
            else:
                content_prompt = content_prompt.replace("<image>", "", 1)

            content_prompts.append(content_prompt)

        if max_length is None:
            max_length = self.rerank_max_length

        if padding is None:
            padding = self.padding

        model_inputs = self.tokenizer(
            content_prompts,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if pixel_values_layout == "flat_tiles":
            pixel_values_list = [pv for pv in pixel_values_list if pv is not None]
            if len(pixel_values_list) > 1:
                pixel_values_squeezed = torch.concat(pixel_values_list, axis=0)
            elif len(pixel_values_list) == 1:
                pixel_values_squeezed = pixel_values_list[0]
            else:
                pixel_values_squeezed = None

            if pixel_values_squeezed is not None and return_tensors == "np":
                pixel_values_return_value = (
                    pixel_values_squeezed.to(dtype=torch.float16).cpu().numpy()
                )
            else:
                pixel_values_return_value = pixel_values_squeezed
        elif pixel_values_layout == "per_image":
            if return_tensors == "np":
                pixel_values_return_value = [
                    pv.to(dtype=torch.float16).cpu().numpy() if pv is not None else None
                    for pv in pixel_values_list
                ]
            else:
                pixel_values_return_value = pixel_values_list
        else:
            raise ValueError(
                f"Invalid pixel_values_layout: {pixel_values_layout!r}. "
                f"Must be 'flat_tiles' or 'per_image'."
            )

        batch_docs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": pixel_values_return_value,
        }

        return batch_docs

    def prompt_template_question_passage(self, question, text):
        return f"question:{question} \n \n passage:{text}"

    def process_queries_documents_crossencoder(self, features: List[Dict], **kwargs) -> Dict[str, Any]:
        images = [feature["doc_image"] for feature in features]
        if self.prompt_template == "v1":
            questions_texts = [
                self.prompt_template_question_passage(
                    feature["question"], feature["doc_text"]
                )
                for feature in features
            ]
        else:
            questions_texts = [
                f"{feature['question']} \n {feature['doc_text']}"
                for feature in features
            ]
        batch_dict = self.process_query_documents(
            {"images": images, "texts": questions_texts}, **kwargs
        )

        if "num_labels" in features[0]:
            batch_dict["labels"] = torch.zeros(
                features[0]["num_labels"], dtype=torch.long
            )

        return batch_dict
