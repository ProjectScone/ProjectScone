# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

import inspect
import math
from typing import List, Optional, Tuple, Union, Any, Dict

import torch
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import AutoProcessor, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.utils import logging

from .configuration_llama_nemotron_vl import (
    LlamaBidirectionalConfig,
    LlamaNemotronVLConfig,
    LlamaNemotronVLForSequenceClassificationConfig,
)
from .processing_llama_nemotron_vl import LlamaNemotronVLRerankProcessor

logger = logging.get_logger(__name__)

# Check if native create_bidirectional_mask exists (transformers >= 5.0)
try:
    from transformers.masking_utils import create_bidirectional_mask

    _HAS_NATIVE_BIDIRECTIONAL_MASK = True
except ImportError:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    _HAS_NATIVE_BIDIRECTIONAL_MASK = False

# Detect API differences via introspection
_decoder_forward_params = inspect.signature(LlamaDecoderLayer.forward).parameters
_dynamic_cache_init_params = inspect.signature(DynamicCache.__init__).parameters

# past_key_value (singular) in < 4.56, past_key_values (plural) in >= 4.56
_USE_PLURAL_CACHE_PARAM = "past_key_values" in _decoder_forward_params
# DynamicCache accepts config parameter in >= 4.56
_DYNAMIC_CACHE_ACCEPTS_CONFIG = "config" in _dynamic_cache_init_params


def pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":
        emb = last_hidden.sum(dim=1)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[
                torch.arange(batch_size, device=last_hidden.device), sequence_lengths
            ]
    elif pool_type == "cls_last":
        emb = last_hidden[:, 0]
    elif pool_type == "colbert":
        emb = last_hidden
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


# ============================================================================
# Bidirectional LLaMA Model
# ============================================================================


class LlamaBidirectionalModel(LlamaModel):
    """
    LlamaModel modified to use bidirectional (non-causal) attention.

    Supports transformers 4.44+ through 5.x with a unified forward() implementation.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaBidirectionalConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _create_bidirectional_mask(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if _HAS_NATIVE_BIDIRECTIONAL_MASK:
            return create_bidirectional_mask(
                self.config,
                input_embeds,
                attention_mask=attention_mask,
            )

        # Fallback for transformers < 5.0
        if getattr(self.config, "_attn_implementation", None) == "flash_attention_2":
            has_masked_tokens = (attention_mask == 0).any()
            return attention_mask if has_masked_tokens else None

        return _prepare_4d_attention_mask(attention_mask, input_embeds.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            if _DYNAMIC_CACHE_ACCEPTS_CONFIG:
                past_key_values = DynamicCache(config=self.config)
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        bidirectional_mask = self._create_bidirectional_mask(
            inputs_embeds, attention_mask
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        # Build decoder layer kwargs with correct cache parameter name
        layer_kwargs = {
            "attention_mask": bidirectional_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        if _USE_PLURAL_CACHE_PARAM:
            layer_kwargs["past_key_values"] = past_key_values
        else:
            layer_kwargs["past_key_value"] = past_key_values

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(hidden_states, **layer_kwargs)

            # Decoder returns tuple in < 4.54, tensor in >= 4.54
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class LlamaBidirectionalForSequenceClassification(LlamaPreTrainedModel):
    config_class = LlamaBidirectionalConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.model = LlamaBidirectionalModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]

        pooled_hidden_states = pool(
            last_hidden_states=hidden_states,
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )

        pooled_logits = self.score(pooled_hidden_states)
        pooled_logits = pooled_logits / self.config.temperature

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# ============================================================================
# LlamaNemotronVL Model Classes
# ============================================================================


class LlamaNemotronVLModel(PreTrainedModel):
    """
    LlamaNemotron VL model for vision-language reranking.

    Combines a vision encoder (SigLIP) with a bidirectional language model (LLaMA)
    for cross-modal reranking tasks.
    """

    config_class = LlamaNemotronVLConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["LlamaDecoderLayer"]
    _supports_flash_attn_2 = True  # transformers < 4.54
    _supports_flash_attn = True  # transformers >= 4.54
    _supports_sdpa = True

    def __init__(
        self,
        config: LlamaNemotronVLConfig,
        vision_model: Optional[PreTrainedModel] = None,
        language_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)

        # Propagate attn_implementation to sub-configs for transformers < 4.56
        # which lacks set_attn_implementation. In 4.56+, set_attn_implementation
        # handles this automatically using the sub_configs declared on the config class.
        if not hasattr(PreTrainedModel, "set_attn_implementation"):
            parent_attn = getattr(config, "_attn_implementation", None)
            if parent_attn is not None:
                for sub_config in (config.vision_config, config.llm_config):
                    if getattr(sub_config, "_attn_implementation_autoset", False):
                        sub_config._attn_implementation = parent_attn
                        sub_config._attn_implementation_autoset = False

        # Calculate image token count
        image_size = config.force_image_size or config.vision_config.image_size
        if hasattr(config.vision_config, "grid_size"):
            grid_size = config.vision_config.grid_size
            self.patch_size = 14
            self.num_image_token = int((grid_size * config.downsample_ratio) ** 2)
        else:
            patch_size = config.vision_config.patch_size
            self.patch_size = patch_size
            self.num_image_token = int(
                (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
            )

        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio

        logger.info(f"num_image_token: {self.num_image_token}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == "siglip_vision_model":
                self.vision_model = SiglipVisionModel(config.vision_config)
            else:
                raise NotImplementedError(
                    f"Unsupported vision model type: {config.vision_config.model_type}"
                )

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] in (
                "LlamaBidirectionalModel",
                "LlamaBidirectionalForSequenceClassification",
            ):
                self.language_model = LlamaBidirectionalModel(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        # Vision-to-language projection
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.img_context_token_id = None

        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(
            config.name_or_path, trust_remote_code=True
        )

        self.post_init()

        # transformers 4.54.0-4.55.x have a bug in the flash attention
        # infrastructure where a parameter name mismatch ('implementation' vs
        # 'attn_implementation') in _flash_attention_forward causes FA3 pad/unpad
        # functions to be used with FA2 attention kernels, producing nondeterministic
        # and numerically different results for batched vision+text sequences.
        # This was fixed in 4.56.0.
        from packaging.version import Version

        _tv = Version(transformers.__version__)
        if Version("4.54.0") <= _tv < Version("4.56.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is not supported by this model. "
                f"Versions 4.54.0-4.55.x have a flash attention bug that produces "
                f"nondeterministic and incorrect image embeddings. "
                f"Please use transformers <=4.53.x or >=4.56.0."
            )

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_patches_list: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get text embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Process and inject vision embeddings if present
        if pixel_values is not None:
            if image_flags is None:
                image_flags = torch.ones(pixel_values.shape[0])
            image_flags = image_flags.squeeze(-1)
            vit_embeds = self.extract_feature(pixel_values).to(
                device=input_embeds.device
            )

            if not isinstance(image_flags, list):
                image_flags = image_flags.squeeze(-1)
                vit_embeds = vit_embeds[image_flags == 1]

            # Inject vision tokens into text embeddings
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.config.img_context_token_id).to(input_embeds.device)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                )

            input_embeds = input_embeds.reshape(B, N, C)

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = None
        loss = None

        if hasattr(outputs, "logits"):
            logits = outputs.logits
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(
                    -1, self.language_model.config.vocab_size
                )
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """Extract and project vision features to language model space."""
        # Extract features from vision encoder
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            )
            if hasattr(vit_embeds, "last_hidden_state"):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]

        # Remove CLS token if not using SigLIP
        if not isinstance(self.vision_model, SiglipVisionModel):
            vit_embeds = vit_embeds[:, 1:, :]

        # Apply pixel shuffle and MLP projection
        _, n, c = vit_embeds.shape
        h = w = int(n**0.5)
        vit_embeds = vit_embeds.reshape(-1, h, w, c)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        _, h_s, w_s, c_s = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, h_s * w_s, c_s)
        vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds

    def build_collator(self, tokenizer, **kwargs):
        return self.processor

    def post_loss(self, loss, inputs):
        # Add Dummy Gradients for Vision Encoder to ensure multi-GPU synchronization
        if "pixel_values" in inputs and inputs["pixel_values"] is None:
            dummy_pixels = torch.zeros(
                1, 3, 512, 512, device=loss.device, dtype=self.vision_model.dtype
            )
            dummy_output = self.extract_feature(dummy_pixels)
            loss = loss + dummy_output.sum() * 0.0
        return loss


class CrossEncoderHead(nn.Linear):
    """Classification head for cross-encoder."""

    pass


class LlamaNemotronVLForSequenceClassification(PreTrainedModel):
    """LlamaNemotron VL model for sequence classification (reranking)."""

    config_class = LlamaNemotronVLForSequenceClassificationConfig
    base_model_prefix = "model"
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _no_split_modules = ["LlamaNemotronVLModel"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels

        self.add_module("model", LlamaNemotronVLModel(config))

        score = CrossEncoderHead(
            config.llm_config.hidden_size,
            self.num_labels,
            bias=False,
            dtype=torch.float32,
        )
        self.add_module("score", score)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CrossEncoderHead):
            # Initialize cross-encoder head to avoid NaN/Inf loss
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_patches_list: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            num_patches_list=num_patches_list,
        )

        hidden_states = transformer_outputs.hidden_states[-1]

        pooled_hidden_states = pool(
            last_hidden_states=hidden_states,
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )

        pooled_logits = self.score(pooled_hidden_states.to(self.score.weight.dtype))
        pooled_logits = pooled_logits / self.config.temperature
        if torch.isnan(pooled_logits).any():
            raise ValueError("NaN detected in pooled_logits!")

        loss = None

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build_collator(self, tokenizer, **kwargs):
        max_input_tiles = kwargs.pop("max_input_tiles", self.config.max_input_tiles)
        prompt_template = kwargs.pop("prompt_template", self.config.prompt_template)
        return LlamaNemotronVLRerankProcessor(
            tokenizer=tokenizer,
            max_input_tiles=max_input_tiles,
            num_image_token=self.model.num_image_token,
            prompt_template=prompt_template,
            **kwargs,
        )

    def post_loss(self, loss, inputs):
        # Add Dummy Gradients for Vision Encoder to ensure multi-GPU synchronization
        if "pixel_values" in inputs and inputs["pixel_values"] is None:
            dummy_pixels = torch.zeros(
                1, 3, 512, 512, device=loss.device, dtype=self.model.vision_model.dtype
            )
            dummy_output = self.model.extract_feature(dummy_pixels)
            loss = loss + dummy_output.sum() * 0.0
        return loss
