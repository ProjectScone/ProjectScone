"""Custom Transformer module for Sentence Transformers to load BGE-VL-CLIP models.

BGE-VL-CLIP uses late fusion for multimodal inputs: text and image features are
projected separately and summed. This module subclasses Transformer to add support
for the ("image", "text") compound modality by summing the text and image projected
embeddings in the forward pass.
"""

from __future__ import annotations

from sentence_transformers.base.modules.transformer import Transformer


class BGEVLCLIPTransformer(Transformer):
    @classmethod
    def load(cls, model_name_or_path, *, trust_remote_code=False, **kwargs):
        # The custom modeling_MMRet_CLIP.py has a non-persistent position_ids buffer
        # bug on transformers v5+. The standard CLIPModel loads these weights fine,
        # so we always load the underlying model without trust_remote_code.
        return super().load(model_name_or_path, trust_remote_code=False, **kwargs)

    def forward(self, features, **kwargs):
        modality = features.get("modality", "text")

        if modality != ("image", "text"):
            return super().forward(features, **kwargs)

        # For ("image", "text") modality: run text and image through their respective
        # forward paths, then sum the projected embeddings.
        text_features = {**features, "modality": "text"}
        image_features = {**features, "modality": "image"}

        text_features = super().forward(text_features, **kwargs)
        image_features = super().forward(image_features, **kwargs)

        features[self.module_output_name] = (
            text_features[self.module_output_name] + image_features[self.module_output_name]
        )
        return features

    @property
    def modalities(self):
        return ["text", "image", ("image", "text")]