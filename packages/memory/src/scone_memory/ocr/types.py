"""Provider-neutral OCR observations in displayed image coordinates."""
from __future__ import annotations

import math
from typing import Protocol

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer, model_validator

#: What a region of a page is, as a reader of the page would call it. One
#: vocabulary for every source of labels: a layout engine's answer is
#: mapped onto it, and the inferred labels use only the part of it that
#: geometry and text can tell (``ocr.labels``).
RegionLabel = Literal['title', 'heading', 'paragraph', 'list', 'table', 'figure', 'caption', 'header', 'footer',
                      'page_number', 'footnote', 'formula', 'code', 'sidebar', 'reference']
REGION_LABELS: tuple[str, ...] = ('title', 'heading', 'paragraph', 'list', 'table', 'figure', 'caption', 'header',
                                  'footer', 'page_number', 'footnote', 'formula', 'code', 'sidebar', 'reference')


class OcrRegion(BaseModel):
    """A word or line; score is a recognizer score, not factual confidence."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    text: str = Field(min_length=1, max_length=100_000)
    box: tuple[float, float, float, float]
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    block: int = Field(default=0, ge=0)
    #: The paragraph within ``block`` the engine read this in; 0 when it reports none.
    paragraph: int = Field(default=0, ge=0)
    line: int = Field(default=0, ge=0)
    #: What the page region holding this is, when a layout engine or the
    #: inferred labels say; None when nothing does, and left out when
    #: serialized, as regions stored before labels were read.
    label: RegionLabel | None = None

    @field_validator('box')
    @classmethod
    def valid_box(cls, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if (not all(math.isfinite(value) and 0 <= value <= 1 for value in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('OCR box must be a nonempty normalized rectangle')
        return box

    @model_serializer(mode='wrap')
    def omit_unreported_paragraph(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return _without_unreported_paragraph(self, handler(self))


def _without_unreported_paragraph(region: OcrRegion, value: dict[str, object]) -> dict[str, object]:
    # Regions stored before paragraphs were read, and engines that report none, serialize as they did.
    if region.paragraph == 0:
        value.pop('paragraph', None)
    if region.label is None:
        value.pop('label', None)
    return value


class OcrResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    engine: str = Field(min_length=1, max_length=96)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    regions: tuple[OcrRegion, ...] = Field(max_length=50_000)


class OrderedOcrRegion(OcrRegion):
    """Optional inferred column membership and original provider position."""
    provider_index: int | None = Field(default=None, ge=0, lt=50_000)
    reading_column: int | None = Field(default=None, ge=0, le=8)

    @model_validator(mode='after')
    def paired_order(self) -> OrderedOcrRegion:
        if (self.provider_index is None) != (self.reading_column is None):
            raise ValueError('OCR reading order requires both provider index and column')
        return self

    @model_serializer(mode='wrap')
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        value: dict[str, object] = handler(self)
        if self.provider_index is None:
            value.pop('provider_index', None)
            value.pop('reading_column', None)
        return _without_unreported_paragraph(self, value)


class OcrEngine(Protocol):
    async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                        max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult: ...


class LayoutRegion(BaseModel):
    """A region of a page as a layout engine sees it: a labelled rectangle,
    the engine's own reading position when it gives one, and its score,
    which is the engine's and not factual confidence."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    label: RegionLabel
    box: tuple[float, float, float, float]
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    order: int | None = Field(default=None, ge=0, lt=10_000)

    @field_validator('box')
    @classmethod
    def valid_box(cls, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if (not all(math.isfinite(value) and 0 <= value <= 1 for value in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('layout box must be a nonempty normalized rectangle')
        return box


class LayoutResult(BaseModel):
    """What a layout engine saw on one page image, in displayed image
    coordinates like an OCR result; ``dropped`` counts the regions whose
    label the engine gave in words this vocabulary has no name for."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    engine: str = Field(min_length=1, max_length=96)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    regions: tuple[LayoutRegion, ...] = Field(max_length=10_000)
    dropped: int = Field(default=0, ge=0)


class LayoutEngine(Protocol):
    """A page-layout analyser: given a page image, the labelled regions on
    it. Optional beside OCR, and never a source of text: what the engine
    labels is what the recognizer read there."""

    async def analyse(self, image: bytes, *, max_pixels: int = 20_000_000,
                      max_regions: int = 10_000, timeout_seconds: float = 30.0) -> LayoutResult: ...
