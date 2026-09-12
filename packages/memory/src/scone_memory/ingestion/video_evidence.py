"""Typed evidence for sampled video pixels and their OCR observations."""
from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VideoFramePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    interval_seconds: int = Field(default=5, ge=1, le=120)
    max_frames: int = Field(default=64, ge=1, le=256)
    max_duration_seconds: int = Field(default=600, ge=1, le=600)
    max_pixels: int = Field(default=20_000_000, ge=1, le=20_000_000)
    max_frame_bytes: int = Field(default=10_000_000, ge=1, le=10_000_000)
    max_total_bytes: int = Field(default=32_000_000, ge=1, le=64_000_000)


class VideoOcrFrame(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    ordinal: int = Field(ge=0, lt=100_000)
    presentation_timestamp: int = Field(ge=-(2**63), lt=2**63)
    requested_seconds: tuple[int, ...] = Field(min_length=1, max_length=600)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    png_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    png_bytes: int = Field(ge=1, le=10_000_000)
    ocr_engine: str = Field(min_length=1, max_length=96)
    empty: bool


class DocumentVideoEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    decoder_revision: str = Field(pattern=r'^[a-f0-9]{64}$')
    policy_revision: str = Field(min_length=1, max_length=96)
    model_revision: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
    policy: VideoFramePolicy
    stream_index: int = Field(ge=0, le=100_000)
    time_base: str = Field(pattern=r'^[1-9][0-9]{0,9}/[1-9][0-9]{0,9}$')
    start_timestamp: int = Field(ge=-(2**63), lt=2**63)
    duration_ticks: int = Field(ge=1, lt=2**63)
    decoded_frames: int = Field(ge=1, le=100_000)
    unavailable_requests: int = Field(ge=0, le=600)
    frames: tuple[VideoOcrFrame, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode='after')
    def consistent(self) -> Self:
        base = Fraction(self.time_base)
        duration = self.duration_ticks * base
        if duration > self.policy.max_duration_seconds or len(self.frames) > self.policy.max_frames:
            raise ValueError('video evidence exceeds its sampling policy')
        grid = tuple(range(0, self.policy.max_duration_seconds, self.policy.interval_seconds))
        requested = tuple(one for one in grid if one < duration)
        covered = tuple(one for frame in self.frames for one in frame.requested_seconds)
        if covered != requested[:len(covered)] or len(covered) + self.unavailable_requests != len(requested):
            raise ValueError('video evidence does not account for its sampling requests')
        last_ordinal, last_pts, total = -1, None, 0
        for frame in self.frames:
            relative = (frame.presentation_timestamp - self.start_timestamp) * base
            if (not last_ordinal < frame.ordinal < self.decoded_frames
                    or (last_pts is not None and frame.presentation_timestamp <= last_pts)
                    or relative < max(frame.requested_seconds) or relative >= duration
                    or frame.width * frame.height > self.policy.max_pixels
                    or frame.png_bytes > self.policy.max_frame_bytes):
                raise ValueError('video frame evidence is inconsistent')
            last_ordinal, last_pts = frame.ordinal, frame.presentation_timestamp
            total += frame.png_bytes
        if total > self.policy.max_total_bytes:
            raise ValueError('video frame bytes exceed their total limit')
        return self
