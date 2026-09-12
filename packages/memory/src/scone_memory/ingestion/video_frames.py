"""Deterministic video frame sampling from decoded presentation timestamps.

This foundation performs no OCR, inference, storage or document ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from time import monotonic
import zlib

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import InvalidInput
from ..ocr.process import python_worker, run_bounded
from .formats.media import VIDEO_EXTENSIONS, _input, _remaining
from .formats.types import DocumentLimits

_POLICY_REVISION = 'decoded-video-frames-v1'
_DEMUXERS = 'mov,matroska,webm,avi,mpeg,mpegts'
_MAX_INVENTORY_BYTES = 16_000_000
_MAX_DECODED_FRAMES = 100_000
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class VideoFramePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    interval_seconds: int = Field(default=5, ge=1, le=120)
    max_frames: int = Field(default=64, ge=1, le=256)
    max_duration_seconds: int = Field(default=600, ge=1, le=600)
    max_pixels: int = Field(default=20_000_000, ge=1, le=20_000_000)
    max_frame_bytes: int = Field(default=10_000_000, ge=1, le=10_000_000)
    max_total_bytes: int = Field(default=32_000_000, ge=1, le=64_000_000)


@dataclass(frozen=True)
class FrameSelection:
    ordinal: int
    presentation_timestamp: int
    requested_seconds: tuple[int, ...]


@dataclass(frozen=True)
class FramePlan:
    stream_index: int
    time_base: Fraction
    start_timestamp: int
    duration: Fraction
    decoded_frames: int
    unavailable_requests: int
    frames: tuple[FrameSelection, ...]


@dataclass(frozen=True)
class VideoFrame:
    selection: FrameSelection
    width: int
    height: int
    sha256: str
    png: bytes


@dataclass(frozen=True)
class VideoFrames:
    source_sha256: str
    decoder_revision: str
    policy_revision: str
    policy: VideoFramePolicy
    plan: FramePlan
    frames: tuple[VideoFrame, ...]


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InvalidInput('video inventory must contain objects')
    return value


def _integer(value: object, label: str, *, minimum: int = -(2**63)) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise InvalidInput(f'video {label} must be a bounded integer')
    return value


def _dimensions(item: dict[str, object], policy: VideoFramePolicy) -> None:
    width = _integer(item.get('width'), 'width', minimum=1)
    height = _integer(item.get('height'), 'height', minimum=1)
    if width * height > policy.max_pixels:
        raise InvalidInput('video exceeds its frame pixel limit')


def _stream(value: object, policy: VideoFramePolicy) -> tuple[int, Fraction, int | None, int | None]:
    root = _object(value)
    streams = root.get('streams')
    if not isinstance(streams, list) or len(streams) != 1:
        raise InvalidInput('video requires one selected non-attached video stream')
    stream = _object(streams[0])
    _dimensions(stream, policy)
    index = _integer(stream.get('index'), 'stream index', minimum=0)
    raw_base = stream.get('time_base')
    if not isinstance(raw_base, str) or not re.fullmatch(r'[1-9][0-9]{0,9}/[1-9][0-9]{0,9}', raw_base):
        raise InvalidInput('video time base must be a positive rational')
    base = Fraction(raw_base)
    start = _integer(stream['start_pts'], 'start timestamp') if 'start_pts' in stream else None
    duration = _integer(stream['duration_ts'], 'duration', minimum=1) if 'duration_ts' in stream else None
    if duration is not None and duration * base > policy.max_duration_seconds:
        raise InvalidInput('video exceeds its duration limit')
    return index, base, start, duration


def _json(data: bytes) -> dict[str, object]:
    if not data or len(data) > _MAX_INVENTORY_BYTES:
        raise InvalidInput('video inventory exceeds its byte limit')
    try:
        return _object(json.loads(data))
    except (ValueError, RecursionError) as error:
        raise InvalidInput('video inventory is malformed') from error


def _validated_policy(policy: VideoFramePolicy) -> VideoFramePolicy:
    if not isinstance(policy, VideoFramePolicy):
        raise InvalidInput('video policy must be a VideoFramePolicy')
    try:
        return VideoFramePolicy.model_validate(policy.model_dump(mode='python'))
    except (ValueError, ValidationError) as error:
        raise InvalidInput('video policy exceeds its declared limits') from error


def _validated_limits(limits: DocumentLimits) -> DocumentLimits:
    if not isinstance(limits, DocumentLimits):
        raise InvalidInput('video limits must be DocumentLimits')
    try:
        return DocumentLimits.model_validate(limits.model_dump(mode='python'))
    except (ValueError, ValidationError) as error:
        raise InvalidInput('video limits exceed their declared bounds') from error


def plan_frames(data: bytes, policy: VideoFramePolicy) -> FramePlan:
    """Plan from ffprobe's decoded inventory; missing PTS are never guessed."""
    policy = _validated_policy(policy)
    root = _json(data)
    stream, base, origin, declared_duration = _stream(root, policy)
    frames = root.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= _MAX_DECODED_FRAMES:
        raise InvalidInput('video decoded frame count is empty or exceeds its limit')
    timestamps: list[int] = []
    ends: list[int] = []
    for value in frames:
        frame = _object(value)
        if _integer(frame.get('stream_index'), 'frame stream', minimum=0) != stream:
            raise InvalidInput('video frame belongs to a different stream')
        _dimensions(frame, policy)
        pts = _integer(frame.get('best_effort_timestamp'), 'presentation timestamp')
        if timestamps and pts <= timestamps[-1]:
            raise InvalidInput('video presentation timestamps must strictly increase')
        frame_duration = _integer(frame.get('duration', frame.get('pkt_duration', 0)), 'frame duration', minimum=0)
        timestamps.append(pts)
        ends.append(pts + frame_duration)
    start = timestamps[0] if origin is None else origin
    if timestamps[0] < start:
        raise InvalidInput('video frame precedes its declared stream start')
    duration = max(Fraction(max(ends) - start) * base,
                   Fraction(declared_duration or 0) * base)
    if not 0 < duration <= policy.max_duration_seconds:
        raise InvalidInput('video duration is missing or exceeds its limit')
    choices: list[FrameSelection] = []
    cursor, unavailable = 0, 0
    for requested in range(0, policy.max_duration_seconds, policy.interval_seconds):
        if requested >= duration:
            break
        while cursor < len(timestamps) and (timestamps[cursor] - start) * base < requested:
            cursor += 1
        if cursor == len(timestamps):
            unavailable += 1
            continue
        if choices and choices[-1].ordinal == cursor:
            previous = choices[-1]
            choices[-1] = FrameSelection(cursor, timestamps[cursor], (*previous.requested_seconds, requested))
        else:
            choices.append(FrameSelection(cursor, timestamps[cursor], (requested,)))
            if len(choices) > policy.max_frames:
                raise InvalidInput('video selected frame count exceeds its limit')
    return FramePlan(stream, base, start, duration, len(frames), unavailable, tuple(choices))


def _executable(path: str) -> tuple[str, str]:
    resolved = Path(path)
    if not resolved.is_absolute() or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise InvalidInput('video requires explicitly configured executable paths')
    resolved = resolved.resolve()
    with resolved.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return str(resolved), digest


def _pngs(data: bytes, policy: VideoFramePolicy) -> tuple[bytes, ...]:
    """Split a bounded image2pipe stream; check framing and CRC before decoding."""
    if len(data) > policy.max_total_bytes:
        raise InvalidInput('video frames exceed their total encoded byte limit')
    result: list[bytes] = []
    offset = 0
    while offset < len(data):
        start = offset
        if data[offset:offset + 8] != _PNG_SIGNATURE:
            raise InvalidInput('video decoder returned invalid PNG framing')
        offset += 8
        first = True
        while True:
            if offset + 12 > len(data):
                raise InvalidInput('video decoder returned an incomplete PNG')
            length = struct.unpack_from('>I', data, offset)[0]
            end = offset + 12 + length
            if end > len(data) or end - start > policy.max_frame_bytes:
                raise InvalidInput('video frame exceeds its encoded byte limit or is incomplete')
            kind = data[offset + 4:offset + 8]
            if first and (kind != b'IHDR' or length != 13):
                raise InvalidInput('video frame is missing its PNG header')
            if zlib.crc32(data[offset + 4:end - 4]) != struct.unpack_from('>I', data, end - 4)[0]:
                raise InvalidInput('video frame PNG checksum mismatch')
            if first:
                width, height = struct.unpack_from('>II', data, offset + 8)
                _dimensions({'width': width, 'height': height}, policy)
            first = False
            offset = end
            if kind == b'IEND':
                if length != 0:
                    raise InvalidInput('video PNG end marker is malformed')
                break
        result.append(data[start:offset])
        if len(result) > policy.max_frames:
            raise InvalidInput('video decoder exceeded its selected frame count')
    return tuple(result)


class VideoFrameDecoder:
    """Caller-owned local decoder; sampling does not configure document ingestion."""
    def __init__(self, *, ffmpeg_path: str, ffprobe_path: str) -> None:
        self.ffmpeg, self._ffmpeg_digest = _executable(ffmpeg_path)
        self.ffprobe, self._ffprobe_digest = _executable(ffprobe_path)

    async def sample(self, data: bytes, filename: str, *, policy: VideoFramePolicy | None = None,
                     limits: DocumentLimits | None = None) -> VideoFrames:
        policy = _validated_policy(VideoFramePolicy() if policy is None else policy)
        limits = _validated_limits(DocumentLimits() if limits is None else limits)
        suffix = _input(data, filename, limits, VIDEO_EXTENSIONS - {'.ts'})
        deadline = monotonic() + limits.timeout_seconds
        versions = []
        for executable, digest in ((self.ffmpeg, self._ffmpeg_digest), (self.ffprobe, self._ffprobe_digest)):
            if _executable(executable)[1] != digest:
                raise InvalidInput('configured video decoder changed; construct a new decoder explicitly')
            version = await run_bounded([executable, '-version'], b'', timeout=_remaining(deadline),
                                         max_output=32000, label='video decoder identity')
            versions.append(digest.encode() + b'\n' + version)
        revision = hashlib.sha256(b'\0'.join(versions)).hexdigest()
        with tempfile.TemporaryDirectory(prefix='scone-video-') as folder:
            path = Path(folder) / ('source.' + suffix)
            path.write_bytes(data)
            common = ['-v', 'error', '-max_alloc', '134217728', '-protocol_whitelist', 'file,pipe',
                      '-format_whitelist', _DEMUXERS]
            probe = [self.ffprobe, *common, '-select_streams', 'V:0']
            entries = 'stream=index,time_base,start_pts,duration_ts,width,height'
            preflight = await run_bounded([*probe, '-show_streams', '-show_entries', entries, '-of', 'json', str(path)],
                b'', timeout=_remaining(deadline), max_output=65536, label='video stream inspection')
            _stream(_json(preflight), policy)
            inventory = await run_bounded([*probe, '-show_streams', '-show_frames', '-show_entries',
                entries + ':frame=stream_index,best_effort_timestamp,duration,pkt_duration,width,height', '-of', 'json', str(path)],
                b'', timeout=_remaining(deadline), max_output=_MAX_INVENTORY_BYTES, label='video frame inventory')
            plan = plan_frames(inventory, policy)
            selection = '+'.join(f'eq(n\\,{frame.ordinal})' for frame in plan.frames)
            encoded = await run_bounded([self.ffmpeg, *common, '-threads', '1', '-i', str(path),
                '-map', f'0:{plan.stream_index}', '-an', '-sn', '-dn', '-vf', 'select=' + selection,
                '-fps_mode', 'passthrough', '-threads', '1', '-c:v', 'png', '-f', 'image2pipe', 'pipe:1'],
                b'', timeout=_remaining(deadline), max_output=policy.max_total_bytes, label='video frame decoding')
        pngs = _pngs(encoded, policy)
        if len(pngs) != len(plan.frames):
            raise InvalidInput('video decoded frames do not match their timestamp inventory')
        frames = []
        for selected, png in zip(plan.frames, pngs, strict=True):
            checked = await run_bounded(python_worker('scone_memory.ingestion._image_worker', 'image/png'), png,
                timeout=_remaining(deadline), max_output=4096, label='video frame validation')
            dimensions = _json(checked)
            _dimensions(dimensions, policy)
            width = _integer(dimensions.get('width'), 'width', minimum=1)
            height = _integer(dimensions.get('height'), 'height', minimum=1)
            frames.append(VideoFrame(selected, width, height, hashlib.sha256(png).hexdigest(), png))
        _remaining(deadline)
        return VideoFrames(hashlib.sha256(data).hexdigest(), revision, _POLICY_REVISION, policy, plan, tuple(frames))
