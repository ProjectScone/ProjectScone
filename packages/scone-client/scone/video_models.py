"""Immutable video evidence with exact clocks and source-bound observations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import json
import re
from typing import Optional

from ._wire import boolean, digest, identifier, integer, invalid, items, record, text
from .document_models import DocumentAttachment


def characters(value: object, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum or '\0' in value:
        raise invalid('video text')
    return text(value, maximum * 4, 'video text')


def clock(value: object, minimum: int = -(2**63)) -> int:
    if not isinstance(value, str) or len(value) > 20 or re.fullmatch(r'0|-?[1-9][0-9]*', value) is None:
        raise invalid('video decimal clock')
    return integer(int(value), minimum)


def ratio(value: object) -> tuple[str, Fraction]:
    label = text(value, 21, 'video time base')
    if re.fullmatch(r'[1-9][0-9]{0,9}/[1-9][0-9]{0,9}', label) is None:
        raise invalid('video time base')
    numerator, denominator = label.split('/')
    return label, Fraction(int(numerator), int(denominator))


def unit(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
        raise invalid('video region coordinate')
    return float(value)


@dataclass(frozen=True)
class VideoRegion:
    text: str
    start: int
    end: int
    box: tuple[float, float, float, float]
    score: Optional[float]


@dataclass(frozen=True)
class VideoFrame:
    ordinal: int
    presentation_timestamp: int
    requested_seconds: tuple[int, ...]
    width: int
    height: int
    png_sha256: str
    png_bytes: int
    ocr_engine: str
    empty: bool
    text: str = ''
    regions: tuple[VideoRegion, ...] = ()


def _regions(value: object, content: str) -> tuple[VideoRegion, ...]:
    encoded = content.encode('utf-8')
    result: list[VideoRegion] = []
    previous = 0
    for item in items(value, 20000):
        row = record(item)
        start, end = integer(row.get('start'), previous, len(encoded)), integer(row.get('end'), 1, len(encoded))
        label = characters(row.get('text'), 100000)
        try:
            matches = encoded[start:end].decode('utf-8') == label and not encoded[previous:start].decode('utf-8').strip()
        except UnicodeError:
            matches = False
        box = tuple(unit(number) for number in items(row.get('box'), 4))
        if (end <= start or not matches or row.get('coordinate_space') != 'normalized_displayed_frame_top_left'
                or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]):
            raise invalid('video region binding')
        result.append(VideoRegion(label, start, end, (box[0], box[1], box[2], box[3]),
                                  None if row.get('score') is None else unit(row['score'])))
        previous = end
    if not result or encoded[previous:].decode('utf-8').strip():
        raise invalid('video regions do not cover the recognized text')
    return tuple(result)


@dataclass(frozen=True)
class VideoCatalogue:
    space: str
    episode_id: int
    original: DocumentAttachment
    manifest: DocumentAttachment
    filename: str
    format: str
    decoder_revision: str
    policy_revision: str
    model_revision: str
    stream_index: int
    time_base: str
    start_timestamp: int
    duration_ticks: int
    interval_seconds: int
    decoded_frames: int
    unavailable_requests: int
    frames: tuple[VideoFrame, ...]
    _encoded: bytes = field(repr=False)

    def frame(self, ordinal: int) -> VideoFrame:
        integer(ordinal, 0, 99999)
        for frame in self.frames:
            if frame.ordinal == ordinal:
                return frame
        raise invalid('frame is not retained in this catalogue')

    def frame_time(self, ordinal: int) -> Fraction:
        return (self.frame(ordinal).presentation_timestamp - self.start_timestamp) * ratio(self.time_base)[1]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, episode_id: int) -> VideoCatalogue:
        envelope = record(value)
        integer(episode_id, 1)
        if (integer(envelope.get('schema_version'), 1, 1) != 1 or envelope.get('space') != identifier(expected_space)
                or envelope.get('episode_id') != str(episode_id) or envelope.get('timestamp_encoding') != 'decimal-string'):
            raise invalid('video catalogue source')
        evidence = record(envelope.get('evidence'))
        original, manifest = DocumentAttachment.from_json(evidence.get('original')), DocumentAttachment.from_json(evidence.get('manifest'))
        if (original.attachment_id == manifest.attachment_id or manifest.media_type != 'application/json'
                or evidence.get('parser') != 'video-frame-ocr' or evidence.get('download_path') != '/v1/attachments/' + original.attachment_id):
            raise invalid('video document binding')
        video = record(evidence.get('video'))
        if digest(video.get('source_sha256')) != original.attachment_id:
            raise invalid('video original hash')
        policy = record(video.get('policy'))
        interval = integer(policy.get('interval_seconds'), 1, 120)
        max_frames = integer(policy.get('max_frames'), 1, 256)
        max_seconds = integer(policy.get('max_duration_seconds'), 1, 600)
        max_pixels = integer(policy.get('max_pixels'), 1, 20000000)
        max_bytes = integer(policy.get('max_frame_bytes'), 1, 10000000)
        max_total = integer(policy.get('max_total_bytes'), 1, 64000000)
        time_base, tick = ratio(video.get('time_base'))
        start, duration = clock(video.get('start_timestamp')), clock(video.get('duration_ticks'), 1)
        if duration * tick > max_seconds:
            raise invalid('video duration')
        expected = tuple(second for second in range(0, max_seconds, interval) if second < duration * tick)
        decoded, unavailable = integer(video.get('decoded_frames'), 1, 100000), integer(video.get('unavailable_requests'), 0, 600)
        frames: list[VideoFrame] = []
        covered: list[int] = []
        for item in items(video.get('frames'), max_frames):
            row = record(item)
            selected = tuple(integer(second, 0, 599) for second in items(row.get('requested_seconds'), 600))
            frame = VideoFrame(integer(row.get('ordinal'), 0, decoded-1), clock(row.get('presentation_timestamp')), selected,
                integer(row.get('width'), 1, 100000), integer(row.get('height'), 1, 100000), digest(row.get('png_sha256')),
                integer(row.get('png_bytes'), 1, max_bytes), characters(row.get('ocr_engine'), 96), boolean(row.get('empty')))
            if (not selected or frame.width * frame.height > max_pixels
                    or not selected[-1] <= (frame.presentation_timestamp-start)*tick < duration*tick
                    or (frames and (frame.ordinal <= frames[-1].ordinal or frame.presentation_timestamp <= frames[-1].presentation_timestamp))):
                raise invalid('video frame inventory')
            covered.extend(selected)
            frames.append(frame)
        if not frames or sum(frame.png_bytes for frame in frames) > max_total or tuple(covered) != expected[:len(covered)] or len(covered)+unavailable != len(expected):
            raise invalid('video sampling coverage')
        stream = integer(video.get('stream_index'), 0, 100000)
        segments = items(evidence.get('segments'), 20000)
        by_ordinal = {frame.ordinal: index for index, frame in enumerate(frames)}
        seen: set[int] = set()
        total_regions = total_text = 0
        for item in segments:
            row = record(item)
            metadata = record(row.get('metadata'))
            ordinal = clock(metadata.get('video_frame_ordinal'), 0)
            index = by_ordinal.get(ordinal)
            if index is None or ordinal in seen:
                raise invalid('video text frame')
            frame = frames[index]
            if (frame.empty or row.get('locator') != f'video:stream:{stream}/frame:{ordinal}'
                    or metadata.get('extraction') != 'ocr' or metadata.get('engine') != frame.ocr_engine):
                raise invalid('video text binding')
            content = text(row.get('text'), 2000000)
            regions = _regions(row.get('regions'), content)
            total_regions += len(regions)
            total_text += len(content.encode('utf-8'))
            if total_regions > 20000 or total_text > 2000000:
                raise invalid('video text limit')
            frames[index] = replace(frame, text=content, regions=regions)
            seen.add(ordinal)
        if any(not frame.empty and frame.ordinal not in seen for frame in frames):
            raise invalid('video OCR text missing')
        model = text(video.get('model_revision'), 128)
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', model) is None:
            raise invalid('video OCR revision')
        try:
            encoded = json.dumps(envelope, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise invalid('video catalogue encoding') from None
        return cls(expected_space, episode_id, original, manifest, characters(evidence.get('filename'), 1024), text(evidence.get('format'), 32),
                   digest(video.get('decoder_revision')), characters(video.get('policy_revision'), 96), model, stream, time_base,
                   start, duration, interval, decoded, unavailable, tuple(frames), encoded)


@dataclass(frozen=True)
class VideoInterpretation:
    text: str
    model: str
    frame: VideoFrame
    source: VideoCatalogue
    persisted: bool = False

    @classmethod
    def from_json(cls, value: object, source: VideoCatalogue, ordinal: int) -> VideoInterpretation:
        body = record(value)
        observed, result = record(body.get('frame')), record(body.get('understanding'))
        frame = source.frame(ordinal)
        if (integer(body.get('schema_version'), 1, 1) != 1 or body.get('space') != source.space
                or body.get('episode_id') != str(source.episode_id) or body.get('persisted') is not False
                or body.get('original_sha256') != source.original.attachment_id or body.get('manifest_sha256') != source.manifest.attachment_id
                or integer(observed.get('ordinal'), 0, 99999) != ordinal or observed.get('presentation_timestamp') != str(frame.presentation_timestamp)
                or observed.get('time_base') != source.time_base or observed.get('png_sha256') != frame.png_sha256
                or result.get('attachment_id') != frame.png_sha256 or result.get('origin') != 'model_generated'
                or result.get('source') != f'video:episode:{source.episode_id}/stream:{source.stream_index}/frame:{ordinal}'
                or result.get('media_type') != 'image/png'):
            raise invalid('video interpretation binding')
        for row in (observed, result):
            if integer(row.get('width'), 1, 100000) != frame.width or integer(row.get('height'), 1, 100000) != frame.height:
                raise invalid('video interpretation dimensions')
        return cls(characters(result.get('text'), 64000), characters(result.get('model'), 256), frame, source)
