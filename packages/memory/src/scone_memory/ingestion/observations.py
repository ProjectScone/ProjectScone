"""Observation streams: frames seen over time, kept when they show something new.

A camera, a screen recorder or a wearable produces frames far faster than the scene changes. Storing every one
fills the disk with near-copies and buries the frame that mattered. An ``ObservationStream`` embeds each frame
once with the engine's image embedder and keeps it only when it differs from the frames kept lately, when it was
taken somewhere else than the last kept frame, or when nothing has been kept for a while. A kept frame is an image episode like any other (``ingestion.images``), dated
at the instant it was seen, so the image lane, recency, ``since`` and forgetting all apply to it unchanged.

``sightings`` answers the question a stream exists for: when and where was this last seen? It searches the
kept frames by a text query in the image embedder's space and returns the matches newest first.

Three things are deliberately the caller's:

- **The similarity that counts as a sighting.** A cosine between a text query and an image is on the embedder's
  own scale (CLIP checkpoints differ from one another), so ``min_similarity`` has no default.
- **Time order.** Frames arrive in the order they were seen; an earlier ``observed_at`` than the last is refused,
  because the gate's heartbeat is measured from the last kept frame.
- **The gate's memory.** The recently kept vectors live in the process. A new ``ObservationStream`` keeps its
  first frame whatever the old one saw.

Every bound says when it bit: a receipt names why a frame was kept or dropped, the stream counts both, and
``Sightings.window_full`` says the search could not see every stored frame.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from typing import TYPE_CHECKING, Literal, cast

from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import Attachment
from ..core.ports import ImageEmbedder
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import check_space
from ..retrieval.image_lane import ImageLane, search_images
from .image_context import ImageAttribute, ImageContext
from .images import ImageIngested, image_provenance, ingest_image

if TYPE_CHECKING:
    from ..core import forget_after as schedule
    from ..memory.engine import MemoryEngine

__all__ = ['DEFAULT_HEARTBEAT_SECONDS', 'DEFAULT_NOVELTY', 'DEFAULT_WINDOW', 'MAX_SIGHTING_WINDOW', 'ObservationStream',
           'Observed', 'SIGHTING_WINDOW', 'Sighting', 'Sightings', 'sightings']

#: A frame is new when it is at least this far (1 - cosine) from every recently kept frame.
DEFAULT_NOVELTY = 0.1
#: Recently kept frames a new one is compared with.
DEFAULT_WINDOW = 8
MAX_WINDOW = 64
#: A frame is kept after this long without one, however familiar it looks, so a still scene leaves a trace.
DEFAULT_HEARTBEAT_SECONDS = 60.0
#: Stored frames one ``sightings`` search looks at.
SIGHTING_WINDOW = 50
MAX_SIGHTING_WINDOW = 200
SOURCE_PREFIX = 'observation:'
_PLACE = 'place'
_OBSERVED_AT = 'observed_at'

MediaType = Literal['image/png', 'image/jpeg', 'image/webp']
Reason = Literal['first', 'novel', 'moved', 'heartbeat', 'seen_recently']


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise InvalidInput(f'{what} must be 1..256 characters')
    return value.strip()


def _tag(kind: str, name: str) -> str:
    # Engine tags are casefolded; hashing keeps "Kitchen" and "kitchen" apart, as entity tags do.
    return hashlib.sha256(f'observation-{kind}:{name}'.encode()).hexdigest()


#: On every kept frame of every stream.
_ANY = _tag('kind', 'frame')


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second))
    scale = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    return 0.0 if scale == 0.0 else dot / scale


def _instant(observed_at: object) -> datetime:
    if not isinstance(observed_at, str):
        raise InvalidInput('observed_at must be an RFC 3339 timestamp')
    try:
        return parse_rfc3339(observed_at)
    except ValueError:
        raise InvalidInput('observed_at must be an RFC 3339 timestamp') from None


@dataclass(frozen=True)
class Observed:
    """What the stream did with one frame."""

    kept: bool
    #: ``first``: nothing kept yet. ``novel``: far enough from every recent kept frame. ``moved``: familiar, but
    #: at another place than the last kept frame. ``heartbeat``: familiar, but the heartbeat was due.
    #: ``seen_recently``: familiar, at the same place and not due, so it was dropped.
    reason: Reason
    #: 1 - the highest cosine with a recently kept frame, never below 0; ``None`` for the first frame.
    novelty: float | None
    #: The stored image, when the frame was kept. Its ``image_lane`` says whether it can be found by sight.
    ingested: ImageIngested | None = None


@dataclass(frozen=True)
class Sighting:
    """One kept frame that matched a query."""

    episode_id: int
    chunk_id: int
    #: When the frame was seen.
    observed_at: str
    #: Cosine between the query and the frame in the image embedder's space. Not a probability.
    similarity: float
    stream: str
    place: str | None
    #: The retained frame.
    image: Attachment


@dataclass(frozen=True)
class Sightings:
    """Frames that matched, newest first, and what the search could not see."""

    items: tuple[Sighting, ...]
    #: Frames the search looked at.
    searched: int
    #: Of those, the frames under ``min_similarity``.
    below_threshold: int
    #: Matching frames left out because their stored image could not be resolved (forgotten since the search).
    unresolved: int
    #: The search filled its window, so stored frames beyond it were not compared: a newer sighting than
    #: ``latest`` may exist. Narrow by stream or place, or raise ``window``.
    window_full: bool

    @property
    def latest(self) -> Sighting | None:
        return self.items[0] if self.items else None

    @property
    def earliest(self) -> Sighting | None:
        return self.items[-1] if self.items else None


class ObservationStream:
    """One source of frames in one space. See the module."""

    def __init__(self, engine: MemoryEngine, space: str, stream: str, *, novelty: float = DEFAULT_NOVELTY,
                 window: int = DEFAULT_WINDOW, heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
                 forget_after: str | schedule.Resolved | None = None) -> None:
        check_space(space)
        if engine.image_embedder is None or engine.image_vectors is None:
            raise InvalidInput('an observation stream needs an engine with the image lane configured '
                               '(image_embedder and image_vectors)')
        if isinstance(novelty, bool) or not isinstance(novelty, (int, float)) or not 0.0 < novelty <= 2.0:
            raise InvalidInput('novelty must be above 0 and at most 2')
        if type(window) is not int or not 1 <= window <= MAX_WINDOW:
            raise InvalidInput(f'window must be an integer from 1 to {MAX_WINDOW}')
        if (isinstance(heartbeat_seconds, bool) or not isinstance(heartbeat_seconds, (int, float))
                or not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0):
            raise InvalidInput('heartbeat_seconds must be a positive number')
        self.engine = engine
        self.space = space
        self.stream = _name(stream, 'stream')
        self.novelty = float(novelty)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.forget_after = forget_after
        self._recent: deque[list[float]] = deque(maxlen=window)
        self._last_seen: datetime | None = None
        self._last_kept: datetime | None = None
        self._last_place: str | None = None
        #: Frames offered, kept and dropped by this object since it was made.
        self.seen = 0
        self.kept = 0
        self.dropped = 0

    def _decide(self, vector: Sequence[float], at: datetime, place: str | None) -> tuple[Reason, float | None]:
        if not self._recent or self._last_kept is None:
            return 'first', None
        novelty = max(0.0, 1.0 - max(_cosine(vector, kept) for kept in self._recent))
        if novelty >= self.novelty:
            return 'novel', novelty
        if place != self._last_place:
            # The same sight somewhere else is a new fact about where things are.
            return 'moved', novelty
        if (at - self._last_kept).total_seconds() >= self.heartbeat_seconds:
            return 'heartbeat', novelty
        return 'seen_recently', novelty

    async def observe(self, data: bytes, *, media_type: MediaType, observed_at: str, place: str | None = None,
                      caption: str | None = None,
                      caption_origin: Literal['supplied', 'model_generated'] = 'supplied') -> Observed:
        """Offer one frame. It is embedded once; a kept frame is stored with that vector.

        ``place`` is wherever the caller says the frame was taken (a room, a waypoint, a camera name).
        ``caption`` is words about the frame, found by the text lanes; say ``model_generated`` when a model
        wrote them."""
        at = _instant(observed_at)
        if self._last_seen is not None and at < self._last_seen:
            raise InvalidInput('frames must be observed in time order')
        where = None if place is None else _name(place, 'place')
        if caption is not None and (not isinstance(caption, str) or not caption.strip()):
            raise InvalidInput('caption must be text')
        # The lane was checked when the stream was made.
        [vector] = await cast(ImageEmbedder, self.engine.image_embedder).embed_images([data])
        reason, novelty = self._decide(vector, at, where)
        self.seen += 1
        self._last_seen = at
        if reason == 'seen_recently':
            self.dropped += 1
            return Observed(False, reason, novelty)
        stamp = format_rfc3339(at)
        attributes = [ImageAttribute(kind='metadata', name=_OBSERVED_AT, value=stamp)]
        if where is not None:
            attributes.append(ImageAttribute(kind='metadata', name=_PLACE, value=where))
        if caption is not None:
            attributes.append(ImageAttribute(kind='caption', value=caption.strip(), origin=caption_origin))
        context = ImageContext(source=SOURCE_PREFIX + self.stream, locator=stamp, attributes=tuple(attributes))
        tags = [_ANY, _tag('stream', self.stream)] + ([] if where is None else [_tag('place', where)])
        ingested = await ingest_image(self.engine, self.space, data, media_type=media_type, context=context,
                                      forget_after=self.forget_after, observed_at=stamp, tags=tags,
                                      image_vector=vector)
        self._recent.append(list(vector))
        self._last_kept = at
        self._last_place = where
        self.kept += 1
        return Observed(True, reason, novelty, ingested)


async def sightings(engine: MemoryEngine, space: str, query: str, *, min_similarity: float,
                    stream: str | None = None, place: str | None = None,
                    window: int = SIGHTING_WINDOW) -> Sightings:
    """Kept frames that look like ``query``, newest first. ``Sightings.latest`` is "last seen".

    The search takes the ``window`` stored frames nearest the query and keeps those at or above
    ``min_similarity``; ``stream`` and ``place`` narrow it to one source or one place. See the module for why
    the threshold has no default."""
    check_space(space)
    if engine.image_embedder is None or engine.image_vectors is None:
        raise InvalidInput('sightings need an engine with the image lane configured')
    if not isinstance(query, str) or not query.strip():
        raise InvalidInput('query must be text')
    if (isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float))
            or not -1.0 <= min_similarity <= 1.0):
        raise InvalidInput('min_similarity must be a cosine from -1 to 1')
    if type(window) is not int or not 1 <= window <= MAX_SIGHTING_WINDOW:
        raise InvalidInput(f'window must be an integer from 1 to {MAX_SIGHTING_WINDOW}')
    # Every kept frame carries ``_ANY``, so images stored some other way never take a place in the window.
    tags = (_ANY, *(_tag(kind, _name(value, kind)) for kind, value in (('stream', stream), ('place', place))
                    if value is not None))
    found = await search_images(ImageLane(engine.image_embedder, engine.image_vectors), engine.documents, space,
                                query.strip(), window, None, tags, {'document_format': 'image'})
    matching = [(chunk_id, cosine) for chunk_id, cosine in found.hits if cosine >= min_similarity]
    chunks = {chunk.chunk_id: chunk for chunk in await engine.documents.get_chunks(space, [c for c, _ in matching])}
    items: list[Sighting] = []
    unresolved = 0
    for chunk_id, cosine in matching:
        chunk = chunks.get(chunk_id)
        try:
            if chunk is None:
                raise NotFound('the frame is no longer stored')
            provenance = await image_provenance(engine, space, chunk.episode_id)
        except (InvalidInput, NotFound, Gone):
            unresolved += 1
            continue
        context = provenance.context
        if not context.source.startswith(SOURCE_PREFIX):
            unresolved += 1  # tagged as a frame but not stored as one
            continue
        where = next((a.value for a in context.attributes if a.kind == 'metadata' and a.name == _PLACE), None)
        items.append(Sighting(chunk.episode_id, chunk_id, chunk.created_at, cosine,
                              context.source.removeprefix(SOURCE_PREFIX), where, provenance.image))
    items.sort(key=lambda sighting: (parse_rfc3339(sighting.observed_at), sighting.episode_id), reverse=True)
    return Sightings(tuple(items), len(found.hits), len(found.hits) - len(matching), unresolved,
                     found.returned >= window)
