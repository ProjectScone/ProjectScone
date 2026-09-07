"""Validated, immutable recall constraints for sessions and integrations."""

from collections.abc import Mapping
from dataclasses import dataclass

from .engine import KINDS, MAX_SOURCE, normalise_metadata, normalise_time
from .errors import InvalidInput


@dataclass(frozen=True)
class RecallScope:
    where: tuple[tuple[str, str], ...]
    kind: str | None
    source_prefix: str | None
    since: str | None
    until: str | None

    @classmethod
    def validated(cls, *, where=None, kind=None, source_prefix=None, since=None, until=None):
        if where is not None and (not isinstance(where, Mapping) or any(not isinstance(k, str) for k in where)):
            raise ValueError("where must be a string-to-string metadata mapping")
        if kind is not None and (not isinstance(kind, str) or kind not in KINDS):
            raise ValueError("kind must name a supported episode kind")
        if source_prefix is not None and (not isinstance(source_prefix, str) or len(source_prefix) > MAX_SOURCE):
            raise ValueError(f"source_prefix must be a string of at most {MAX_SOURCE} characters")
        for name, value in (("since", since), ("until", until)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty RFC3339 timestamp")
        try:
            clean = normalise_metadata(where if where is not None else {})
            start = normalise_time(since) if since is not None else None
            end = normalise_time(until) if until is not None else None
        except InvalidInput as exc:
            raise ValueError(str(exc)) from exc
        if start is not None and end is not None and start > end:
            raise ValueError("since must not be later than until")
        return cls(tuple(sorted(clean.items())), kind, source_prefix, start, end)

    def kwargs(self):
        # Give each request its own mapping; caller or engine mutations cannot
        # change the fixed scope used by later turns.
        return {"where": dict(self.where), "kind": self.kind,
                "source_prefix": self.source_prefix, "since": self.since, "until": self.until}

    def as_dict(self):
        """Canonical JSON shape, omitting constraints that do not narrow recall."""
        # An empty source prefix still excludes episodes with no source.
        return {key: value for key, value in self.kwargs().items()
                if value is not None and value != {}}

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            value = {}
        if not isinstance(value, Mapping) or set(value) - {"where", "kind", "source_prefix", "since", "until"}:
            raise InvalidInput("recall_scope must contain only where, kind, source_prefix, since and until")
        try:
            return cls.validated(**value)
        except ValueError as exc:
            raise InvalidInput(str(exc)) from exc
