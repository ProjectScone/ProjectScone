"""Final episode-scope checks shared by passage and fact retrieval."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..core.models import Episode

if TYPE_CHECKING:
    from .filters import Filter


def episode_fits(episode: Episode, kind: Optional[str], source_prefix: Optional[str], since: Optional[str],
          until: Optional[str], conditions: Filter | None = None) -> bool:
    """The narrowing rule, shared with the Rust engine: kind equal, source
    starting with the prefix as literal text (an episode without a source
    matches no prefix), created_at inside the inclusive bounds.

    Metadata conditions are checked here as well as pushed into the store,
    and that is not redundant. The push-down keeps the store from spending
    its window on memories the caller excluded. This final check prevents
    out-of-scope answers even when a store cannot express the filter; it
    cannot recover eligible memories omitted from a candidate window."""
    if conditions is not None and not conditions.matches(episode.metadata):
        return False
    if kind is not None and episode.kind != kind:
        return False
    if source_prefix is not None and (episode.source is None or not episode.source.startswith(source_prefix)):
        return False
    if since is not None and episode.created_at < since:
        return False
    return not (until is not None and episode.created_at > until)
