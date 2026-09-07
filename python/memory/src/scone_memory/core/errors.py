class SconeError(Exception):
    """Base for everything the engine raises on purpose."""


class InvalidInput(SconeError):
    """The caller sent something the engine refuses: bad space name,
    empty content, an unparsable date. Maps to HTTP 422."""


class NotFound(SconeError):
    """The id names nothing in this space. Maps to HTTP 404."""


class Gone(NotFound):
    """The id named an episode that was forgotten on purpose; the tombstone
    says when. A kind of NotFound for every caller that only asks whether
    there is something to read. Maps to HTTP 410."""

    def __init__(self, message: str, forgotten_at: str):
        super().__init__(message)
        self.forgotten_at = forgotten_at


class Conflict(SconeError):
    """The space moved since the caller read it, so a decision made
    against that reading is refused whole. Carries the current revision so
    the caller can re-read rather than guess. Maps to HTTP 409."""

    def __init__(self, message: str, revision: int) -> None:
        super().__init__(message)
        self.revision = revision
