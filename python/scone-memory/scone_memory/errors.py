class SconeError(Exception):
    """Base for everything the engine raises on purpose."""


class InvalidInput(SconeError):
    """The caller sent something the engine refuses: bad space name,
    empty content, an unparsable date. Maps to HTTP 422."""


class NotFound(SconeError):
    """The id names nothing in this space. Maps to HTTP 404."""
