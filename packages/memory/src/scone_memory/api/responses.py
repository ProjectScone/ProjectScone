"""JSON responses that can carry whatever the ledger holds.

The ledger accepts any Python string, and a lone surrogate (U+D800 to
U+DFFF on its own) is one UTF-8 cannot encode. A plain JSON response fails
on it with a 500 the first time a stored name holds one. This response
writes such a character as its JSON escape (``\\ud800``) instead: valid
JSON that a client reads back as the same text.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse
from pydantic_core import PydanticSerializationError, to_json


class LedgerJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # The compiled encoder is quick and writes the same bytes, but
        # refuses a lone surrogate. Only then is the answer written again.
        try:
            return to_json(content)
        except PydanticSerializationError:
            pass
        # Outside strings JSON is ASCII, so the only characters UTF-8 can
        # refuse here sit inside a string, where a backslash escape is JSON's.
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None,
                          separators=(",", ":")).encode("utf-8", "backslashreplace")
