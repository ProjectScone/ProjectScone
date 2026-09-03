"""Python client for the Scone HTTP API.

    from scone import Scone

    with Scone("http://127.0.0.1:7437", "sk-...") as memory:
        memory.add("deploys happen on Thursdays", tags=["ops"])
        for item in memory.recall("when do we deploy"):
            print(item.text)
"""

from .client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, Scone
from .errors import SconeError
from .models import Added, Fact, Memory, Profile, Recall, Status, Tag

__version__ = "0.2.1"

__all__ = [
    "Scone",
    "SconeError",
    "Added",
    "Fact",
    "Memory",
    "Profile",
    "Recall",
    "Status",
    "Tag",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "__version__",
]
