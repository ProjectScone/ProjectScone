"""Python client for the Scone HTTP API.

    from scone import Scone

    with Scone("http://127.0.0.1:7437", "sk-...") as memory:
        memory.add("deploys happen on Thursdays", tags=["ops"])
        for item in memory.recall("when do we deploy"):
            print(item.text)
"""

from ._wire import Capabilities
from .agent_inputs import InputRecord
from .agent_results import AgentResult, HandoffHop, HandoffResult, HumanOutput, ModelOutput, TaskResult
from .agent_evidence import EvidencePacket
from .agents import AgentClient, PlanPage, RunPage
from .agent_models import (AgentChoice, HandoffAgent, HandoffPlan, HumanInput, ModelChoice, ModelTask,
                           RunPolicy, RunRequest, RunStatus, SavedPlan, TaskPlan)
from .client import DEFAULT_BASE_URL, DEFAULT_MAX_RESPONSE_BYTES, DEFAULT_TIMEOUT, Scone
from .document_jobs import DocumentJobs, DocumentPage
from .document_models import (DocumentAttachment, DocumentFormat, DocumentFormats, DocumentRequest, DocumentResult, DocumentSpec,
                              DocumentStatus, DocumentStored, ParserLimits, PdfOcr)
from .errors import SconeError
from .models import Added, Fact, Memory, Profile, Recall, Status, Tag

__version__ = "0.2.1"

__all__ = [
    "Scone",
    "Capabilities",
    "AgentClient",
    "InputRecord",
    "AgentResult", "HandoffHop", "HandoffResult", "HumanOutput", "ModelOutput", "TaskResult", "EvidencePacket",
    "DocumentJobs", "DocumentPage", "DocumentFormat", "DocumentFormats", "DocumentAttachment", "DocumentRequest", "DocumentResult",
    "DocumentSpec", "DocumentStatus", "DocumentStored", "ParserLimits", "PdfOcr",
    "PlanPage", "RunPage", "AgentChoice", "HandoffAgent", "HandoffPlan", "HumanInput",
    "ModelChoice", "ModelTask", "RunPolicy", "RunRequest", "RunStatus", "SavedPlan", "TaskPlan",
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
    "DEFAULT_MAX_RESPONSE_BYTES",
    "__version__",
]
