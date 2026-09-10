"""Read-only document duplication analysis with traceable source spans."""
from .detector import DocumentDuplicateDetector
from .memory import MemoryCandidateProvider, MemoryDuplicateInspector
from .review import (
    DuplicateNotification, RetainedPassage, ReviewAction, ReviewSource,
    SearchProjection, project_document, review_notification,
)
from .types import (
    CandidateBatch, CandidateProvider, CopiedSpan, DeduplicationConfig,
    DeduplicationMetrics, DocumentRevision, DuplicateReport, PassageEmbedding,
    SemanticCandidate, SemanticMatch,
)

__all__ = [
    "MemoryCandidateProvider", "MemoryDuplicateInspector", "DuplicateNotification",
    "RetainedPassage", "ReviewAction", "ReviewSource", "SearchProjection",
    "project_document", "review_notification",
    "CandidateBatch", "CandidateProvider", "CopiedSpan", "DeduplicationConfig",
    "DeduplicationMetrics", "DocumentDuplicateDetector", "DocumentRevision",
    "DuplicateReport", "PassageEmbedding", "SemanticCandidate", "SemanticMatch",
]
