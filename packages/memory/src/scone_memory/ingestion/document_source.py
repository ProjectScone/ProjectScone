"""Stable source ownership, distinct from a document's content revision."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from ..core.errors import InvalidInput


@dataclass(frozen=True)
class DocumentSource:
    """A path in a caller-owned collection, interpreted by an explicit parser revision.

    Collection IDs are random UUID hex strings supplied by the owning journal.
    Paths are relative POSIX names; neither case nor Unicode is normalized.
    This identity does not itself authorize deletion or synchronize a directory.
    """

    collection_id: str
    path: str
    parser_revision: str
    generation: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.generation) is not int or not 0 <= self.generation < 2**63:
            raise InvalidInput('document source generation must be a bounded nonnegative integer')
        if not isinstance(self.collection_id, str) or not re.fullmatch(r'[0-9a-f]{32}', self.collection_id):
            raise InvalidInput('document collection_id must be a UUID hex string')
        for name, value, maximum in (('path', self.path, 1024), ('parser_revision', self.parser_revision, 128)):
            if not isinstance(value, str) or not 1 <= len(value) <= maximum:
                raise InvalidInput(f'document source {name} must contain 1..={maximum} characters')
            try:
                value.encode('utf-8')
            except UnicodeError as error:
                raise InvalidInput(f'document source {name} must be valid UTF-8') from error
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise InvalidInput(f'document source {name} cannot contain control characters')
        if '\\' in self.path or any(part in ('', '.', '..') for part in self.path.split('/')):
            raise InvalidInput('document source path must be a canonical relative POSIX path')

    @property
    def path_hash(self) -> str:
        return hashlib.sha256(self.path.encode('utf-8')).hexdigest()

    def metadata(self) -> dict[str, str]:
        return {'source_collection': self.collection_id, 'source_path_hash': self.path_hash,
                'source_parser_revision': self.parser_revision,
                **({'source_generation': str(self.generation)} if self.generation else {})}


def source_revision_key(source: DocumentSource, original_sha256: str, manifest_sha256: str) -> str:
    """Bound retries to the collection, exact path, parser revision and both byte identities."""
    if not isinstance(source, DocumentSource):
        raise InvalidInput('source must be a DocumentSource')
    source.validate()
    if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
           for value in (original_sha256, manifest_sha256)):
        raise InvalidInput('source revision requires original and manifest SHA-256 digests')
    parts: list[str | int] = [source.parser_revision, original_sha256, manifest_sha256]
    if source.generation:
        parts.append(source.generation)
    binding = json.dumps(parts,
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    revision = hashlib.sha256(binding).hexdigest()
    return f'document-source-v1:{source.collection_id}:{source.path_hash}:{revision}'
